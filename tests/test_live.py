"""Live tests (#64): each provider route's real server, on its cheapest tool-calling model.

Off unless `AID_LIVE_TESTS=1`; a route without its key skips. They spend money: a full run costs about a cent.
`nix develop --file . shell --command env AID_LIVE_TESTS=1 python -m pytest -p no:cacheprovider -m live tests/test_live.py`
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from aid.agent import Agent
from aid.coding import read
from aid.discovery import discover
from aid.images import IMAGES_DIR, UPLOADS, UPLOADS_DIR, ImageStore
from aid.messages import ImagePart, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from aid.model import ModelInfo
from aid.prices import ENV_PRICES
from aid.protocol import AidError, Output, SessionEvent, Usage
from aid.providers import catalog_model, embedder, reranker, resolve
from tests.test_images import png
from tests.test_native_backend import backend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

PLUGINS: Final = Path(__file__).parent.parent / "plugins"
TURN_BOUND: Final = 0.01
"""Dollars one turn may cost; a turn over it means a wrong model or a runaway loop."""

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("AID_LIVE_TESTS") != "1", reason="live tests spend money: AID_LIVE_TESTS=1"),
    pytest.mark.timeout(180),
]


@dataclass(frozen=True)
class Route:
    model: str
    key: str
    """The variable the route reads; the test skips without it."""
    plugin: str | None = None
    priced: bool = True
    """models.dev prices it: the turn's Usage carries a cost."""
    settings: Mapping[str, JsonValue] | None = None
    source_key: str | None = None
    """Where the key lives when not under `key` (`CLAUDE_API_KEY` in .privrc)."""
    vision: str | None = None
    """The model the image test runs on, where `model` takes text only (models.dev's `modalities.input`)."""


ROUTES: Final = [
    Route("openai/gpt-5-nano", "OPENAI_API_KEY"),
    Route("anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY", source_key="CLAUDE_API_KEY"),
    Route("google/gemini-3.1-flash-lite", "GEMINI_API_KEY"),
    Route("deepseek/deepseek-flash", "DEEPSEEK_API_KEY", plugin="deepseek"),
    Route("opencode-zen/deepseek-v4.1-flash", "OPENCODE_API_KEY", plugin="opencode"),
    Route("opencode-zen/claude-haiku-5-5", "OPENCODE_API_KEY", plugin="opencode"),
    Route("opencode-go/deepseek-v4.1-flash", "OPENCODE_GO_API_KEY", plugin="opencode"),
    Route(
        "openrouter/deepseek/deepseek-v4-flash",
        "OPENROUTER_API_KEY",
        plugin="openrouter",
        vision="openrouter/openai/gpt-5-nano",
    ),
]

INSTRUCTIONS: Final = "You are a test. Do all arithmetic with the add tool, never in your head. Answer with the number."


def keyed(monkeypatch: pytest.MonkeyPatch, key: str, source: str | None = None) -> None:
    value = os.environ.get(key) or (os.environ.get(source) if source else None)
    if not value:
        pytest.skip(f"no {key}")
    monkeypatch.setenv(key, value)


@pytest.mark.parametrize("route", ROUTES, ids=[route.model for route in ROUTES])
async def test_a_tool_loop_over_two_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    """A tool call and its answer, then a second turn over the replayed history: reasoning and signatures go back as
    the server sent them, or it refuses."""
    keyed(monkeypatch, route.key, route.source_key)
    monkeypatch.delenv(ENV_PRICES)
    providers = discover([PLUGINS], only=[route.plugin]).model_providers if route.plugin else {}
    calls: list[tuple[int, int]] = []

    async def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent(route.model, INSTRUCTIONS, tools=[add], model_settings=dict(route.settings or {}))
    # The session name, as `open_native` passes it: OpenCode Go refuses a request without one.
    transport = await resolve(route.model, providers, tmp_path.name)
    async with backend(tmp_path, agent, transport=transport, price_as=catalog_model(route.model, providers)) as session:
        first = await session.prompt("What is 17 plus 25?", emit)
        second = await session.prompt("Now add 100 to that.", emit)
    assert isinstance(first, Output)
    assert "42" in str(first.output)
    assert "142" in str(second.output)
    assert (17, 25) in calls or (25, 17) in calls
    assert (42, 100) in calls or (100, 42) in calls
    history = ModelMessagesTypeAdapter.validate_json((tmp_path / "history.json").read_bytes())
    called = [p for m in history if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)]
    assert len(called) >= 2
    usages = [item for item in events if isinstance(item, Usage)]
    assert len(usages) == 2
    for usage in usages:
        assert usage.input_tokens and usage.output_tokens
        if route.priced:
            assert usage.cost is not None
            assert 0 < usage.cost.amount < TURN_BOUND


@pytest.mark.parametrize("route", ROUTES, ids=[route.model for route in ROUTES])
async def test_an_image_in_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    """The model sees the image the prompt carries (#66). The prompt names the choices: asked "what colour", DeepSeek
    V4 answered "white" for a red square in about one of twenty runs (2026-11-13); named, 28 of 28 read it."""
    keyed(monkeypatch, route.key, route.source_key)
    monkeypatch.delenv(ENV_PRICES)
    providers = discover([PLUGINS], only=[route.plugin]).model_providers if route.plugin else {}
    name = route.vision or route.model
    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent(name, "You are a test.", model_settings=dict(route.settings or {}))
    transport = await resolve(name, providers, tmp_path.name)
    async with backend(
        tmp_path, agent, transport=transport, price_as=catalog_model(name, providers), images=ImageStore(tmp_path)
    ) as session:
        answer = await session.prompt(
            "The image is one solid colour. Is it red, green or blue? Answer with one word.",
            emit,
            [png((255, 0, 0), size=256)],
        )
    assert "red" in str(answer.output).lower()
    assert_by_file_id(tmp_path, name)
    (usage,) = [item for item in events if isinstance(item, Usage)]
    if route.priced:
        assert usage.cost is not None
        assert 0 < usage.cost.amount < TURN_BOUND


@pytest.mark.parametrize("route", ROUTES, ids=[route.model for route in ROUTES])
async def test_an_image_from_a_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    """The model sees the image `read` hands back (#66): inside the tool result (Anthropic), after the tool messages
    (Chat Completions), as a user part beside the function responses (Gemini)."""
    keyed(monkeypatch, route.key, route.source_key)
    monkeypatch.delenv(ENV_PRICES)
    providers = discover([PLUGINS], only=[route.plugin]).model_providers if route.plugin else {}
    name = route.vision or route.model
    (tmp_path / "pic.png").write_bytes(png((0, 255, 0), size=256))
    agent = Agent(name, "You are a test.", tools=[read], model_settings=dict(route.settings or {}))
    transport = await resolve(name, providers, tmp_path.name)
    async with backend(
        tmp_path, agent, transport=transport, price_as=catalog_model(name, providers), images=ImageStore(tmp_path)
    ) as session:
        answer = await session.prompt(
            "Read pic.png with the read tool. The image is one solid colour. Is it red, green or blue? Answer with one"
            " word.",
            emit_nothing,
        )
    assert "green" in str(answer.output).lower()
    history = ModelMessagesTypeAdapter.validate_json((tmp_path / "history.json").read_bytes())
    assert any(isinstance(p, ImagePart) and p.tool_call_id for m in history for p in m.parts)
    assert_by_file_id(tmp_path, name)


def assert_by_file_id(root: Path, model: str) -> None:
    """Anthropic's own route sends images by Files API id (#69): the session's id table names one."""
    if model.startswith("anthropic/"):
        tables = list((root / IMAGES_DIR / UPLOADS_DIR).glob("*.json"))
        assert len(tables) == 1
        assert any(file_id.startswith("file_") for file_id in UPLOADS.validate_json(tables[0].read_bytes()).values())


async def emit_nothing(item: SessionEvent) -> None:
    pass


@pytest.mark.parametrize("route", ROUTES, ids=[route.model for route in ROUTES])
async def test_a_refused_key_is_named(route: Route) -> None:
    """Free: the server refuses before any model runs."""
    providers = discover([PLUGINS], only=[route.plugin]).model_providers if route.plugin else {}
    model = await resolve(route.model, providers, "live-test")
    try:
        with pytest.raises(AidError) as refused:
            await anext(
                model.stream([ModelRequest(parts=[UserPromptPart("hi")])], ModelInfo(settings={"api_key": "refused"}))
            )
    finally:
        await model.aclose()
    assert refused.value.code == "bad_credentials"


EMBEDDERS: Final = [
    Route("openrouter/qwen/qwen3-embedding-8b", "OPENROUTER_API_KEY", plugin="openrouter"),
    Route("openai/text-embedding-3-small", "OPENAI_API_KEY"),
    Route("google/gemini-embedding-2", "GEMINI_API_KEY"),
    # Reports no tokens (measured 2026-11-13), so nothing prices it.
    Route("google/gemini-embedding-001", "GEMINI_API_KEY", priced=False),
]
TEXTS: Final = ["a fetchurl hash mismatch", "the hash of a fixed-output derivation changed", "a recipe for pancakes"]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


@pytest.mark.parametrize("route", EMBEDDERS, ids=[route.model for route in EMBEDDERS])
async def test_an_embedder(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    """Vectors of the asked length, nearer for the two texts about one thing (#60, #68)."""
    keyed(monkeypatch, route.key)
    monkeypatch.delenv(ENV_PRICES)
    providers = discover([PLUGINS], only=[route.plugin]).model_providers if route.plugin else {}
    found = await embedder(route.model, providers, dimensions=256)
    try:
        result = await found.embed(TEXTS)
    finally:
        await found.aclose()
    hash_a, hash_b, pancakes = result.vectors
    assert [len(vector) for vector in result.vectors] == [256] * 3
    assert cosine(hash_a, hash_b) > cosine(hash_a, pancakes)
    if route.priced:
        assert result.tokens > 0
        assert result.cost is not None
        assert 0 < result.cost < TURN_BOUND
    else:
        assert result.cost is None


RERANKERS: Final = ["openrouter/voyageai/rerank-2.5-lite", "openrouter/cohere/rerank-v3.5"]


@pytest.mark.parametrize("name", RERANKERS)
async def test_a_reranker(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    keyed(monkeypatch, "OPENROUTER_API_KEY")
    found = await reranker(name, discover([PLUGINS], only=["openrouter"]).model_providers)
    try:
        result = await found.rerank("why did my nix build fail with a hash mismatch?", TEXTS)
        best = await found.rerank("pancakes", TEXTS, top_n=1)
    finally:
        await found.aclose()
    assert sorted(r.index for r in result.ranked) == [0, 1, 2]
    assert result.ranked[-1].index == 2
    assert [r.index for r in best.ranked] == [2]
    assert result.cost is not None
    assert 0 < result.cost < TURN_BOUND
