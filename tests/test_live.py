"""Live tests (#64): each provider route's real server, on its cheapest tool-calling model.

Off unless `AID_LIVE_TESTS=1`; a route without its key skips. They spend money: a full run costs about a cent.
`nix develop --file . shell --command env AID_LIVE_TESTS=1 python -m pytest -p no:cacheprovider -m live tests/test_live.py`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from aid.agent import Agent
from aid.discovery import discover
from aid.images import ImageStore
from aid.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from aid.model import ModelInfo
from aid.prices import ENV_PRICES
from aid.protocol import AidError, Output, SessionEvent, Usage
from aid.providers import catalog_model, embedder, resolve
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
    (usage,) = [item for item in events if isinstance(item, Usage)]
    if route.priced:
        assert usage.cost is not None
        assert 0 < usage.cost.amount < TURN_BOUND


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


async def test_the_openrouter_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    keyed(monkeypatch, "OPENROUTER_API_KEY")
    providers = discover([PLUGINS], only=["openrouter"]).model_providers
    found = await embedder("openrouter/qwen/qwen3-embedding-8b", providers, dimensions=256)
    try:
        result = await found.embed(["a fetchurl hash mismatch", "the hash of a fixed-output derivation changed"])
    finally:
        await found.aclose()
    assert [len(vector) for vector in result.vectors] == [256, 256]
    assert result.tokens > 0
    assert result.cost is not None
    assert result.cost < TURN_BOUND
