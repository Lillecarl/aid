"""Embedders over `/v1/embeddings`: what the request carries, what comes back, and how failures read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import pytest
from pydantic import BaseModel, JsonValue

from aid import embedding, wire
from aid.discovery import discover
from aid.embedding import OpenAIEmbedder
from aid.model import check_host
from aid.protocol import AidError
from aid.providers import embedder

if TYPE_CHECKING:
    from collections.abc import Callable

PLUGINS = Path(__file__).parent.parent / "plugins"


class Asked(BaseModel):
    model: str
    input: list[str]
    dimensions: int = 3


type Reply = Callable[[Asked], httpx2.Response]


def answer(monkeypatch: pytest.MonkeyPatch, reply: Reply) -> list[httpx2.Request]:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return reply(Asked.model_validate_json(request.content))

    def make(timeout: float) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), timeout=timeout)

    monkeypatch.setattr(wire, "client", make)
    return seen


def _item(index: int, size: int) -> JsonValue:
    vector: list[JsonValue] = [float(index), 1.0, *[0.0] * (size - 2)]
    return {"object": "embedding", "index": index, "embedding": vector}


def vectors(asked: Asked, *, cost: float | None = None) -> httpx2.Response:
    """One vector per input, reversed in order, as servers may answer: [i, 1, 0, ...] at `dimensions`."""
    data = [_item(i, asked.dimensions) for i in range(len(asked.input))][::-1]
    usage: dict[str, JsonValue] = {"prompt_tokens": 2 * len(asked.input), "total_tokens": 2 * len(asked.input)}
    if cost is not None:
        usage["cost"] = cost
    return httpx2.Response(200, json={"object": "list", "model": asked.model, "data": data, "usage": usage})


def busy(_: Asked) -> httpx2.Response:
    return httpx2.Response(503, json={"error": {"message": "busy"}})


def refused_model(_: Asked) -> httpx2.Response:
    return httpx2.Response(400, json={"error": {"message": "bad model"}})


def refused_key(_: Asked) -> httpx2.Response:
    return httpx2.Response(401, json={"error": {"message": "invalid key"}})


def one_short(asked: Asked) -> httpx2.Response:
    return vectors(asked.model_copy(update={"input": asked.input[:-1]}))


def too_long(asked: Asked) -> httpx2.Response:
    return vectors(asked.model_copy(update={"dimensions": 5}))


async def test_openrouter_embeds_in_order_with_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = answer(monkeypatch, lambda asked: vectors(asked, cost=0.5))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(embedding, "BATCH", 2)
    providers = discover([PLUGINS], only=["openrouter"]).model_providers
    found = await embedder("openrouter/qwen/qwen3-embedding-8b", providers, dimensions=4)
    try:
        result = await found.embed(["a", "b", "c"])
        check_host(found, {}, ["openrouter.ai"])
    finally:
        await found.aclose()
    assert result.vectors == [[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert (result.tokens, result.cost) == (6, 1.0)
    assert [str(r.url) for r in seen] == ["https://openrouter.ai/api/v1/embeddings"] * 2
    first = json.loads(seen[0].content)
    assert first == {
        "model": "qwen/qwen3-embedding-8b",
        "input": ["a", "b"],
        "encoding_format": "float",
        "dimensions": 4,
    }
    assert seen[0].headers["X-Title"] == "aid"


async def test_a_compatible_prefix_needs_no_plugin_and_says_its_host(monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, vectors)
    found = await embedder("vllm/embed", {})
    try:
        assert found.host({}) == "localhost"
        with pytest.raises(AidError) as refused:
            check_host(found, {}, ["model.test"])
        assert refused.value.code == "private_host"
        result = await found.embed(["x"])
    finally:
        await found.aclose()
    assert (result.vectors, result.cost) == ([[0.0, 1.0, 0.0]], None)


async def test_a_provider_without_embeddings_refuses() -> None:
    providers = discover([PLUGINS], only=["deepseek"]).model_providers
    with pytest.raises(AidError) as refused:
        await embedder("deepseek/deepseek-flash", providers)
    assert refused.value.code == "no_embeddings"


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (busy, "model_unavailable"),
        (refused_model, "embedding_failed"),
        (refused_key, "bad_credentials"),
        (one_short, "bad_embedding"),
        (too_long, "bad_embedding"),
    ],
)
async def test_failures_are_named(monkeypatch: pytest.MonkeyPatch, reply: Reply, code: str) -> None:
    answer(monkeypatch, reply)
    found = OpenAIEmbedder("vllm/embed", dimensions=3)
    try:
        with pytest.raises(AidError) as refused:
            await found.embed(["a", "b"])
    finally:
        await found.aclose()
    assert refused.value.code == code
