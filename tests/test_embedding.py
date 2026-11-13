"""Embedders over `/v1/embeddings`: what the request carries, what comes back, and how failures read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import pytest
from pydantic import BaseModel, JsonValue

from aid import embedding, prices, wire
from aid.discovery import discover
from aid.embedding import GoogleEmbedder, OpenAIEmbedder
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


def serve(monkeypatch: pytest.MonkeyPatch, reply: Callable[[httpx2.Request], httpx2.Response]) -> list[httpx2.Request]:
    """Any request to `reply`, kept in order."""
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return reply(request)

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


async def test_openai_reports_tokens_and_models_dev_prices_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI's usage carries no cost (#68): the tokens at the catalog's input price."""
    answer(monkeypatch, vectors)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def price_for(model: str) -> prices.Price | None:
        return prices.Price(0.02, 0, 0, 0) if model == "openai/text-embedding-3-small" else None

    monkeypatch.setattr(prices, "price_for", price_for)
    found = await embedder("openai/text-embedding-3-small", {}, dimensions=3)
    try:
        result = await found.embed(["a", "b"])
        assert found.host({}) == "api.openai.com"
    finally:
        await found.aclose()
    assert (result.tokens, result.cost) == (4, 4 * 0.02 / 1_000_000)


def gemini_vectors(request: httpx2.Request) -> httpx2.Response:
    """[i, 1, 0, ...] at `outputDimensionality` per request; 3 tokens a text, as `gemini-embedding-2` reports."""
    asked = json.loads(request.content)["requests"]
    values = [[float(i), 1.0, *[0.0] * (item["outputDimensionality"] - 2)] for i, item in enumerate(asked)]
    usage = {"promptTokenCount": 3 * len(asked)}
    return httpx2.Response(200, json={"embeddings": [{"values": v} for v in values], "usageMetadata": usage})


async def test_google_embeds_over_the_gemini_api_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch, gemini_vectors)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(embedding, "GOOGLE_BATCH", 2)
    found = await embedder("google/gemini-embedding-2", {}, dimensions=3)
    try:
        result = await found.embed(["a", "b", "c"])
        check_host(found, {}, ["generativelanguage.googleapis.com"])
    finally:
        await found.aclose()
    assert isinstance(found, GoogleEmbedder)
    assert result.vectors == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    assert (result.tokens, result.cost) == (9, None)
    assert [str(r.url) for r in seen] == [
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    ] * 2
    assert seen[0].headers["x-goog-api-key"] == "k"
    assert json.loads(seen[1].content) == {
        "requests": [
            {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": "c"}]}, "outputDimensionality": 3}
        ]
    }


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (400, {"error": {"code": 400, "details": [{"reason": "API_KEY_INVALID"}]}}, "bad_credentials"),
        (400, {"error": {"code": 400, "message": "bad model"}}, "embedding_failed"),
        (503, {"error": {"code": 503}}, "model_unavailable"),
        (200, {"embeddings": [{"values": [1.0, 2.0]}]}, "bad_embedding"),
    ],
    ids=["key", "refused", "busy", "one-short"],
)
async def test_google_failures_are_named(
    monkeypatch: pytest.MonkeyPatch, status: int, body: JsonValue, code: str
) -> None:
    serve(monkeypatch, lambda _: httpx2.Response(status, json=body))
    found = GoogleEmbedder("google/gemini-embedding-001", settings={"api_key": "k", "max_retries": 0})
    try:
        with pytest.raises(AidError) as refused:
            await found.embed(["a", "b"])
    finally:
        await found.aclose()
    assert refused.value.code == code


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
