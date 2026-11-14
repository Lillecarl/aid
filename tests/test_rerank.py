"""Rerankers (#68): what the request carries, how scores come back, and how failures read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import pytest

from aid.discovery import discover
from aid.protocol import AidError
from aid.providers import reranker
from aid_models.rerank import Ranked, Reranked
from aid_models.transport import check_host
from tests.test_embedding import serve

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

    from aid_models.rerank import Reranker

PLUGINS = Path(__file__).parent.parent / "plugins"


def scored(request: httpx2.Request) -> httpx2.Response:
    """Scores by document length, longest best, in request order (a server need not sort); OpenRouter's usage."""
    asked = json.loads(request.content)
    results: list[dict[str, JsonValue]] = [
        {"index": i, "relevance_score": len(text) / 100, "document": {"text": text}}
        for i, text in enumerate(asked["documents"])
    ]
    if "top_n" in asked:
        results = sorted(results, key=lambda r: -float(str(r["relevance_score"])))[: asked["top_n"]]
    return httpx2.Response(200, json={"results": results, "usage": {"total_tokens": 7, "cost": 0.25}})


async def openrouter(name: str, **settings: JsonValue) -> Reranker:
    return await reranker(name, discover([PLUGINS], only=["openrouter"]).model_providers, settings=settings)


async def test_openrouter_scores_every_document_best_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch, scored)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    found = await openrouter("openrouter/voyageai/rerank-2.5-lite")
    try:
        result = await found.rerank("hash mismatch", ["ab", "abcd", "abc"])
        check_host(found, {}, ["openrouter.ai"])
    finally:
        await found.aclose()
    assert result == Reranked([Ranked(1, 0.04), Ranked(2, 0.03), Ranked(0, 0.02)], 7, 0.25)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/rerank"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert json.loads(seen[0].content) == {
        "model": "voyageai/rerank-2.5-lite",
        "query": "hash mismatch",
        "documents": ["ab", "abcd", "abc"],
    }


async def test_top_n_asks_for_the_best_few(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch, scored)
    found = await openrouter("openrouter/voyageai/rerank-2.5-lite", api_key="k")
    try:
        result = await found.rerank("q", ["a", "abc", "ab"], top_n=2)
        nothing = await found.rerank("q", [])
    finally:
        await found.aclose()
    assert [r.index for r in result.ranked] == [1, 2]
    assert json.loads(seen[0].content)["top_n"] == 2
    assert (nothing, len(seen)) == (Reranked([]), 1)


def reply(status: int, body: JsonValue) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda _: httpx2.Response(status, json=body)


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (reply(401, {"error": {"message": "invalid key"}}), "bad_credentials"),
        (reply(400, {"error": {"message": "bad model"}}), "rerank_failed"),
        (reply(503, {"error": {"message": "busy"}}), "model_unavailable"),
        (
            reply(200, {"results": [{"index": 5, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0}]}),
            "bad_rerank",
        ),
        (reply(200, {"results": [{"index": 0, "relevance_score": 0.1}]}), "bad_rerank"),
        (reply(200, {"results": "none"}), "bad_rerank"),
    ],
    ids=["key", "refused", "busy", "out-of-range", "one-short", "malformed"],
)
async def test_failures_are_named(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[httpx2.Request], httpx2.Response], code: str
) -> None:
    serve(monkeypatch, answer)
    found = await openrouter("openrouter/cohere/rerank-v3.5", api_key="k", max_retries=0)
    try:
        with pytest.raises(AidError) as refused:
            await found.rerank("q", ["a", "b"])
    finally:
        await found.aclose()
    assert refused.value.code == code


@pytest.mark.parametrize("name", ["deepseek/deepseek-flash", "openai/rerank"])
async def test_a_prefix_without_rerankers_refuses(name: str) -> None:
    with pytest.raises(AidError) as refused:
        await reranker(name, discover([PLUGINS], only=["deepseek"]).model_providers)
    assert refused.value.code == "no_reranker"
