"""Rerankers (#68): a query and documents → each document's relevance score, over aid's own client (#65).

The wire is Cohere's `/rerank` as OpenRouter serves it for Voyage AI and Cohere models (measured 2026-11-13:
`results[{index, relevance_score, document}]`, `usage{total_tokens, cost}`). `aid.providers.reranker` resolves a
`provider/model` name through the plugin claiming its prefix (`ModelProvider.reranker`). Rerankers run where their
caller runs, never in the daemon; a private caller checks `host()` first (`aid_models.transport.check_host`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

import httpx2
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from aid_models import chat_api, prices, wire
from aid_models.errors import AidError
from aid_models.transport import client_key, model_answer, split_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aid_models.transport import ClientKey, Provider

TIMEOUT: Final = 120.0


@dataclass(frozen=True)
class Ranked:
    index: int
    """The document's position in the request."""
    score: float


@dataclass(frozen=True)
class Reranked:
    ranked: list[Ranked]
    """Best first."""
    tokens: int = 0
    cost: float | None = None
    """The provider's own cost, else the tokens at models.dev's price; None when neither is known."""


class Reranker(Protocol):
    model_name: str

    async def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> Reranked:
        """`top_n`: the best that many; None scores every document."""
        ...

    def host(self, settings: Mapping[str, JsonValue]) -> str: ...
    async def aclose(self) -> None: ...


class _Result(BaseModel):
    index: int
    relevance_score: float


class _Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_tokens: int = 0
    cost: float | None = None


class RerankAnswer(BaseModel):
    results: list[_Result]
    usage: _Usage | None = None


@dataclass
class RerankClient:
    """`POST <base>/rerank`: `{model, query, documents, top_n}` → a score per document."""

    model_name: str
    provider: Provider
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])
    """`api_key`, `api_base`, `max_retries`, as a chat model's `model_settings`."""
    _http: httpx2.AsyncClient | None = field(default=None, repr=False, compare=False)

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self.provider
        return client_key(self.model_name, settings, provider.key_env, provider.base_url, local=provider.local)

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return httpx2.URL(chat_api.url(self._key({**self.settings, **settings}).base_url, "rerank")).host

    async def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> Reranked:
        if not documents:
            return Reranked([])
        key = self._key(self.settings)
        if self._http is None:
            self._http = wire.client(timeout=TIMEOUT)
        asked: dict[str, JsonValue] = {
            "model": split_name(self.model_name)[1],
            "query": query,
            "documents": [*documents],
        }
        if top_n is not None:
            asked["top_n"] = top_n
        headers = {"authorization": f"Bearer {key.api_key}", **self.provider.headers}
        body = await model_answer(
            self.model_name,
            self._http,
            chat_api.url(key.base_url, "rerank"),
            json.dumps(asked).encode(),
            headers,
            TIMEOUT,
            key.max_retries,
            "rerank_failed",
        )
        try:
            answer = RerankAnswer.model_validate_json(body)
        except ValidationError as error:
            raise AidError("bad_rerank", f"{self.model_name}: {error}") from error
        ranked = self._ranked(answer, len(documents), top_n)
        tokens = answer.usage.total_tokens if answer.usage else 0
        cost = await prices.input_cost(self.model_name, tokens, answer.usage.cost if answer.usage else None)
        return Reranked(ranked, tokens, cost)

    def _ranked(self, answer: RerankAnswer, count: int, top_n: int | None) -> list[Ranked]:
        indices = [result.index for result in answer.results]
        if len(set(indices)) != len(indices) or not all(0 <= i < count for i in indices):
            raise AidError("bad_rerank", f"{self.model_name} answered indices {indices} for {count} documents")
        if len(indices) != min(count, top_n if top_n is not None else count):
            raise AidError("bad_rerank", f"{self.model_name} answered {len(indices)} of {count} documents")
        if not all(math.isfinite(result.relevance_score) for result in answer.results):
            raise AidError("bad_rerank", f"{self.model_name} answered a score that is not finite")
        ranked = [Ranked(result.index, result.relevance_score) for result in answer.results]
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()
