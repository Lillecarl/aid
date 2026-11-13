"""Embedders: text to vectors for `aid.vectors`, through any server that answers OpenAI's `/v1/embeddings` (#60), over aid's own client (#65).

`aid.providers.embedder` resolves a `provider/model` name as chat models resolve: a plugin claiming the prefix
(`ModelProvider.embedder`), else a prefix of `aid.model.OPENAI_COMPATIBLE`. Embedders run where their caller
runs (a plugin, a worker), never in the daemon. A private caller checks `host()` against its hosts first
(`aid.model.check_host`), as a private session does for its chat model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

import httpx2
from pydantic import ValidationError

from aid import chat_api, wire
from aid.model import (
    OPENAI_COMPATIBLE,
    ClientKey,
    ModelUnavailable,
    Provider,
    client_key,
    open_model_stream,
    split_name,
)
from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

BATCH: Final = 128
"""Texts per request: OpenAI takes up to 2048, other servers fewer."""
TIMEOUT: Final = 120.0


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    tokens: int = 0
    cost: float | None = None
    """What the provider says the requests cost (OpenRouter's `usage.cost`); None when it says nothing."""


class Embedder(Protocol):
    model_name: str
    """`provider/model`: what a `VectorTable.model` records, with `dimensions`."""
    dimensions: int | None
    """Asked of the server (models that shorten their vectors); None takes the model's own length."""

    async def embed(self, texts: Sequence[str]) -> Embedded: ...
    def host(self, settings: Mapping[str, JsonValue]) -> str: ...
    async def aclose(self) -> None: ...


@dataclass
class OpenAIEmbedder:
    """`/v1/embeddings` over aid's own client (`aid.chat_api`, `aid.wire`; #65)."""

    model_name: str
    provider: Provider | None = None
    """None: the one `OPENAI_COMPATIBLE` names by the prefix."""
    dimensions: int | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])
    """`api_key`, `api_base`, `max_retries`, as a chat model's `model_settings`."""
    _http: httpx2.AsyncClient | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = OPENAI_COMPATIBLE.get(split_name(self.model_name)[0])

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self.provider
        key = client_key(
            self.model_name,
            settings,
            provider.key_env if provider else None,
            provider.base_url if provider else None,
            local=provider is not None and provider.local,
        )
        if key.base_url is None and provider is None:
            raise AidError("unknown_provider", f"no provider for {self.model_name}: set api_base")
        return key

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return httpx2.URL(chat_api.url(self._key({**self.settings, **settings}).base_url, "embeddings")).host

    async def embed(self, texts: Sequence[str]) -> Embedded:
        """One vector per text, in order. An unreachable or overloaded server raises `model_unavailable`."""
        key = self._key(self.settings)
        if self._http is None:
            self._http = wire.client(timeout=TIMEOUT)
        headers = {"authorization": f"Bearer {key.api_key}", **(self.provider.headers if self.provider else {})}
        url = chat_api.url(key.base_url, "embeddings")
        model = split_name(self.model_name)[1]
        vectors: list[list[float]] = []
        tokens, cost = 0, None
        for start in range(0, len(texts), BATCH):
            batch = list(texts[start : start + BATCH])
            asked: dict[str, JsonValue] = {"model": model, "input": [*batch], "encoding_format": "float"}
            if self.dimensions is not None:
                asked["dimensions"] = self.dimensions
            try:
                response = await open_model_stream(
                    self.model_name, self._http, url, json.dumps(asked).encode(), headers, TIMEOUT, key.max_retries
                )
                try:
                    body = await response.aread()
                finally:
                    await response.aclose()
            except ModelUnavailable as error:
                raise AidError("model_unavailable", f"{self.model_name}: {error}") from error
            except wire.StatusError as error:
                raise AidError("embedding_failed", f"{self.model_name}: {error}") from error
            try:
                answer = chat_api.Embeddings.model_validate_json(body)
            except ValidationError as error:
                raise AidError("bad_embedding", f"{self.model_name}: {error}") from error
            vectors.extend(self._vectors(answer, len(batch)))
            tokens += answer.usage.prompt_tokens if answer.usage else 0
            if (spent := _cost(answer)) is not None:
                cost = (cost or 0.0) + spent
        return Embedded(vectors, tokens, cost)

    def _vectors(self, response: chat_api.Embeddings, count: int) -> list[list[float]]:
        found = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in found] != list(range(count)):
            raise AidError("bad_embedding", f"{self.model_name} answered {len(found)} vectors for {count} texts")
        lengths = {len(item.embedding) for item in found}
        if len(lengths) > 1 or (self.dimensions is not None and lengths != {self.dimensions}):
            raise AidError("bad_embedding", f"{self.model_name} answered vectors of {sorted(lengths)} dimensions")
        if not all(math.isfinite(x) for item in found for x in item.embedding):
            raise AidError("bad_embedding", f"{self.model_name} answered a vector that is not finite")
        return [item.embedding for item in found]

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


def _cost(response: chat_api.Embeddings) -> float | None:
    extra = response.usage.model_extra if response.usage else None
    value = (extra or {}).get("cost")
    return float(value) if isinstance(value, int | float) else None
