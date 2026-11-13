"""Embedders: text to vectors for `aid.vectors`, through any server that answers OpenAI's `/v1/embeddings` (#60).

`aid.providers.embedder` resolves a `provider/model` name as chat models resolve: a plugin claiming the prefix
(`ModelProvider.embedder`), else a prefix of `aid.model.OPENAI_COMPATIBLE`. Embedders run where their caller
runs (a plugin, a worker), never in the daemon. A private caller checks `host()` against its hosts first
(`aid.model.check_host`), as a private session does for its chat model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

import openai

from aid.model import (
    OPENAI_COMPATIBLE,
    ClientKey,
    Provider,
    client_key,
    refused_key,
    rejected_key,
    split_name,
    unavailable,
)
from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from openai.types import CreateEmbeddingResponse
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
    """`/v1/embeddings` through the OpenAI SDK."""

    model_name: str
    provider: Provider | None = None
    """None: the one `OPENAI_COMPATIBLE` names by the prefix."""
    dimensions: int | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])
    """`api_key`, `api_base`, `max_retries`, as a chat model's `model_settings`."""
    _clients: dict[ClientKey, openai.AsyncOpenAI] = field(default_factory=dict[ClientKey, openai.AsyncOpenAI])

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = OPENAI_COMPATIBLE.get(split_name(self.model_name)[0])

    def _client(self, settings: Mapping[str, JsonValue]) -> openai.AsyncOpenAI:
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
        if (client := self._clients.get(key)) is None:
            client = openai.AsyncOpenAI(
                api_key=key.api_key,
                base_url=key.base_url,
                max_retries=key.max_retries,
                default_headers=dict(provider.headers) if provider else None,
                http_client=openai.DefaultAsyncHttpx2Client(),
            )
            self._clients[key] = client
        return client

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return self._client({**self.settings, **settings}).base_url.host

    async def embed(self, texts: Sequence[str]) -> Embedded:
        """One vector per text, in order. An unreachable or overloaded server raises `model_unavailable`."""
        client = self._client(self.settings)
        model = split_name(self.model_name)[1]
        vectors: list[list[float]] = []
        tokens, cost = 0, None
        for start in range(0, len(texts), BATCH):
            batch = list(texts[start : start + BATCH])
            try:
                response = await client.embeddings.create(
                    model=model,
                    input=batch,
                    # The SDK's default asks for base64, which not every server answers.
                    encoding_format="float",
                    dimensions=self.dimensions if self.dimensions is not None else openai.omit,
                    timeout=TIMEOUT,
                )
            except openai.OpenAIError as error:
                if unavailable(error):
                    raise AidError("model_unavailable", f"{self.model_name}: {error}") from error
                if rejected_key(error):
                    raise refused_key(self.model_name, error) from error
                raise AidError("embedding_failed", f"{self.model_name}: {error}") from error
            vectors.extend(self._vectors(response, len(batch)))
            tokens += response.usage.prompt_tokens if response.usage else 0
            if (spent := _cost(response)) is not None:
                cost = (cost or 0.0) + spent
        return Embedded(vectors, tokens, cost)

    def _vectors(self, response: CreateEmbeddingResponse, count: int) -> list[list[float]]:
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
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.close()


def _cost(response: CreateEmbeddingResponse) -> float | None:
    extra = response.usage.model_extra if response.usage else None
    value = (extra or {}).get("cost")
    return float(value) if isinstance(value, int | float) else None
