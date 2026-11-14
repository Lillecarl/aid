"""Embedders: text to vectors for `aid.vectors` (#60), over aid's own client (#65): any server that answers OpenAI's
`/v1/embeddings`, and the Gemini API's `batchEmbedContents` (#68).

`aid.providers.embedder` resolves a `provider/model` name as chat models resolve: a plugin claiming the prefix
(`ModelProvider.embedder`), else `google/`, else a prefix of `aid_models.transport.OPENAI_COMPATIBLE`. Embedders run where their caller
runs (a plugin, a worker), never in the daemon. A private caller checks `host()` against its hosts first
(`aid_models.transport.check_host`), as a private session does for its chat model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

import httpx2
from pydantic import ValidationError

from aid_models import chat_api, gemini, prices, wire
from aid_models.errors import AidError
from aid_models.transport import (
    GOOGLE_KEY_ENV,
    OPENAI_COMPATIBLE,
    ClientKey,
    Provider,
    client_key,
    model_answer,
    split_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pydantic import JsonValue

BATCH: Final = 128
"""Texts per request: OpenAI takes up to 2048, other servers fewer."""
GOOGLE_BATCH: Final = 100
"""`batchEmbedContents` takes at most 100 requests."""
TIMEOUT: Final = 120.0


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    tokens: int = 0
    cost: float | None = None
    """What the provider says the requests cost (OpenRouter's `usage.cost`), else the tokens at models.dev's price;
    None when neither is known."""


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
    """`/v1/embeddings` over aid's own client (`aid_models.chat_api`, `aid_models.wire`; #65)."""

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
            body = await _post(self.model_name, self._http, url, json.dumps(asked).encode(), headers, key.max_retries)
            try:
                answer = chat_api.Embeddings.model_validate_json(body)
            except ValidationError as error:
                raise AidError("bad_embedding", f"{self.model_name}: {error}") from error
            vectors.extend(self._vectors(answer, len(batch)))
            tokens += answer.usage.prompt_tokens if answer.usage else 0
            if (spent := _cost(answer)) is not None:
                cost = (cost or 0.0) + spent
        return Embedded(vectors, tokens, await prices.input_cost(self.model_name, tokens, cost))

    def _vectors(self, response: chat_api.Embeddings, count: int) -> list[list[float]]:
        found = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in found] != list(range(len(found))):
            raise AidError("bad_embedding", f"{self.model_name} answered vectors out of order")
        return _checked(self.model_name, [item.embedding for item in found], count, self.dimensions)

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


@dataclass
class GoogleEmbedder:
    """The Gemini API's `batchEmbedContents` over aid's own client (`aid_models.gemini`; #68)."""

    model_name: str
    dimensions: int | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])
    provider: Provider | None = None
    """Another server of the Gemini API; None is Google AI Studio (`GEMINI_API_KEY`)."""
    _http: httpx2.AsyncClient | None = field(default=None, repr=False, compare=False)

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self.provider
        key_env = provider.key_env if provider else GOOGLE_KEY_ENV
        return client_key(self.model_name, settings, key_env, provider.base_url if provider else None)

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        model = split_name(self.model_name)[1]
        return httpx2.URL(gemini.embed_url(self._key({**self.settings, **settings}).base_url, model)).host

    async def embed(self, texts: Sequence[str]) -> Embedded:
        key = self._key(self.settings)
        if self._http is None:
            self._http = wire.client(timeout=TIMEOUT)
        headers = {gemini.KEY_HEADER: key.api_key, **(self.provider.headers if self.provider else {})}
        model = split_name(self.model_name)[1]
        url = gemini.embed_url(key.base_url, model)
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), GOOGLE_BATCH):
            batch = texts[start : start + GOOGLE_BATCH]
            request = gemini.BatchEmbedRequest(
                requests=[
                    gemini.EmbedContentRequest(
                        model=f"models/{model}",
                        content=gemini.Content(parts=[gemini.Part(text=text)]),
                        output_dimensionality=self.dimensions,
                    )
                    for text in batch
                ]
            )
            body = await _post(
                self.model_name,
                self._http,
                url,
                request.body(),
                headers,
                key.max_retries,
                lambda error: gemini.refused_key(error.body),
            )
            try:
                answer = gemini.BatchEmbedResponse.model_validate_json(body)
            except ValidationError as error:
                raise AidError("bad_embedding", f"{self.model_name}: {error}") from error
            vectors.extend(
                _checked(self.model_name, [e.values for e in answer.embeddings], len(batch), self.dimensions)
            )
            tokens += answer.usage_metadata.prompt_token_count if answer.usage_metadata else 0
        return Embedded(vectors, tokens, await prices.input_cost(self.model_name, tokens, None))

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


async def _post(
    name: str,
    http: httpx2.AsyncClient,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    retries: int,
    refused: Callable[[wire.StatusError], bool] = lambda error: False,
) -> bytes:
    return await model_answer(name, http, url, body, headers, TIMEOUT, retries, "embedding_failed", refused)


def _checked(name: str, found: list[list[float]], count: int, dimensions: int | None) -> list[list[float]]:
    if len(found) != count:
        raise AidError("bad_embedding", f"{name} answered {len(found)} vectors for {count} texts")
    lengths = {len(vector) for vector in found}
    if len(lengths) > 1 or (dimensions is not None and lengths != {dimensions}):
        raise AidError("bad_embedding", f"{name} answered vectors of {sorted(lengths)} dimensions")
    if not all(math.isfinite(x) for vector in found for x in vector):
        raise AidError("bad_embedding", f"{name} answered a vector that is not finite")
    return found


def _cost(response: chat_api.Embeddings) -> float | None:
    extra = response.usage.model_extra if response.usage else None
    value = (extra or {}).get("cost")
    return float(value) if isinstance(value, int | float) else None
