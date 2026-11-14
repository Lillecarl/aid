"""Model providers: what serves a `provider/model` name.

A plugin offers one by subclassing `aid.ModelProvider` (`aid.agents`); `aid.discovery` collects it as it does
agents. Its `name` is the prefix it claims, and it wins over the generic resolution, which covers:

- `anthropic/`: the Messages API (`AnthropicModel`);
- `google/`: the Gemini API (`GoogleModel`), named as models.dev names it; `gemini/` is its Chat Completions route;
- `openai/` without an `api_base`: the Responses API (`OpenAIResponsesModel`); with one, Chat Completions;
- the prefixes in `aid_models.transport.OPENAI_COMPATIBLE`: servers models.dev gives no URL for, and local ones;
- every models.dev provider whose package is `@ai-sdk/openai-compatible`: its `api`, keyed by its first `env`.

Lifecycle: a provider is built per use and keeps no state. The transport it returns owns its clients, and
whoever resolved it closes them (the native backend). Workers construct providers, never the daemon, which
imports no plugin. A failure in `transport` fails the session's start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from aid.model import AnthropicModel, GoogleModel, OpenAIChatModel, OpenAIResponsesModel, responses_route
from aid.protocol import AidError
from aid_models import models_dev
from aid_models.embedding import Embedder, GoogleEmbedder, OpenAIEmbedder
from aid_models.transport import OPENAI_COMPATIBLE, Provider, split_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from aid.agents import ModelProvider
    from aid.model import Transport
    from aid_models.rerank import Reranker

OPENAI_COMPATIBLE_PACKAGE: Final = "@ai-sdk/openai-compatible"
"""models.dev's package for a provider that serves Chat Completions."""


def catalog_provider(catalog: models_dev.ModelsDev | None, prefix: str) -> Provider | None:
    """A models.dev provider aid can serve as Chat Completions: one with an `api` URL and the compatible package."""
    entry = models_dev.provider(catalog, prefix)
    if entry is None or entry.npm != OPENAI_COMPATIBLE_PACKAGE or not entry.api:
        return None
    return Provider(entry.api, entry.env[0] if entry.env else "")


def claimed(prefix: str, providers: Mapping[str, type[ModelProvider]]) -> type[ModelProvider] | None:
    """The plugin that serves models.dev's `prefix` under another name (`opencode` as `opencode-zen`)."""
    return next((cls for name, cls in providers.items() if name != prefix and cls.catalog_name() == prefix), None)


def catalog_model(name: str, providers: Mapping[str, type[ModelProvider]]) -> str:
    """`name` as models.dev keys it, for prices: the plugin's catalog id in place of its prefix."""
    prefix, model = split_name(name)
    cls = providers.get(prefix)
    return f"{cls.catalog_name()}/{model}" if cls is not None else name


async def resolve(
    name: str,
    providers: Mapping[str, type[ModelProvider]],
    session: str | None = None,
    settings: Mapping[str, JsonValue] | None = None,
) -> Transport:
    """The transport for `name` under the session's model `settings`: the plugin provider that claims its prefix, else
    the generic resolution. A prefix nobody knows still resolves; its first request fails unless the settings name an
    `api_base`."""
    prefix, model = split_name(name)
    if (cls := providers.get(prefix)) is not None:
        return await cls(session).transport(model)
    if (owner := claimed(prefix, providers)) is not None:
        raise AidError("unknown_provider", f"{prefix}/ is served as {owner.provider_name()}/: {owner.description()}")
    if prefix == "anthropic":
        return AnthropicModel(name)
    if prefix == "google":
        return GoogleModel(name)
    if responses_route(name, settings or {}):
        return OpenAIResponsesModel(name)
    if prefix in OPENAI_COMPATIBLE:
        return OpenAIChatModel(name)
    provider = catalog_provider(await models_dev.catalog(), prefix)
    return OpenAIChatModel(name, provider)


async def embedder(
    name: str,
    providers: Mapping[str, type[ModelProvider]],
    *,
    dimensions: int | None = None,
    settings: Mapping[str, JsonValue] | None = None,
) -> Embedder:
    """The embedder for `name`: the plugin provider that claims its prefix, else `google/` (the Gemini API), else an
    `OPENAI_COMPATIBLE` prefix, else any prefix whose `settings` name an `api_base`. The caller closes it."""
    prefix, model = split_name(name)
    if (cls := providers.get(prefix)) is not None:
        return await cls().embedder(model, dimensions, settings or {})
    if prefix == "google":
        return GoogleEmbedder(name, dimensions, settings or {})
    return OpenAIEmbedder(name, dimensions=dimensions, settings=settings or {})


async def reranker(
    name: str, providers: Mapping[str, type[ModelProvider]], *, settings: Mapping[str, JsonValue] | None = None
) -> Reranker:
    """The reranker for `name`: the plugin provider that claims its prefix; nothing else serves one. The caller
    closes it."""
    prefix, model = split_name(name)
    if (cls := providers.get(prefix)) is None:
        raise AidError("no_reranker", f"no provider serves rerankers under {prefix}/")
    return await cls().reranker(model, settings or {})
