"""Model providers: what serves a `provider/model` name.

A plugin offers one by subclassing `aid.ModelProvider` (`aid.agents`); `aid.discovery` collects it as it does
agents. Its `name` is the prefix it claims, and it wins over the generic resolution, which covers:

- `anthropic/`: the Messages API (`AnthropicModel`);
- the prefixes in `aid.model.OPENAI_COMPATIBLE`: servers models.dev gives no URL for, and local ones;
- every models.dev provider whose package is `@ai-sdk/openai-compatible`: its `api`, keyed by its first `env`.

Lifecycle: a provider is built per use and keeps no state. The transport it returns owns its clients, and
whoever resolved it closes them (the native backend). Workers construct providers, never the daemon, which
imports no plugin. A failure in `transport` fails the session's start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from aid import models_dev
from aid.model import OPENAI_COMPATIBLE, AnthropicModel, OpenAIChatModel, Provider, split_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aid.agents import ModelProvider
    from aid.model import Transport

OPENAI_COMPATIBLE_PACKAGE: Final = "@ai-sdk/openai-compatible"
"""models.dev's package for a provider that serves Chat Completions."""


def catalog_provider(catalog: Any, prefix: str) -> Provider | None:
    """A models.dev provider aid can serve as Chat Completions: one with an `api` URL and the compatible package."""
    entry = models_dev.provider(catalog, prefix)
    if entry is None or entry.npm != OPENAI_COMPATIBLE_PACKAGE or not entry.api:
        return None
    return Provider(entry.api, entry.env[0] if entry.env else "")


async def resolve(name: str, providers: Mapping[str, type[ModelProvider]]) -> Transport:
    """The transport for `name`: the plugin provider that claims its prefix, else the generic resolution. A prefix
    nobody knows still resolves; its first request fails unless the settings name an `api_base`."""
    prefix, model = split_name(name)
    if (cls := providers.get(prefix)) is not None:
        return cls().transport(model)
    if prefix == "anthropic":
        return AnthropicModel(name)
    if prefix in OPENAI_COMPATIBLE:
        return OpenAIChatModel(name)
    provider = catalog_provider(await models_dev.catalog(), prefix)
    return OpenAIChatModel(name, provider)
