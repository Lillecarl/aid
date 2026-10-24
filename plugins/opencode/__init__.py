"""OpenCode's two gateways, each a provider of its own: Zen (`opencode-zen/`, pay per use; models.dev's `opencode`)
and Go (`opencode-go/`, the subscription), with separate URLs, model lists and keys. models.dev names each model's protocol and URL
(`provider.npm`, `provider.api`): Chat Completions for most, Anthropic's Messages API for `claude-*`, and OpenAI's
Responses API and Google's Gemini API for others, which aid has no transport for yet.

Unmeasured: no inference call has run through this plugin (no key here). The model lists are public.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

import httpx
from pydantic import BaseModel, Field, ValidationError

import aid
from aid import models_dev
from aid.model import AnthropicModel, OpenAIChatModel, Provider
from aid.protocol import ModelEntry

if TYPE_CHECKING:
    from aid.model import Transport

FETCH_TIMEOUT: Final = 10.0
ANTHROPIC_PACKAGE: Final = "@ai-sdk/anthropic"
UNSUPPORTED: Final = {
    "@ai-sdk/openai": "OpenAI's Responses API, which aid has no transport for",
    "@ai-sdk/google": "Google's Gemini API, which aid has no transport for",
}
"""models.dev's package for a model → why aid cannot run it."""


class _Route(BaseModel):
    npm: str | None = None
    api: str | None = None


class _CatalogModel(BaseModel):
    provider: _Route = Field(default_factory=_Route)


class _Model(BaseModel):
    id: str


class _Models(BaseModel):
    data: list[_Model] = Field(default_factory=list[_Model])


def routes(catalog: Any, provider: str) -> dict[str, _Route]:
    """Each of `provider`'s models' protocol and URL, by id, as models.dev names them."""
    entry = models_dev.provider(catalog, provider)
    found: dict[str, _Route] = {}
    for model_id, raw in (entry.models if entry else {}).items():
        try:
            found[model_id] = _CatalogModel.model_validate(raw).provider
        except ValidationError:
            continue
    return found


class _Gateway(aid.ModelProvider):
    """One OpenCode gateway: its prefix and models.dev id, its key, its Chat Completions URL for a model models.dev
    names no URL for, and its public model list (ids only)."""

    chat_url: ClassVar[str]
    models_url: ClassVar[str]

    async def transport(self, model: str) -> Transport:
        route = routes(await models_dev.catalog(), self.catalog_name()).get(model, _Route())
        full = f"{self.name}/{model}"
        key_env = self.key_env or ""
        if route.npm == ANTHROPIC_PACKAGE:
            # The AI SDK's URL ends in /v1; Anthropic's SDK adds /v1/messages itself.
            base = (route.api or "").removesuffix("/").removesuffix("/v1")
            return AnthropicModel(full, Provider(base or None, key_env))
        if (reason := UNSUPPORTED.get(route.npm or "")) is not None:
            raise aid.AidError("unsupported_model", f"{full} speaks {reason}")
        return OpenAIChatModel(full, Provider(route.api or self.chat_url, key_env))

    async def fetch_models(self) -> list[ModelEntry]:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            response = await client.get(self.models_url)
            response.raise_for_status()
        known = routes(await models_dev.catalog(), self.catalog_name())
        return [
            ModelEntry(id=model.id, unsupported=UNSUPPORTED.get(known.get(model.id, _Route()).npm or ""))
            for model in _Models.model_validate_json(response.content).data
        ]


class OpenCodeZen(_Gateway):
    """OpenCode Zen, pay per use, keyed by OPENCODE_API_KEY."""

    name = "opencode-zen"
    catalog_id = "opencode"
    key_env = "OPENCODE_API_KEY"
    chat_url = "https://opencode.ai/inference/openai/v1"
    models_url = "https://opencode.ai/zen/v1/models"
    """Measured 2026-10-24; `/inference/openai/v1/models` answers 404."""


class OpenCodeGo(_Gateway):
    """OpenCode Go, the subscription, keyed by OPENCODE_GO_API_KEY: its own key, not Zen's."""

    name = "opencode-go"
    catalog_id = "opencode-go"
    key_env = "OPENCODE_GO_API_KEY"
    chat_url = "https://opencode.ai/inference/go/openai/v1"
    models_url = "https://opencode.ai/zen/go/v1/models"
    """Measured 2026-10-24; `/inference/go/openai/v1/models` answers 404."""
