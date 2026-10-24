"""OpenCode Zen (`opencode/` models): one key, several protocols. models.dev names each model's protocol and URL
(`provider.npm`, `provider.api`): Chat Completions for most, Anthropic's Messages API for `claude-*`, and OpenAI's
Responses API for `gpt-*`, which aid has no transport for yet.

Unmeasured: no inference call has run through this plugin (no key here). The model list is public.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel, Field, ValidationError

import aid
from aid import models_dev
from aid.model import AnthropicModel, OpenAIChatModel, Provider
from aid.protocol import ModelEntry

if TYPE_CHECKING:
    from aid.model import Transport

NAME: Final = "opencode"
KEY_ENV: Final = "OPENCODE_API_KEY"
CHAT_URL: Final = "https://opencode.ai/inference/openai/v1"
"""models.dev's provider-level `api`, for a model it names no URL of its own for."""
MODELS_URL: Final = "https://opencode.ai/zen/v1/models"
"""Public, ids only (measured 2026-10-24); `/inference/openai/v1/models` answers 404."""
FETCH_TIMEOUT: Final = 10.0
ANTHROPIC_PACKAGE: Final = "@ai-sdk/anthropic"
RESPONSES_PACKAGE: Final = "@ai-sdk/openai"
UNSUPPORTED: Final = "OpenAI's Responses API, which aid has no transport for"


class _Route(BaseModel):
    npm: str | None = None
    api: str | None = None


class _CatalogModel(BaseModel):
    provider: _Route = Field(default_factory=_Route)


class _Model(BaseModel):
    id: str


class _Models(BaseModel):
    data: list[_Model] = Field(default_factory=list[_Model])


def routes(catalog: Any) -> dict[str, _Route]:
    """Each Zen model's protocol and URL, by id, as models.dev names them."""
    entry = models_dev.provider(catalog, NAME)
    found: dict[str, _Route] = {}
    for model_id, raw in (entry.models if entry else {}).items():
        try:
            found[model_id] = _CatalogModel.model_validate(raw).provider
        except ValidationError:
            continue
    return found


class OpenCode(aid.ModelProvider):
    """OpenCode Zen, keyed by OPENCODE_API_KEY."""

    name = NAME
    key_env = KEY_ENV

    async def transport(self, model: str) -> Transport:
        route = routes(await models_dev.catalog()).get(model, _Route())
        full = f"{NAME}/{model}"
        if route.npm == ANTHROPIC_PACKAGE:
            # The AI SDK's URL ends in /v1; Anthropic's SDK adds /v1/messages itself.
            base = (route.api or "").removesuffix("/").removesuffix("/v1")
            return AnthropicModel(full, Provider(base or None, KEY_ENV))
        if route.npm == RESPONSES_PACKAGE:
            raise aid.AidError("unsupported_model", f"{full} speaks {UNSUPPORTED}")
        return OpenAIChatModel(full, Provider(route.api or CHAT_URL, KEY_ENV))

    async def fetch_models(self) -> list[ModelEntry]:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            response = await client.get(MODELS_URL)
            response.raise_for_status()
        known = routes(await models_dev.catalog())
        return [
            ModelEntry(
                id=model.id,
                unsupported=UNSUPPORTED if known.get(model.id, _Route()).npm == RESPONSES_PACKAGE else None,
            )
            for model in _Models.model_validate_json(response.content).data
        ]
