"""OpenRouter (`openrouter/` models, named `vendor/model` after the prefix): Chat Completions with OpenRouter's
usage accounting and reasoning field.

Unmeasured: no inference call has run through this plugin (no key here). The model list is public (measured
2026-10-24: 469 models, prices per token as strings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, overload

import httpx
from pydantic import BaseModel, Field, JsonValue

import aid
from aid.model import OpenAIChatModel, Provider
from aid.protocol import AidError, ModelCost, ModelEntry

if TYPE_CHECKING:
    from collections.abc import Mapping


KEY_ENV: Final = "OPENROUTER_API_KEY"
PROVIDER: Final = Provider("https://openrouter.ai/api/v1", KEY_ENV, headers={"X-Title": "aid"})
MODELS_URL: Final = "https://openrouter.ai/api/v1/models"
FETCH_TIMEOUT: Final = 10.0
PER: Final = 1_000_000
"""OpenRouter prices per token; aid's per million."""


@overload
def per_million(price: float) -> float: ...
@overload
def per_million(price: None) -> None: ...
def per_million(price: float | None) -> float | None:
    """Rounded: per-token strings times a million leave float noise (0.19999999999999998)."""
    return None if price is None else round(price * PER, 9)


class _Pricing(BaseModel):
    prompt: float | None = None
    completion: float | None = None
    input_cache_read: float | None = None
    input_cache_write: float | None = None


class _TopProvider(BaseModel):
    max_completion_tokens: int | None = None


class _Model(BaseModel):
    id: str
    name: str | None = None
    context_length: int | None = None
    pricing: _Pricing = Field(default_factory=_Pricing)
    top_provider: _TopProvider = Field(default_factory=_TopProvider)
    supported_parameters: list[str] | None = None

    def entry(self) -> ModelEntry:
        price = self.pricing
        cost = (
            ModelCost(
                input=per_million(price.prompt),
                output=per_million(price.completion),
                cache_read=per_million(price.input_cache_read),
                cache_write=per_million(price.input_cache_write),
            )
            # Negative: a router model, priced by the model it picks.
            if price.prompt is not None and price.completion is not None and min(price.prompt, price.completion) >= 0
            else None
        )
        parameters = self.supported_parameters
        return ModelEntry(
            id=self.id,
            name=self.name,
            context=self.context_length,
            output=self.top_provider.max_completion_tokens,
            reasoning=None if parameters is None else "reasoning" in parameters,
            tools=None if parameters is None else "tools" in parameters,
            cost=cost,
        )


class _Models(BaseModel):
    data: list[_Model] = Field(default_factory=list[_Model])


class OpenRouterChat(OpenAIChatModel):
    """Asks for OpenRouter's usage accounting and replays no reasoning text: OpenRouter takes reasoning back only with
    the `reasoning_details` aid does not keep."""

    replays_reasoning = False

    def request_settings(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        kwargs = super().request_settings(settings)
        extra = kwargs.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise AidError("bad_model_settings", f"{self.model_name}: extra_body must be an object")
        kwargs["extra_body"] = {"usage": {"include": True}, **extra}
        return kwargs


class OpenRouter(aid.ModelProvider):
    """OpenRouter, keyed by OPENROUTER_API_KEY."""

    name = "openrouter"
    key_env = KEY_ENV

    async def transport(self, model: str) -> OpenRouterChat:
        return OpenRouterChat(f"{self.name}/{model}", PROVIDER)

    async def fetch_models(self) -> list[ModelEntry]:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            response = await client.get(MODELS_URL)
            response.raise_for_status()
        return [model.entry() for model in _Models.model_validate_json(response.content).data]
