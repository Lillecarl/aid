"""DeepSeek's API (`deepseek/` models): Chat Completions with DeepSeek's cache counting and output limit."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, Field

import aid
from aid.model import OpenAIChatModel, Provider, RequestUsage
from aid.protocol import ModelEntry

if TYPE_CHECKING:
    from openai.types import CompletionUsage

KEY_ENV: Final = "DEEPSEEK_API_KEY"
PROVIDER: Final = Provider(
    "https://api.deepseek.com/v1",
    KEY_ENV,
    # Unset, max_tokens is 8K without thinking and 64K with it (128K at effort max); the API allows 384K.
    defaults={"max_tokens": 131072},
)
MODELS_URL: Final = "https://api.deepseek.com/models"
"""Needs the key (401 without), and names each model's window and output limit (measured 2026-10-24)."""
FETCH_TIMEOUT: Final = 10.0


class _Model(BaseModel):
    id: str
    name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None


class _Models(BaseModel):
    data: list[_Model] = Field(default_factory=list[_Model])


class DeepSeekChat(OpenAIChatModel):
    """Counts cache hits from `prompt_cache_hit_tokens`, which DeepSeek reports beside the standard details."""

    def usage(self, usage: CompletionUsage) -> RequestUsage:
        counted = super().usage(usage)
        hits = (usage.model_extra or {}).get("prompt_cache_hit_tokens")
        if not counted.cache_read_tokens and isinstance(hits, int):
            return replace(counted, cache_read_tokens=hits)
        return counted


class DeepSeek(aid.ModelProvider):
    """DeepSeek's own API, keyed by DEEPSEEK_API_KEY."""

    name = "deepseek"
    key_env = KEY_ENV

    async def transport(self, model: str) -> DeepSeekChat:
        return DeepSeekChat(f"{self.name}/{model}", PROVIDER)

    async def fetch_models(self) -> list[ModelEntry]:
        if not (key := os.environ.get(KEY_ENV)):
            raise aid.AidError("missing_credentials", f"listing DeepSeek's models needs {KEY_ENV}")
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            response = await client.get(MODELS_URL, headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
        return [
            ModelEntry(id=model.id, name=model.name, context=model.context_window, output=model.max_output_tokens)
            for model in _Models.model_validate_json(response.content).data
        ]
