"""DeepSeek's API (`deepseek/` models): Chat Completions with DeepSeek's cache counting and output limit."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

import aid
from aid.model import OpenAIChatModel, Provider, RequestUsage

if TYPE_CHECKING:
    from openai.types import CompletionUsage

PROVIDER: Final = Provider(
    "https://api.deepseek.com/v1",
    "DEEPSEEK_API_KEY",
    # Unset, max_tokens is 8K without thinking and 64K with it (128K at effort max); the API allows 384K.
    defaults={"max_tokens": 131072},
)


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

    def transport(self, model: str) -> DeepSeekChat:
        return DeepSeekChat(f"{self.name}/{model}", PROVIDER)
