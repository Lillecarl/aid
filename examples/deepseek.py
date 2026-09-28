"""An aid agent on DeepSeek. The provider reads DEEPSEEK_API_KEY from the worker's environment.

Put this directory on AID_AGENTS_PATH and it shows up as `deep_seek`:

    aid agents
    aid new-py deepseek deep_seek
    aid prompt deepseek "What is aid?"
"""

from __future__ import annotations

from pydantic_ai import Agent

import aid


class DeepSeek(aid.PydanticAgent):
    """Short, plain answers from DeepSeek's chat model."""

    def build(self) -> Agent[None, str]:
        return Agent("deepseek:deepseek-chat", instructions="Answer briefly and plainly.")
