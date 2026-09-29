"""Agents for the catalog tests, on a test-only AID_AGENTS_PATH."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import aid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage


async def _shout(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    # The send policy trails every request with session metadata; the prompt lives before it.
    prompts = [
        p.content
        for m in messages
        for p in m.parts
        if isinstance(p, UserPromptPart)
        and isinstance(p.content, str)
        and not p.content.startswith("[Session metadata, not stored]")
    ]
    yield (prompts[-1] if prompts else "").upper()


class Base(aid.PydanticAgent):
    """Abstract: no build, so not an agent."""


class Shouter(Base):
    """Repeats the prompt in capitals.

    This second paragraph is not part of the description.
    """

    def build(self) -> Agent[None, str]:
        return Agent(FunctionModel(stream_function=_shout))


class Named(aid.PydanticAgent):
    name = "custom-name"

    def build(self) -> Agent[None, str]:
        return Agent(FunctionModel(stream_function=_shout))
