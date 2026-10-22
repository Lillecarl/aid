"""Agents for the catalog tests, on a test-only AID_PLUGINS_PATH."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aid
from aid.agent import Agent
from aid.messages import UserPromptPart
from tests.models import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aid.messages import ModelMessage


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


class Base(aid.AgentDefinition):
    """Abstract: no build, so not an agent."""


class Shouter(Base):
    """Repeats the prompt in capitals.

    This second paragraph is not part of the description.
    """

    def build(self) -> Agent:
        return Agent(FunctionModel(stream_function=_shout))


class Named(aid.AgentDefinition):
    name = "custom-name"

    def build(self) -> Agent:
        return Agent(FunctionModel(stream_function=_shout))
