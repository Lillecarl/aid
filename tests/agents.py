"""pydantic-ai agents for tests. The daemon imports these by target, through `python_path`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage


def _last_prompt(messages: list[ModelMessage]) -> str:
    for part in reversed(messages[-1].parts):
        content = getattr(part, "content", None)
        if isinstance(content, str):
            return content
    return ""


def _user_prompts(messages: list[ModelMessage]) -> int:
    return sum(1 for m in messages for p in m.parts if p.part_kind == "user-prompt")


async def _echo(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    prompt = _last_prompt(messages)
    if prompt == "slow":
        yield "waiting"
        await anyio.sleep_forever()
    if prompt == "env":
        yield f"{os.environ.get('AID_TEST_VAR')} {Path.cwd()}"
        return
    yield f"turn {_user_prompts(messages)}: "
    yield f"echo {prompt}"


echo = Agent(FunctionModel(stream_function=_echo))


class Review(BaseModel):
    verdict: str
    score: int


reviewer = Agent(TestModel(call_tools=[], custom_output_args={"verdict": "approve", "score": 7}), output_type=Review)

# TestModel calls each tool in call_tools once, then answers with their results. Not "all": aid's own tools
# reach every session too.
tool_caller = Agent(TestModel(call_tools=["add", "whoami"]))
every_tool_caller = Agent(TestModel())
