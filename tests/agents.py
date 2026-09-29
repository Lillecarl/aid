"""pydantic-ai agents for tests. The daemon imports these by target, through `python_path`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

import aid

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
# Calls send_message once; TestModel makes up the arguments, so it writes "a" to session "a".
messenger = Agent(TestModel(call_tools=["send_message"]))


def _planned(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The prompt is a JSON plan, [[tool, args], ...]: call each in turn, then answer with every result joined."""
    start = max(i for i, m in enumerate(messages) for p in m.parts if p.part_kind == "user-prompt")
    plan: list[tuple[str, dict[str, object]]] = json.loads(_last_prompt(messages[: start + 1]))
    results = [
        p.model_response() if isinstance(p, RetryPromptPart) else str(p.content)
        for m in messages[start:]
        for p in m.parts
        if isinstance(p, ToolReturnPart | RetryPromptPart)
    ]
    if len(results) < len(plan):
        tool, args = plan[len(results)]
        return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=f"call{len(results)}")])
    return ModelResponse(parts=[TextPart("\n=====\n".join(results))])


async def _planned_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _planned(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


# Its answer is every tool result of the plan, joined by =====.
coder = Agent(FunctionModel(_planned, stream_function=_planned_stream), toolsets=[aid.coding_tools])
