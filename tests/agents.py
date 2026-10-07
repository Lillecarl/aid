"""pydantic-ai agents for tests. The daemon imports these by target, through `python_path`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel

import aid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage


def _is_note(content: object) -> bool:
    # The send policy trails every request with session metadata; it is wire-only, and the plan and the turn
    # count live in the real prompts.
    return isinstance(content, str) and content.startswith("[Session metadata, not stored]")


def _last_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        for part in reversed(message.parts):
            content = getattr(part, "content", None)
            if isinstance(content, str) and not _is_note(content):
                return content
    return ""


def _user_prompts(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for m in messages
        for p in m.parts
        if p.part_kind == "user-prompt" and not _is_note(getattr(p, "content", None))
    )


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


def _summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A summarizer for compaction tests: the digest is always the same text."""
    return ModelResponse(parts=[TextPart("kept decisions")])


async def _summarize_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield "kept decisions"


summarizer = Agent(FunctionModel(_summarize, stream_function=_summarize_stream))


async def _think(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaThinkingCalls]:
    """A thinker for the visibility tests: a thought, then the test's gate file, then the rest."""
    yield {0: DeltaThinkingPart(content="hmm, ")}
    gate = anyio.Path(os.environ["AID_THINK_GATE"])
    while not await gate.exists():  # noqa: ASYNC110 -- the gate is a file the test drops; no event crosses the worker boundary
        await anyio.sleep(0.1)
    yield {0: DeltaThinkingPart(content="let me think")}
    yield "thought through"


thinker = Agent(FunctionModel(stream_function=_think))


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


def _plan_at(messages: list[ModelMessage]) -> tuple[int, list[tuple[str, dict[str, object]]]]:
    """The latest user prompt that parses as a plan, and its index: trailing metadata parses as JSON too, but
    never as a list."""
    for i in range(len(messages) - 1, -1, -1):
        for part in reversed(messages[i].parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                try:
                    plan = json.loads(part.content)
                except ValueError:
                    continue
                if isinstance(plan, list):
                    return i, cast("list[tuple[str, dict[str, object]]]", plan)
    raise ValueError("no plan in messages")


def _planned(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The prompt is a JSON plan, [[tool, args], ...]: call each in turn, then answer with every result joined."""
    start, plan = _plan_at(messages)
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


def _watcher(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Arms one background sleep and its monitor, answers "armed", then echoes whatever wakes it. The wake
    turn's answer proves what the resume delivered."""
    called = {p.tool_name for m in messages for p in m.parts if isinstance(p, ToolCallPart)}
    if "background" not in called:
        return ModelResponse(parts=[ToolCallPart("background", {"argv": ["sleep", "0.5"]}, tool_call_id="call0")])
    if "monitor" not in called:
        return ModelResponse(parts=[ToolCallPart("monitor", {"task_id": "bg1"}, tool_call_id="call1")])
    if _user_prompts(messages) == 1:
        return ModelResponse(parts=[TextPart("armed")])
    return ModelResponse(parts=[TextPart(f"woke: {_last_prompt(messages)}")])


async def _watcher_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _watcher(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


watcher = Agent(FunctionModel(_watcher, stream_function=_watcher_stream), toolsets=[aid.coding_tools])
