"""Native agents for tests. The daemon imports these by target, through `python_path`."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import anyio
from pydantic import BaseModel, JsonValue, ValidationError

import aid
from aid.agent import Agent
from aid.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.messaging import ARRIVED
from aid.protocol import AidError, Goal, decode_request
from plugins.supervision.classifier import ClassifierVerdict
from tests.models import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
    TestModel,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aid.messages import ModelMessage


def _is_note(content: object) -> bool:
    # Each turn's prompt carries a session metadata part, and messages a turn takes mid-way open with ARRIVED; the plan
    # and the turn count live in the real prompts.
    return isinstance(content, str) and content.startswith(("[Session metadata]", ARRIVED))


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


steady = Agent(TestModel(call_tools=[], custom_output_text="steady"))
"""A repeatable plain-text agent: every turn and every summary answers the same, with no tools."""


async def repeat() -> str:
    """A slow, unchanged result, so a live supervisor has time to inspect and interrupt the fake loop."""
    await anyio.sleep(0.02)
    return "unchanged"


async def _looping(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if _last_prompt(messages) == "continue":
        yield "continued"
        return
    number = sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
    yield {0: DeltaToolCall(name="repeat", json_args="{}", tool_call_id=f"repeat{number}")}


looping = Agent(FunctionModel(stream_function=_looping), tools=[repeat])


async def _loop_judge(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match = re.search(
        r"Suspected tool loop in supervised session '([^']+)', live turn '([^']+)'", _last_prompt(messages)
    )
    if match is None:
        yield "reviewed"
        return
    yield {
        0: DeltaToolCall(
            name="interrupt", json_args=json.dumps({"session": match[1], "turn": match[2]}), tool_call_id="interrupt1"
        )
    }


loop_judge = Agent(FunctionModel(stream_function=_loop_judge), toolsets=[aid.supervisor_tools])


async def _failing(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """A failing worker: it says something, then raises, so the turn ends with `agent_error`."""
    yield "about to fail"
    raise RuntimeError("boom")


failer = Agent(FunctionModel(stream_function=_failing))


def _summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A summarizer for compaction tests: the digest is always the same text."""
    return ModelResponse(parts=[TextPart("kept decisions")])


async def _summarize_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield "kept decisions"


summarizer = Agent(FunctionModel(_summarize, stream_function=_summarize_stream))


async def _summarize_at_gate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A compaction that marks its start, then waits for the test's gate file, so a test can act while it runs."""
    gate = anyio.Path(os.environ["AID_COMPACT_GATE"])
    await gate.with_suffix(".started").touch()
    while not await gate.exists():  # noqa: ASYNC110 -- the gate is a file the test drops; no event crosses the worker boundary
        await anyio.sleep(0.05)
    return _summarize(messages, info)


slow_summarizer = Agent(FunctionModel(_summarize_at_gate, stream_function=_summarize_stream))


async def _think(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaThinkingCalls]:
    """A thinker for the visibility tests: a thought, then the test's gate file, then the rest."""
    yield {0: DeltaThinkingPart(content="hmm, ")}
    gate = anyio.Path(os.environ["AID_THINK_GATE"])
    while not await gate.exists():  # noqa: ASYNC110 -- the gate is a file the test drops; no event crosses the worker boundary
        await anyio.sleep(0.1)
    yield {0: DeltaThinkingPart(content="let me think")}
    yield "thought through"


thinker = Agent(FunctionModel(stream_function=_think))

WIDE_THOUGHT: Final = "Planning:\n\n```\n" + "x = 1; " * 60 + "\n```\n\n" + "a" * 300 + "\n"


async def _think_wide(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaThinkingCalls]:
    """A thought with a long code line and a long unbroken token, for layout tests."""
    yield {0: DeltaThinkingPart(content=WIDE_THOUGHT)}
    yield "done"


wide_thinker = Agent(FunctionModel(stream_function=_think_wide))


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


def _plan_at(messages: list[ModelMessage]) -> tuple[int, list[tuple[str, dict[str, JsonValue]]]]:
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
                    return i, cast(list[tuple[str, dict[str, JsonValue]]], plan)
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


async def control_call(request: dict[str, JsonValue]) -> str:
    """Call the daemon through this agent's normal connector and return the result or refusal code."""
    try:
        async with aid.connect() as client:
            result = await client.call(decode_request(json.dumps(request).encode()))
        return json.dumps(result)
    except AidError as error:
        return json.dumps({"error": error.code})


control_probe = Agent(FunctionModel(_planned, stream_function=_planned_stream), tools=[control_call])


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


def _arming_watcher(background: dict[str, JsonValue], monitor: dict[str, JsonValue]) -> Agent:
    """An agent that starts one background task, arms one watch, answers "armed", then echoes whatever wakes
    it. The wake turn's answer proves what each firing delivered."""

    def decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called = {p.tool_name for m in messages for p in m.parts if isinstance(p, ToolCallPart)}
        if "background" not in called:
            return ModelResponse(parts=[ToolCallPart("background", background, tool_call_id="call0")])
        if "monitor" not in called:
            return ModelResponse(parts=[ToolCallPart("monitor", monitor, tool_call_id="call1")])
        if _user_prompts(messages) == 1:
            return ModelResponse(parts=[TextPart("armed")])
        return ModelResponse(parts=[TextPart(f"woke: {_last_prompt(messages)}")])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        match decide(messages, info).parts:
            case [ToolCallPart() as call]:
                yield {
                    0: DeltaToolCall(
                        name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id
                    )
                }
            case [TextPart() as text]:
                yield text.content
            case parts:
                raise ValueError(f"unexpected parts {parts}")

    return Agent(FunctionModel(decide, stream_function=stream), toolsets=[aid.coding_tools])


pattern_watcher = _arming_watcher(
    {"argv": ["printf", "plain one\nREADY one\nplain two\nREADY two\n"]},
    {"task_id": "bg1", "on": "pattern", "pattern": "READY"},
)
timeout_watcher = _arming_watcher(
    {"argv": ["sleep", "30"]},
    {"task_id": "bg1", "on": "running_after", "after": 0.5},
)


def _supervising_call(tool: str, args: dict[str, JsonValue], call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=call_id)])


def _supervising_goal_id(messages: list[ModelMessage], text: str) -> str:
    """The goal id the goals tool reported for `text`: the wake turn's answer proves what the daemon delivered."""
    for message in messages:
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or not isinstance(part.content, str):
                continue
            try:
                reported: JsonValue = json.loads(part.content)
            except ValueError:
                continue
            if isinstance(reported, list):
                for entry in reported:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        goal = Goal.model_validate(entry)
                    except ValidationError:
                        continue
                    if goal.text == text:
                        return goal.id
    raise ValueError(f"no goal {text!r} in tool results")


def _supervising_decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A supervisor probe: over three wake turns it reads its subject's history, searches it, tracks two goals,
    then settles both. The subject comes from AID_SUPERVISED, so one probe serves any test."""
    target = os.environ.get("AID_SUPERVISED", "coder")
    turn = _user_prompts(messages)
    called = [p.tool_name for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
    if turn == 1:
        if "supervised_history" not in called:
            return _supervising_call("supervised_history", {"session": target, "limit": 50}, "call0")
        if "supervised_search" not in called:
            return _supervising_call("supervised_search", {"session": target, "query": "do the thing"}, "call1")
        if called.count("goal_add") == 0:
            return _supervising_call("goal_add", {"session": target, "text": "ship it"}, "call2")
        if called.count("goal_add") == 1:
            return _supervising_call("goal_add", {"session": target, "text": "skip it"}, "call3")
        if "goals" not in called:
            return _supervising_call("goals", {"session": target}, "call4")
        return ModelResponse(parts=[TextPart("noted")])
    if turn == 2:
        if "goal_done" not in called:
            return _supervising_call(
                "goal_done", {"session": target, "goal_id": _supervising_goal_id(messages, "ship it")}, "call5"
            )
        if "goal_drop" not in called:
            return _supervising_call(
                "goal_drop", {"session": target, "goal_id": _supervising_goal_id(messages, "skip it")}, "call6"
            )
        return ModelResponse(parts=[TextPart("settled")])
    return ModelResponse(parts=[TextPart("all done")])


async def _supervising_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _supervising_decide(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


supervisor_probe = Agent(
    FunctionModel(_supervising_decide, stream_function=_supervising_stream), toolsets=[aid.supervisor_tools]
)


def _answering_decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A supervisor that answers one waiting permission: its first wake names the session, the request id
    and the option ids, and it answers with AID_ANSWER_OPTION (default: the first). Its answer proves the
    backstop woke it; any other wake (a turn-end firing) gets a verdict and no tool call."""
    prompt = _last_prompt(messages)
    session = re.search(r"Supervised session '([^']+)'", prompt)
    request = re.search(r"request ([0-9a-f]+)", prompt)
    options = re.findall(r"(?m)^- (\S+):", prompt)
    called = {p.tool_name for m in messages for p in m.parts if isinstance(p, ToolCallPart)}
    if _user_prompts(messages) == 1 and session and request and options and "answer_permission" not in called:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "answer_permission",
                    {
                        "session": session.group(1),
                        "request_id": request.group(1),
                        "option_id": os.environ.get("AID_ANSWER_OPTION") or options[0],
                    },
                    tool_call_id="call0",
                )
            ]
        )
    if "answer_permission" in called:
        return ModelResponse(parts=[TextPart("answered")])
    return ModelResponse(parts=[TextPart("noted")])


async def _answering_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _answering_decide(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


supervisor_answerer = Agent(
    FunctionModel(_answering_decide, stream_function=_answering_stream), toolsets=[aid.supervisor_tools]
)


def _last_user_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if part.part_kind == "user-prompt" and isinstance(part.content, str) and not _is_note(part.content):
                return part.content
    return ""


def _forwarding_decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A supervisor unsure of everything: it forwards each permission request it may forward, and answers
    with the first option once a person had the request and left it."""
    if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
        return ModelResponse(parts=[TextPart("done")])
    prompt = _last_user_prompt(messages)
    session = re.search(r"Supervised session '([^']+)' waits on permission", prompt)
    request = re.search(r"request ([0-9a-f]+)", prompt)
    options = re.findall(r"(?m)^- (\S+):", prompt)
    if not (session and request and options):
        return ModelResponse(parts=[TextPart("noted")])
    args: dict[str, JsonValue] = {"session": session.group(1), "request_id": request.group(1)}
    if "cannot be forwarded" in prompt:
        call = ToolCallPart("answer_permission", {**args, "option_id": options[0]}, tool_call_id="answer")
    else:
        call = ToolCallPart("forward_permission", {**args, "note": "unsure: check the target"}, tool_call_id="fwd")
    return ModelResponse(parts=[call])


async def _forwarding_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _forwarding_decide(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


supervisor_forwarder = Agent(
    FunctionModel(_forwarding_decide, stream_function=_forwarding_stream), toolsets=[aid.supervisor_tools]
)


async def _looping_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """A stuck worker: every turn outputs the same line, so the loop score reads 1.0 from the second turn."""
    yield "stuck on the same thing"


looper = Agent(FunctionModel(stream_function=_looping_stream))


_NUDGES = ("keep digging", "try another angle", "check the logs", "ask what changed")


def _nagging_decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A nudging supervisor: one send_message per wake turn, each worded differently, until AID_NUDGES
    sends (default 2). Then it stands by: past the budget the daemon refuses, and silence is the verdict."""
    target = os.environ.get("AID_SUPERVISED", "coder")
    budget = int(os.environ.get("AID_NUDGES", "2"))
    turn = _user_prompts(messages)
    sent = sum(1 for m in messages for p in m.parts if isinstance(p, ToolCallPart) and p.tool_name == "send_message")
    if sent < budget and sent < turn and sent < len(_NUDGES):
        return ModelResponse(
            parts=[ToolCallPart("send_message", {"to": target, "text": _NUDGES[sent]}, tool_call_id=f"nudge{sent}")]
        )
    if sent:
        return ModelResponse(parts=[TextPart("nudged enough, standing by")])
    return ModelResponse(parts=[TextPart("noted")])


async def _nagging_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    match _nagging_decide(messages, info).parts:
        case [ToolCallPart() as call]:
            yield {
                0: DeltaToolCall(name=call.tool_name, json_args=call.args_as_json_str(), tool_call_id=call.tool_call_id)
            }
        case [TextPart() as text]:
            yield text.content
        case parts:
            raise ValueError(f"unexpected parts {parts}")


nagger = Agent(FunctionModel(_nagging_decide, stream_function=_nagging_stream))
failed_watcher = _arming_watcher(
    {"argv": ["true"]},
    {"task_id": "bg1", "on": "failed"},
)


def _classifier(verdict: str, reason: str) -> Agent:
    return Agent(
        TestModel(call_tools=[], custom_output_args={"verdict": verdict, "reason": reason}),
        output_type=ClassifierVerdict,
    )


classifier_allow = _classifier("allow", "touching a file in the project")
classifier_deny = _classifier("deny", "write it with the edit tools instead")
classifier_escalate = _classifier("escalate", "needs the task to judge")
