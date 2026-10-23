"""Turn lifecycle regressions against real native runs with deterministic local models."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

import anyio
import pytest
from pydantic import BaseModel, Field

from aid.agent import Agent, ModelRetry, Tool
from aid.backends.native import (
    DEFAULT_DESIRED_MAX,
    NOT_FORKED,
    THOUGHT_LIMIT,
    NativeBackend,
    UsageLimitExceeded,
    aid_toolset,
    open_turn,
)
from aid.backends.permissions import PermissionWaits
from aid.coding import CODING, Coding, coding_tools
from aid.discovery import Catalog
from aid.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import ModelInfo, StreamDelta
from aid.protocol import AidError, Output, SessionEvent, Started, TextDelta, ToolCall, Usage
from aid.spec import PermissionMode
from aid.tools import McpTool
from tests.agents import coder, echo
from tests.models import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel, TestModel
from tests.test_coding import plan

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path


pytestmark = pytest.mark.anyio


@asynccontextmanager
async def backend(
    directory: Path,
    agent: Agent,
    *,
    history: list[ModelMessage] | None = None,
    max_requests: int | None = None,
    desired_max: int = DEFAULT_DESIRED_MAX,
    open_tail: int = 0,
) -> AsyncGenerator[NativeBackend]:
    coding = Coding(
        cwd=directory,
        store=directory / "runs",
        outputs=directory / "outputs",
        mode=PermissionMode.DENY,
        timeout=10,
        waits=PermissionWaits(),
    )
    session = NativeBackend(
        agent,
        anyio.Path(directory / "history.json"),
        history or [],
        [],
        Started(pid=1, agent="test", model="local"),
        coding,
        max_requests=max_requests,
        desired_max=desired_max,
        open_tail=open_tail,
    )
    try:
        yield session
    finally:
        try:
            await session.aclose()
        finally:
            await coding.close()


async def stored(directory: Path) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(await anyio.Path(directory / "history.json").read_bytes())


@pytest.mark.parametrize("arguments", ['{"value":', '{"value":"wrong"}'])
async def test_invalid_tool_arguments_return_a_correction_without_execution(tmp_path: Path, arguments: str) -> None:
    calls: list[int] = []

    def record(value: int) -> int:
        calls.append(value)
        return value

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts):
            return ModelResponse([TextPart("corrected")])
        return ModelResponse([ToolCallPart("record", arguments, "c1")])

    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(FunctionModel(respond), tools=[record])) as session:
        assert (await session.prompt("go", emit)).output == "corrected"
    assert calls == []
    assert any(isinstance(event, ToolCall) and event.status == "failed" for event in events)
    assert any(
        isinstance(p, RetryPromptPart) and p.tool_call_id == "c1" for m in await stored(tmp_path) for p in m.parts
    )


class TypedAnswer(BaseModel):
    count: int


async def test_tool_preserves_pydantic_argument_constraints() -> None:
    executed: list[int] = []

    def positive(value: Annotated[int, Field(gt=0, description="A positive value")]) -> int:
        executed.append(value)
        return value

    tool = Tool(positive)
    schema = tool.definition()["function"]["parameters"]["properties"]["value"]
    assert schema["exclusiveMinimum"] == 0
    assert schema["description"] == "A positive value"
    with pytest.raises(ModelRetry):
        await tool.call({"value": 0}, "c1")
    assert executed == []
    assert await tool.call({"value": 1}, "c2") == 1


@pytest.mark.parametrize("identities", [[""], ["c1", "c1"]])
async def test_invalid_tool_identities_never_execute_and_history_remains_resumable(
    tmp_path: Path, identities: list[str]
) -> None:
    executed: list[bool] = []

    def action() -> str:
        executed.append(True)
        return "done"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse([ToolCallPart("action", {}, identity) for identity in identities])

    async def emit(event: SessionEvent) -> None:
        pass

    async with backend(tmp_path, Agent(FunctionModel(respond), tools=[action])) as session:
        with pytest.raises(AidError, match="identities"):
            await session.prompt("go", emit)
    history = await stored(tmp_path)
    assert not executed
    assert not any(isinstance(part, ToolCallPart) for message in history for part in message.parts)
    response = next(message for message in history if isinstance(message, ModelResponse))
    assert response.model_extra and len(response.model_extra["invalid_tool_calls"]) == len(identities)
    async with backend(tmp_path, Agent(TestModel(call_tools=[])), history=history) as session:
        assert (await session.prompt("continue", emit)).stop_reason == "end_turn"


async def test_typed_output_correction_is_validated_and_counted(tmp_path: Path) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        corrected = any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
        return ModelResponse([TextPart('{"count":2}' if corrected else '{"count":"wrong"}')])

    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(FunctionModel(respond), output_type=TypedAnswer)) as session:
        assert (await session.prompt("go", emit)).output == {"count": 2}
    assert next(event.requests for event in events if isinstance(event, Usage)) == 2


async def test_missing_usage_stays_unknown(tmp_path: Path) -> None:
    class SilentModel(TestModel):
        async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
            yield StreamDelta(0, TextPart("done"))

    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(SilentModel())) as session:
        await session.prompt("go", emit)
    usage = next(event for event in events if isinstance(event, Usage))
    assert usage.requests == 1
    assert usage.input_tokens is usage.output_tokens is usage.cache_read_tokens is None
    # The context gauge is aid's own estimate, so it is known even when the engine reports no tokens.
    assert usage.context_size == DEFAULT_DESIRED_MAX
    assert usage.context_used is not None and usage.context_used > 0


async def test_turn_reports_the_context_gauge(tmp_path: Path) -> None:
    """The native backend has no engine window to report, so a turn reports aid's estimate against the session's gauge."""
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(TestModel(call_tools=[])), desired_max=1000) as session:
        await session.prompt("first", emit)
        first = [event for event in events if isinstance(event, Usage)][-1]
        assert first.context_size == 1000
        assert first.context_used is not None and first.context_used > 0
        await session.prompt("second", emit)

    second = [event for event in events if isinstance(event, Usage)][-1]
    assert second.context_size == 1000
    assert second.context_used is not None and second.context_used > first.context_used


async def test_unset_request_budget_does_not_inherit_library_limit(tmp_path: Path) -> None:
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, coder) as session:
        with anyio.fail_after(15):
            calls: list[tuple[str, dict[str, Any]]] = [("ls", {})] * 55
            result = await session.prompt(plan(*calls), emit)
    assert result.stop_reason == "end_turn"
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert usage[0].requests == 56


async def test_explicit_budget_preserves_history_usage_and_resume(tmp_path: Path) -> None:
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, coder, max_requests=2) as session:
        with pytest.raises(UsageLimitExceeded):
            calls: list[tuple[str, dict[str, Any]]] = [("ls", {})] * 4
            await session.prompt(plan(*calls), emit)
    history = await stored(tmp_path)
    assert sum(isinstance(part, ToolReturnPart) for message in history for part in message.parts) == 2
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert usage[0].requests == 2
    assert not any(
        isinstance(part, UserPromptPart) and str(part.content).startswith("[Session metadata")
        for message in history
        for part in message.parts
    )
    async with backend(tmp_path, coder, history=history) as resumed:
        assert (await resumed.prompt(plan(("ls", {})), emit)).stop_reason == "end_turn"
    assert len(await stored(tmp_path)) > len(history)


async def test_cancelled_stream_preserves_partial_response_and_usage(tmp_path: Path) -> None:
    events: list[SessionEvent] = []
    async with backend(tmp_path, echo) as session:

        async def emit(event: SessionEvent) -> None:
            events.append(event)
            if isinstance(event, TextDelta):
                await session.cancel()

        with anyio.fail_after(10):
            result = await session.prompt("slow", emit)
    assert result == Output(output=None, stop_reason="cancelled")
    history = await stored(tmp_path)
    assert any(isinstance(part, TextPart) and part.content == "waiting" for m in history for part in m.parts)
    assert len([event for event in events if isinstance(event, Usage)]) == 1


async def test_cancelled_tool_can_resume_without_reexecuting_it(tmp_path: Path) -> None:
    calls = 0

    async def wait_forever() -> str:
        nonlocal calls
        calls += 1
        await anyio.sleep_forever()
        return "never"

    agent = Agent(TestModel(call_tools=["wait_forever"]), tools=[wait_forever])
    async with backend(tmp_path, agent) as session:

        async def emit(event: SessionEvent) -> None:
            if isinstance(event, ToolCall):
                await session.cancel()

        with anyio.fail_after(10):
            assert (await session.prompt("wait", emit)).stop_reason == "cancelled"
    history = await stored(tmp_path)
    assert history
    interrupted = [part for m in history for part in m.parts if isinstance(part, ToolReturnPart)]
    assert len(interrupted) == 1
    assert interrupted[0].outcome == "interrupted"
    before_resume = calls
    # Cancellation may arrive before or during execution; neither case replays the tool on resume.
    async with backend(tmp_path, Agent(TestModel(call_tools=[])), history=history) as resumed:
        assert (await resumed.prompt("continue", emit)).stop_reason == "end_turn"
    assert calls == before_resume


async def test_external_cancellation_preserves_state_without_becoming_success(tmp_path: Path) -> None:
    events: list[SessionEvent] = []
    async with backend(tmp_path, echo) as session:
        with anyio.CancelScope() as scope:

            async def emit(event: SessionEvent) -> None:
                events.append(event)
                if isinstance(event, TextDelta):
                    scope.cancel()

            await session.prompt("slow", emit)
            pytest.fail("external cancellation must propagate")
    assert scope.cancelled_caught
    history = await stored(tmp_path)
    assert any(isinstance(part, TextPart) and part.content == "waiting" for m in history for part in m.parts)
    assert len([event for event in events if isinstance(event, Usage)]) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
async def test_failed_tool_preserves_original_error_and_closes_unanswered_call(
    tmp_path: Path, error_type: type[Exception]
) -> None:
    async def fail() -> str:
        raise error_type("tool broke")

    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(TestModel(call_tools=["fail"]), tools=[fail])) as session:
        with pytest.raises(error_type, match="tool broke"):
            await session.prompt("go", emit)
    history = await stored(tmp_path)
    assert any(isinstance(part, ToolReturnPart) and part.outcome == "interrupted" for m in history for part in m.parts)
    assert len([event for event in events if isinstance(event, Usage)]) == 1
    async with backend(tmp_path, Agent(TestModel(call_tools=[])), history=history) as resumed:
        assert (await resumed.prompt("continue", emit)).stop_reason == "end_turn"


async def _compact_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if any(isinstance(part, ToolReturnPart) and part.tool_name == "compact" for m in messages for part in m.parts):
        yield "finished"
    else:
        yield {0: DeltaToolCall(name="compact", json_args='{"instructions":"keep decisions"}', tool_call_id="c1")}


def _summary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("decisions kept")])


@pytest.mark.parametrize("cancel_after_compaction", [False, True])
async def test_in_turn_compaction_survives_turn_end(tmp_path: Path, cancel_after_compaction: bool) -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[TextPart("one")]),
        ModelRequest(parts=[UserPromptPart("second")]),
        ModelResponse(parts=[TextPart("two")]),
    ]
    agent = Agent(FunctionModel(_summary, stream_function=_compact_stream), toolsets=[coding_tools])
    async with backend(tmp_path, agent, history=history) as session:

        async def emit(event: SessionEvent) -> None:
            if cancel_after_compaction and isinstance(event, ToolCall) and event.status == "completed":
                await session.cancel()

        with anyio.fail_after(10):
            result = await session.prompt("compact now", emit)
    assert result.stop_reason == ("cancelled" if cancel_after_compaction else "end_turn")
    prompts = [part.content for m in await stored(tmp_path) for part in m.parts if isinstance(part, UserPromptPart)]
    assert prompts == ["[Summary of earlier work]\ndecisions kept", "second", "compact now"]


async def test_a_request_that_fails_before_output_stores_no_empty_response(tmp_path: Path) -> None:
    """A provider refusal (missing key, 400) leaves no assistant message behind: an empty one would be sent on
    every later turn, and DeepSeek refuses those, so the session could never answer again."""

    class RefusingModel(TestModel):
        async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
            raise RuntimeError("missing credentials")
            yield StreamDelta()  # pragma: no cover -- makes this a generator

    async def emit(event: SessionEvent) -> None:
        pass

    async with backend(tmp_path, Agent(RefusingModel())) as session:
        with pytest.raises(RuntimeError, match="missing credentials"):
            await session.prompt("go", emit)
    history = await stored(tmp_path)
    assert not any(isinstance(m, ModelResponse) and not m.parts for m in history)
    async with backend(tmp_path, Agent(TestModel(call_tools=[])), history=history) as resumed:
        assert (await resumed.prompt("again", emit)).stop_reason == "end_turn"


async def test_catalog_tool_errors_go_back_to_the_model(tmp_path: Path) -> None:
    """An `@aid.mcptool` failing over MCP answers the model with the error; in-process it must too, not end the
    turn. An agent's own tools keep ending it (test_failed_tool_preserves_original_error_...)."""

    def fail() -> str:
        raise RuntimeError("no such session")

    [tool] = aid_toolset(Catalog(tools={"fail": McpTool(fail, "fail", None)})).tools
    assert isinstance(tool, Tool) and tool.errors_to_model

    async def emit(event: SessionEvent) -> None:
        pass

    async with backend(tmp_path, Agent(TestModel(call_tools=["fail"]), tools=[tool])) as session:
        assert (await session.prompt("go", emit)).stop_reason == "end_turn"
    retries = [p for m in await stored(tmp_path) for p in m.parts if isinstance(p, RetryPromptPart)]
    assert [p.content for p in retries] == ["RuntimeError: no such session"]


async def test_snapshot_holds_the_turn_so_far_and_closes_later_calls(tmp_path: Path) -> None:
    """A fork starts from inside a tool call: history.json lacks the running turn, the snapshot must not."""
    snapshots: list[Path] = []

    def look() -> str:
        return "looked"

    async def fork() -> str:
        coding = CODING.get()
        assert coding.snapshot is not None
        snapshots.append(await coding.snapshot("you are the fork"))
        return "forked"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse([TextPart("done")])
        return ModelResponse(
            [ToolCallPart("look", "{}", "c1"), ToolCallPart("fork", "{}", "c2"), ToolCallPart("look", "{}", "c3")]
        )

    async def emit(event: SessionEvent) -> None:
        pass

    earlier = [ModelRequest(parts=[UserPromptPart("earlier")]), ModelResponse([TextPart("ok")])]
    async with backend(tmp_path, Agent(FunctionModel(respond), tools=[look, fork]), history=earlier) as session:
        await session.prompt("go", emit)
        with pytest.raises(AidError, match="inside a tool call"):
            await session.snapshot("outside")

    [path] = snapshots
    seed = ModelMessagesTypeAdapter.validate_json(path.read_bytes())
    assert path.parent == tmp_path / "forks"
    assert seed[:2] == earlier
    assert [p.content for p in seed[2].parts if isinstance(p, UserPromptPart)] == ["go"]
    assert [p.tool_call_id for p in seed[3].parts if isinstance(p, ToolCallPart)] == ["c1", "c2", "c3"]
    assert [(p.tool_call_id, p.content) for p in seed[4].parts if isinstance(p, ToolReturnPart)] == [("c1", "looked")]
    assert [(p.tool_call_id, p.content) for p in seed[5].parts if isinstance(p, ToolReturnPart)] == [
        ("c2", "you are the fork"),
        ("c3", NOT_FORKED),
    ]
    assert len(seed) == 6
    returns = [p.tool_call_id for m in await stored(tmp_path) for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns == ["c1", "c2", "c3"]


async def test_a_forks_first_turn_sends_its_parents_open_turn_unpruned(tmp_path: Path) -> None:
    """The parent sent its running turn's thinking whole; the fork must too, or its prefix and cache part there."""
    long = "x" * (THOUGHT_LIMIT * 2)
    seed: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("earlier")]),
        ModelResponse([ThinkingPart(long), TextPart("ok")]),
        ModelRequest(parts=[UserPromptPart("now")]),
        ModelResponse([ThinkingPart(long), ToolCallPart("fork", "{}", "c1")]),
        ModelRequest(parts=[ToolReturnPart("fork", "you are the fork", "c1")]),
    ]
    assert open_turn(seed) == 3
    seen: list[list[int]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append([len(p.content) for m in messages for p in m.parts if isinstance(p, ThinkingPart)])
        return ModelResponse([TextPart("done")])

    async def emit(event: SessionEvent) -> None:
        pass

    async with backend(tmp_path, Agent(FunctionModel(respond)), history=seed, open_tail=open_turn(seed)) as session:
        await session.prompt("the fork's instructions", emit)
        await session.prompt("a later turn", emit)
    assert seen[0] == [THOUGHT_LIMIT + 1, len(long)]
    assert seen[1] == [THOUGHT_LIMIT + 1, THOUGHT_LIMIT + 1]
