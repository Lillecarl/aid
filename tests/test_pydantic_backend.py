"""Turn lifecycle regressions against real pydantic-ai runs with deterministic local models."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from aid.backends.permissions import PermissionWaits
from aid.backends.pydantic_ai import PydanticAIBackend
from aid.coding import Coding, coding_tools
from aid.protocol import Output, SessionEvent, Started, TextDelta, ToolCall, Usage
from aid.spec import PermissionMode
from tests.agents import coder, echo
from tests.test_coding import plan

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path

    from pydantic_ai.agent import AbstractAgent

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def backend(
    directory: Path,
    agent: AbstractAgent[Any, Any],
    *,
    history: list[ModelMessage] | None = None,
    max_requests: int | None = None,
) -> AsyncGenerator[PydanticAIBackend]:
    coding = Coding(
        cwd=directory,
        store=directory / "runs",
        outputs=directory / "outputs",
        mode=PermissionMode.DENY,
        timeout=10,
        waits=PermissionWaits(),
    )
    try:
        yield PydanticAIBackend(
            agent,
            anyio.Path(directory / "history.json"),
            history or [],
            [],
            Started(pid=1, agent="test", model="local"),
            coding,
            max_requests=max_requests,
        )
    finally:
        await coding.close()


async def stored(directory: Path) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(await anyio.Path(directory / "history.json").read_bytes())


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


async def test_failed_tool_preserves_original_error_and_closes_unanswered_call(tmp_path: Path) -> None:
    async def fail() -> str:
        raise RuntimeError("tool broke")

    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with backend(tmp_path, Agent(TestModel(call_tools=["fail"]), tools=[fail])) as session:
        with pytest.raises(RuntimeError, match="tool broke"):
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
