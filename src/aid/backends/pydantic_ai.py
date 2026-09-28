"""Backend for a `pydantic_ai` agent imported into the worker."""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunCancelled
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_core import to_jsonable_python

from aid.protocol import Output, SessionEvent, TextDelta, ThoughtDelta, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import anyio
    from pydantic_ai.agent import AgentRunEvents

    from aid.backends.base import Emit
    from aid.spec import PydanticAISpec

log = logging.getLogger(__name__)

HISTORY_FILE = "history.json"


def load_target(target: str) -> AbstractAgent[Any, Any]:
    module_name, _, attr_path = target.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not isinstance(obj, AbstractAgent):
        raise TypeError(f"{target} is {type(obj).__name__}, not a pydantic_ai agent")
    return obj  # pyright: ignore[reportUnknownVariableType] -- isinstance cannot narrow the generic parameters


def to_event(event: object) -> SessionEvent | None:
    match event:
        case PartStartEvent(part=TextPart(content=text)) | PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
            return TextDelta(text=text) if text else None
        case PartStartEvent(part=ThinkingPart(content=text)):
            return ThoughtDelta(text=text) if text else None
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text)):
            return ThoughtDelta(text=text) if text else None
        case FunctionToolCallEvent(part=part):
            return ToolCall(tool_call_id=part.tool_call_id, title=part.tool_name, status="in_progress")
        case FunctionToolResultEvent():
            return ToolCall(tool_call_id=event.tool_call_id, status="completed")
        case _:
            return None


class PydanticAIBackend:
    def __init__(self, agent: AbstractAgent[Any, Any], history_file: anyio.Path, history: list[ModelMessage]) -> None:
        self._agent = agent
        self._history_file = history_file
        self._history = history
        self._run: AgentRunEvents[Any] | None = None

    async def prompt(self, text: str, emit: Emit) -> Output:
        async with self._agent.run_stream_events(text, message_history=self._history) as run:
            self._run = run
            try:
                async for event in run:
                    if isinstance(event, AgentRunResultEvent):
                        result = event.result
                        break
                    if (session_event := to_event(event)) is not None:
                        await emit(session_event)
                else:
                    raise RuntimeError("pydantic_ai run ended without a result")
            except RunCancelled:
                return Output(output=None, stop_reason="cancelled")
            finally:
                self._run = None
        self._history = result.all_messages()
        await _write_atomic(self._history_file, ModelMessagesTypeAdapter.dump_json(self._history))
        return Output(output=to_jsonable_python(result.output), stop_reason="end_turn")

    async def cancel(self) -> None:
        if self._run is not None:
            self._run.cancel()


async def _write_atomic(path: anyio.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    await tmp.write_bytes(data)
    await tmp.rename(path)


@asynccontextmanager
async def open_pydantic_ai(spec: PydanticAISpec, state_dir: anyio.Path) -> AsyncGenerator[PydanticAIBackend]:
    agent = load_target(spec.target)
    history_file = state_dir / HISTORY_FILE
    history = (
        ModelMessagesTypeAdapter.validate_json(await history_file.read_bytes()) if await history_file.exists() else []
    )
    log.info("loaded %s with %d history messages", spec.target, len(history))
    yield PydanticAIBackend(agent, history_file, history)
