"""Backend for a `pydantic_ai` agent imported into the worker."""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
from pydantic_ai import FunctionToolset, RunCancelled, Tool
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

from aid.agents import ENV_AGENTS_PATH, Catalog, agents_path, discover
from aid.protocol import Output, SessionEvent, TextDelta, ThoughtDelta, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai.agent import AgentRunEvents
    from pydantic_ai.toolsets import AbstractToolset

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


def load_named(catalog: Catalog, name: str) -> AbstractAgent[Any, Any]:
    if (cls := catalog.agents.get(name)) is None:
        known = ", ".join(sorted(catalog.agents)) or "none"
        problems = "".join(f"\n  {problem}" for problem in catalog.problems)
        raise LookupError(f"no agent {name!r} on {ENV_AGENTS_PATH} (found: {known}){problems}")
    return cls().build()


def aid_toolset(catalog: Catalog) -> FunctionToolset[Any]:
    tools = [Tool[Any](t.fn, takes_ctx=False, name=t.name, description=t.description) for t in catalog.tools.values()]
    return FunctionToolset[Any](tools)


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
    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        history_file: anyio.Path,
        history: list[ModelMessage],
        toolsets: list[AbstractToolset[Any]],
    ) -> None:
        self._agent = agent
        self._history_file = history_file
        self._history = history
        self._toolsets = toolsets
        self._run: AgentRunEvents[Any] | None = None

    async def prompt(self, text: str, emit: Emit) -> Output:
        async with self._agent.run_stream_events(text, message_history=self._history, toolsets=self._toolsets) as run:
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
    # Imports and filesystem walks, run off the event loop.
    catalog = await anyio.to_thread.run_sync(discover, agents_path()) if spec.agent or spec.aid_tools else Catalog()
    if spec.agent is not None:
        agent = await anyio.to_thread.run_sync(load_named, catalog, spec.agent)
    elif spec.target is not None:
        agent = await anyio.to_thread.run_sync(load_target, spec.target)
    else:
        raise ValueError("the spec names no agent")
    toolsets: list[AbstractToolset[Any]] = [aid_toolset(catalog)] if spec.aid_tools and catalog.tools else []
    history_file = state_dir / HISTORY_FILE
    history = (
        ModelMessagesTypeAdapter.validate_json(await history_file.read_bytes()) if await history_file.exists() else []
    )
    log.info("loaded %s with %d history messages", spec.agent or spec.target, len(history))
    yield PydanticAIBackend(agent, history_file, history, toolsets)
