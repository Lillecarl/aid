"""Backend for a `pydantic_ai` agent imported into the worker."""

from __future__ import annotations

import importlib
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio
import anyio.to_thread
import structlog
from pydantic_ai import FunctionToolset, RunCancelled, Tool
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python

from aid.agents import ENV_AGENTS_PATH, Catalog, agents_path, discover
from aid.backends.permissions import PermissionWaits
from aid.coding import CODING, OUTPUTS_DIR, Coding
from aid.instructions import compose
from aid.plugins import discover as discover_plugins
from aid.plugins import plugins_path
from aid.protocol import (
    AidError,
    Output,
    SessionEvent,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    Usage,
    clip,
    to_json,
)
from aid.tools import tool_paths

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from pydantic_ai.agent import AgentRunEvents
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.toolsets import AbstractToolset

    from aid.backends.base import Emit, Notifier
    from aid.spec import PydanticAISpec

log = structlog.get_logger(__name__)

HISTORY_FILE = "history.json"
RUNS_DIR = "runs"
"""pyrun's store for what the session's `python` tool runs."""
THOUGHT_LIMIT: Final = 500
"""Characters of a thinking part kept for replay: thought serves the turn, not the replay."""
DEFAULT_DESIRED_MAX: Final = 100_000
"""Tokens the context gauge assumes when the session names no `max_context`. Rule of thumb: half the model's
window; unknown models err toward compacting early, which fails safe."""


@dataclass
class _MetadataRequest(ModelRequest):
    """Ephemeral per-request metadata: sent trailing every request, stripped before anything is stored."""


def prune_history(history: list[ModelMessage], boundary: int) -> list[ModelMessage]:
    """What the model is sent: before `boundary` past thinking rides as shells; the turn's own messages flow
    whole. Oversize tool results never reach this filter: they spill to files at production time. The rule is
    uniform per message and frozen once written — never by age, never retroactive — so sends grow
    monotonically and the provider's prefix cache holds. The stored history keeps the turn whole; older turns
    were already frozen this way when they ended."""
    pruned: list[ModelMessage] = []
    for index, message in enumerate(history):
        past = index < boundary
        parts: list[Any] = []
        changed = False
        for part in message.parts:
            if past and isinstance(part, ThinkingPart):
                # The provider's reasoning signature dwarfs the thought; each request stands alone without it.
                thought = part.content[:THOUGHT_LIMIT] + "…" if len(part.content) > THOUGHT_LIMIT else part.content
                parts.append(replace(part, content=thought, signature=None))
                changed = True
            else:
                parts.append(part)
        pruned.append(replace(message, parts=parts) if changed else message)
    return pruned


def _send_policy(boundary: int, desired_max: int) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """A `ProcessHistory` processor for one turn: `boundary` is where the stored history ended when the turn
    started, so the turn's own thinking flows whole and everything before it rides pruned. A metadata message
    trails every send; the backend strips those before storing."""

    def policy(messages: list[ModelMessage]) -> list[ModelMessage]:
        clean = [message for message in messages if not isinstance(message, _MetadataRequest)]
        pruned = prune_history(clean, boundary)
        used = len(ModelMessagesTypeAdapter.dump_json(pruned)) // 4
        meta = {"context_pct": round(used / desired_max * 100), "unixtime": int(time.time())}
        return [
            *pruned,
            _MetadataRequest(parts=[UserPromptPart(content=f"[Session metadata, not stored]\n{json.dumps(meta)}")]),
        ]

    return policy


def without_metadata(history: list[ModelMessage]) -> list[ModelMessage]:
    """The stored form: everything the run tracked, minus the ephemeral per-request metadata."""
    return [message for message in history if not isinstance(message, _MetadataRequest)]


def interrupted_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Close unanswered calls without replaying them: cancellation can land between a model response and
    tool execution, before the engine marks its trailing message interrupted. A failed tool may also have
    applied effects before raising. The next prompt must be resumable without guessing those effects."""
    pending: dict[str, ToolCallPart] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                pending[part.tool_call_id] = part
            elif isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_name is not None:
                pending.pop(part.tool_call_id, None)
    if not pending:
        return messages
    return [
        *messages,
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                    content="The turn was interrupted before this tool returned. Its effects are unknown; "
                    "inspect the current state before retrying it.",
                    outcome="interrupted",
                )
                for call in pending.values()
            ]
        ),
    ]


def load_target(target: str) -> AbstractAgent[Any, Any]:
    module_name, _, attr_path = target.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not isinstance(obj, AbstractAgent):
        raise TypeError(f"{target} is {type(obj).__name__}, not a pydantic_ai agent")
    return obj  # pyright: ignore[reportUnknownVariableType] -- isinstance cannot narrow the generic parameters


def _session_catalog() -> Catalog:
    catalog = discover(agents_path())
    plugins = discover_plugins(plugins_path())
    catalog.problems.extend(plugins.problems)
    for name, tool in plugins.tools.items():
        # On a name clash the agents path wins: it is the established tool source.
        catalog.tools.setdefault(name, tool)
    return catalog


def load_named(catalog: Catalog, name: str) -> AbstractAgent[Any, Any]:
    if (cls := catalog.agents.get(name)) is None:
        known = ", ".join(sorted(catalog.agents)) or "none"
        problems = "".join(f"\n  {problem}" for problem in catalog.problems)
        raise LookupError(f"no agent {name!r} on {ENV_AGENTS_PATH} (found: {known}){problems}")
    return cls().build()


def aid_toolset(catalog: Catalog) -> FunctionToolset[Any]:
    tools = [Tool[Any](t.fn, takes_ctx=False, name=t.name, description=t.description) for t in catalog.tools.values()]
    return FunctionToolset[Any](tools)


def to_event(event: AgentStreamEvent) -> SessionEvent | None:
    match event:
        case PartStartEvent(part=TextPart(content=text)) | PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
            return TextDelta(text=text) if text else None
        case PartStartEvent(part=ThinkingPart(content=text)):
            return ThoughtDelta(text=text) if text else None
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text)):
            return ThoughtDelta(text=text) if text else None
        case FunctionToolCallEvent(part=part):
            args = part.args_as_dict()
            return ToolCall(
                tool_call_id=part.tool_call_id,
                title=part.tool_name,
                status="in_progress",
                input=to_json(args),
                paths=tool_paths(args),
            )
        case FunctionToolResultEvent(part=ToolReturnPart() as part):
            return ToolCall(tool_call_id=part.tool_call_id, status="completed", output=clip(part.model_response_str()))
        case FunctionToolResultEvent(part=RetryPromptPart() as part):
            return ToolCall(tool_call_id=part.tool_call_id, status="failed", output=clip(part.model_response()))
        case _:
            return None


def usage_of(result: AgentRunResult[Any] | AgentRunEvents[Any] | RunCancelled) -> Usage:
    usage = result.usage
    models = [m.model_name for m in result.new_messages() if isinstance(m, ModelResponse) and m.model_name]
    # Providers name thinking tokens differently in `details`.
    thought = usage.details.get("reasoning_tokens") or usage.details.get("thinking_tokens")
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        thought_tokens=thought,
        requests=usage.requests,
        models=list(dict.fromkeys(models)),
    )


def model_name(agent: AbstractAgent[Any, Any]) -> str | None:
    model = agent.model
    return model if isinstance(model, str) or model is None else model.model_name


class PydanticAIBackend:
    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        history_file: anyio.Path,
        history: list[ModelMessage],
        toolsets: list[AbstractToolset[Any]],
        started: Started,
        coding: Coding,
        desired_max: int = DEFAULT_DESIRED_MAX,
        max_requests: int | None = None,
        refresh_instructions: Callable[[], str] | None = None,
    ) -> None:
        self._agent = agent
        self._history_file = history_file
        self._history = history
        self._toolsets = toolsets
        self._started = started
        self._run: AgentRunEvents[Any] | None = None
        self._coding = coding
        self._desired_max = desired_max
        self._max_requests = max_requests
        self._refresh_instructions = refresh_instructions
        self._history_lock = anyio.Lock()
        coding.compact = self.compact

    def started(self) -> Started:
        return self._started

    def answer_permission(
        self,
        request_id: str,
        option_id: str | None,
        plugin: str | None = None,
        text: str | None = None,
        supervisor: str | None = None,
    ) -> bool:
        return self._coding.waits.answer(request_id, option_id, plugin, text, supervisor)

    async def prompt(self, text: str, emit: Emit) -> Output:
        # The tools pydantic-ai runs are tasks it starts inside the run below: they see this context.
        self._coding.emit = emit
        token = CODING.set(self._coding)
        try:
            return await self._prompt(text, emit)
        finally:
            CODING.reset(token)
            self._coding.waits.cancel_all()
            self._coding.emit = None

    async def _prompt(self, text: str, emit: Emit) -> Output:
        # None is aid's explicit unlimited policy, not pydantic-ai's implicit 50-request default.
        limits = UsageLimits(request_limit=self._max_requests)
        state: AgentRunResult[Any] | AgentRunEvents[Any] | RunCancelled | None = None
        completed = False
        try:
            async with self._agent.run_stream_events(
                text,
                message_history=self._history,
                toolsets=self._toolsets,
                capabilities=[ProcessHistory(_send_policy(len(self._history), self._desired_max))],
                usage_limits=limits,
            ) as run:
                self._run = run
                state = run
                async for event in run:
                    if isinstance(event, AgentRunResultEvent):
                        result = event.result
                        state = result
                        completed = True
                        break
                    if (session_event := to_event(event)) is not None:
                        await emit(session_event)
                else:
                    raise RuntimeError("pydantic_ai run ended without a result")
            return Output(output=to_jsonable_python(result.output), stop_reason="end_turn")
        except RunCancelled as error:
            state = error
            return Output(output=None, stop_reason="cancelled")
        except anyio.get_cancelled_exc_class() as error:
            # The context manager has drained the run and attached its detached partial-stream snapshot.
            state = RunCancelled.from_cancellation(error) or state
            raise
        finally:
            self._run = None
            if state is not None:
                # Completed work survives cancellation and failure, not just a successful final answer.
                with anyio.CancelScope(shield=True):
                    await self._finish_turn(state, emit, interrupted=not completed)

    async def _finish_turn(
        self, state: AgentRunResult[Any] | AgentRunEvents[Any] | RunCancelled, emit: Emit, *, interrupted: bool
    ) -> None:
        try:
            messages = without_metadata(state.new_messages())
            usage = usage_of(state)
        except UserError:
            # A run that failed before binding (or a custom AbstractAgent) exposes no partial state.
            log.warning("run_state_unavailable", exc_info=True)
            return
        if interrupted:
            messages = interrupted_messages(messages)
        async with self._history_lock:
            # Compaction owns the older prefix; the run owns only this turn's messages. Replacing the
            # whole history with all_messages() would silently undo an in-turn compact.
            self._history = [*self._history, *messages]
            try:
                await _write_atomic(self._history_file, ModelMessagesTypeAdapter.dump_json(self._history))
            except OSError:
                # Keep the in-memory conversation and the original turn outcome, as the daemon recorder does.
                log.exception("history_write_failed", path=str(self._history_file))
        await emit(usage)

    async def cancel(self) -> None:
        self._coding.waits.cancel_all()
        if self._run is not None:
            self._run.cancel()

    async def watch(self, notify: Notifier) -> None:
        """The session's monitor loop, for the worker's life: the coding tools own the watches, this only
        hosts them where turns cannot end them."""
        await self._coding.watch(notify)

    async def compact(self, instructions: str) -> str:
        """Summarize the history into a digest focused by `instructions`, and replace everything before the
        latest turn with it. A direct model call, no tools: the turn's own run stays out of it. The leading
        system instructions are re-composed from the providers, not kept: instruction files change, and a
        compaction is when a long session picks that up. Without a refresher they stay as they are.
        During a turn it changes only the older prefix: turn finalization appends the run's new messages."""
        async with self._history_lock:
            return await self._compact(instructions)

    async def _compact(self, instructions: str) -> str:
        tail_at = max(
            (i for i, m in enumerate(self._history) if any(p.part_kind == "user-prompt" for p in m.parts)),
            default=None,
        )
        lead = next(
            (i for i, m in enumerate(self._history) if any(p.part_kind == "user-prompt" for p in m.parts)),
            len(self._history),
        )
        if tail_at is None or not self._history[lead:tail_at]:
            return "history is short; nothing compacted"
        if self._refresh_instructions is None:
            head = self._history[:lead]
        else:
            fresh = await anyio.to_thread.run_sync(self._refresh_instructions)
            head = [ModelRequest(parts=[SystemPromptPart(content=fresh)])] if fresh else []
        if self._agent.model is None:
            raise AidError("no_model", "compaction needs the agent to name a model")
        model = infer_model(self._agent.model)
        request: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content="You summarize conversations so other turns can continue the work. "
                        "Keep decisions, file paths, and what remains to do."
                    )
                ]
            ),
            *head,
            *self._history[lead:tail_at],
            ModelRequest(
                parts=[UserPromptPart(content=f"Summarize this conversation for upcoming work: {instructions}")]
            ),
        ]
        response = await model.request(request, model_settings=None, model_request_parameters=ModelRequestParameters())
        digest = "\n".join(part.content for part in response.parts if isinstance(part, TextPart)).strip()
        if not digest:
            raise AidError("empty_digest", "the summarizer returned no text")
        self._history = [
            *head,
            ModelRequest(parts=[UserPromptPart(content=f"[Summary of earlier work]\n{digest}")]),
            *self._history[tail_at:],
        ]
        await _write_atomic(self._history_file, ModelMessagesTypeAdapter.dump_json(self._history))
        return digest


async def _write_atomic(path: anyio.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    await tmp.write_bytes(data)
    await tmp.rename(path)


@asynccontextmanager
async def open_pydantic_ai(spec: PydanticAISpec, state_dir: anyio.Path) -> AsyncGenerator[PydanticAIBackend]:
    # Imports and filesystem walks, run off the event loop. Instructions compose for every session: None
    # selects every provider, so the catalog is always needed.
    catalog = await anyio.to_thread.run_sync(_session_catalog)
    if spec.agent is not None:
        agent = await anyio.to_thread.run_sync(load_named, catalog, spec.agent)
    elif spec.target is not None:
        agent = await anyio.to_thread.run_sync(load_target, spec.target)
    else:
        raise ValueError("the spec names no agent")
    toolsets: list[AbstractToolset[Any]] = [aid_toolset(catalog)] if spec.aid_tools and catalog.tools else []
    history_file = state_dir / HISTORY_FILE
    history: list[ModelMessage] = (
        ModelMessagesTypeAdapter.validate_json(await history_file.read_bytes()) if await history_file.exists() else []
    )

    def refresh_instructions() -> str:
        return compose(
            spec.cwd,
            catalog,
            select=spec.instruction_providers,
            skip=frozenset(spec.skip_instructions),
            session_name=state_dir.name,
        )

    if not history:
        composed = await anyio.to_thread.run_sync(refresh_instructions)
        if composed:
            history = [ModelRequest(parts=[SystemPromptPart(content=composed)])]
    log.info("history_loaded", agent=spec.agent or spec.target, messages=len(history))
    started = Started(pid=os.getpid(), resumed=bool(history), agent=spec.agent or spec.target, model=model_name(agent))
    coding = Coding(
        cwd=Path(spec.cwd),
        store=Path(state_dir) / RUNS_DIR,
        outputs=Path(state_dir) / OUTPUTS_DIR,
        mode=spec.permission,
        timeout=spec.permission_timeout,
        autoselect_after=spec.ask_autoselect_after,
        hooks=tuple(spec.hooks),
        waits=PermissionWaits(),
    )
    try:
        yield PydanticAIBackend(
            agent,
            history_file,
            history,
            toolsets,
            started,
            coding,
            desired_max=spec.max_context if spec.max_context is not None else DEFAULT_DESIRED_MAX,
            max_requests=spec.max_requests,
            refresh_instructions=refresh_instructions,
        )
    finally:
        await coding.close()
