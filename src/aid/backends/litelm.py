"""Aid-owned agent execution over LiteLM transport."""

from __future__ import annotations

import importlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio
import anyio.to_thread
import litelm
import structlog
from anyio.lowlevel import checkpoint
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from aid.agent import Agent, ModelRetry, Tool, Toolset
from aid.agents import ENV_AGENTS_PATH, Catalog, agents_path, discover
from aid.backends.permissions import PermissionWaits
from aid.coding import CODING, OUTPUTS_DIR, Coding
from aid.instructions import compose, environment
from aid.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import LiteLMModel, Model, ModelInfo, RequestUsage, accumulate
from aid.plugins import discover as discover_plugins
from aid.plugins import plugins_path
from aid.protocol import (
    AidError,
    Output,
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

    from aid.backends.base import Emit, Notifier
    from aid.spec import LiteLMSpec

log = structlog.get_logger(__name__)

HISTORY_FILE = "history.json"
RUNS_DIR = "runs"
"""pyrun's store for what the session's `python` tool runs."""
THOUGHT_LIMIT: Final = 500
"""Characters of a thinking part kept for replay: thought serves the turn, not the replay."""
DEFAULT_DESIRED_MAX: Final = 100_000
"""Tokens the context gauge assumes when the session names no `max_context`. Rule of thumb: half the model's
window; unknown models err toward compacting early, which fails safe."""


class _MetadataRequest(ModelRequest):
    """Ephemeral per-request metadata: sent trailing every request, stripped before anything is stored."""


def prune_history(
    history: list[ModelMessage], boundary: int, *, preserve_signed_thinking: bool = False
) -> list[ModelMessage]:
    """Trim past thinking in the send view, without changing stored history.

    The turn's own thinking and signed Anthropic blocks stay whole. A fixed boundary keeps sends monotonic.
    """
    pruned: list[ModelMessage] = []
    for index, message in enumerate(history):
        past = index < boundary
        parts: list[Any] = []
        changed = False
        for part in message.parts:
            if past and isinstance(part, ThinkingPart):
                if preserve_signed_thinking and part.signature:
                    parts.append(part)
                    continue
                thought = part.content[:THOUGHT_LIMIT] + "…" if len(part.content) > THOUGHT_LIMIT else part.content
                parts.append(part.model_copy(update={"content": thought, "signature": None}))
                changed = True
            else:
                parts.append(part)
        pruned.append(message.model_copy(update={"parts": parts}) if changed else message)
    return pruned


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """The gauge's rough unit: the serialized size over four."""
    return len(ModelMessagesTypeAdapter.dump_json(messages)) // 4


def context_tokens(history: list[ModelMessage], boundary: int, *, preserve_signed_thinking: bool = False) -> int:
    """How many tokens a request over `history` would carry, in the gauge's own unit.

    Rough but monotonic, and the same estimate the per-request `context_pct` uses, so the number a turn
    reports as `context_used` matches what its requests measured.
    """
    return estimate_tokens(prune_history(history, boundary, preserve_signed_thinking=preserve_signed_thinking))


def _send_policy(
    boundary: int, desired_max: int, *, preserve_signed_thinking: bool = False
) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """The send policy for one turn: `boundary` is where the stored history ended when the turn
    started, so the turn's own thinking flows whole and everything before it rides pruned. A metadata message
    trails every send; the backend strips those before storing."""

    def policy(messages: list[ModelMessage]) -> list[ModelMessage]:
        clean = [message for message in messages if not isinstance(message, _MetadataRequest)]
        pruned = prune_history(clean, boundary, preserve_signed_thinking=preserve_signed_thinking)
        used = estimate_tokens(pruned)
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
    """Close unanswered calls without replaying them. A failed or cancelled tool can already have effects."""
    normalized: list[ModelMessage] = []
    for message in messages:
        calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
        ids = [call.tool_call_id for call in calls]
        if calls and (any(not identity for identity in ids) or len(ids) != len(set(ids))):
            message = message.model_copy(
                update={
                    "parts": [
                        *(part for part in message.parts if not isinstance(part, ToolCallPart)),
                        TextPart("The provider returned invalid tool-call identities. These calls did not execute."),
                    ],
                    "invalid_tool_calls": [call.model_dump(mode="json") for call in calls],
                }
            )
        normalized.append(message)
    messages = normalized
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


def load_target(target: str) -> Agent:
    module_name, _, attr_path = target.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not isinstance(obj, Agent):
        raise TypeError(f"{target} is {type(obj).__name__}, not an aid.Agent")
    return obj


def _session_catalog() -> Catalog:
    catalog = discover(agents_path())
    plugins = discover_plugins(plugins_path())
    catalog.problems.extend(plugins.problems)
    for name, tool in plugins.tools.items():
        # On a name clash the agents path wins: it is the established tool source.
        catalog.tools.setdefault(name, tool)
    return catalog


def load_named(catalog: Catalog, name: str) -> Agent:
    if (cls := catalog.agents.get(name)) is None:
        known = ", ".join(sorted(catalog.agents)) or "none"
        problems = "".join(f"\n  {problem}" for problem in catalog.problems)
        raise LookupError(f"no agent {name!r} on {ENV_AGENTS_PATH} (found: {known}){problems}")
    return cls().build()


def aid_toolset(catalog: Catalog) -> Toolset:
    tools = [Tool(t.fn, takes_ctx=False, name=t.name, description=t.description) for t in catalog.tools.values()]
    return Toolset(tools)


def to_event(part: ToolCallPart | ToolReturnPart | RetryPromptPart, kind: str | None = None) -> ToolCall:
    if isinstance(part, ToolCallPart):
        try:
            args = part.args_as_dict()
        except ValueError:
            return ToolCall(
                tool_call_id=part.tool_call_id,
                title=part.tool_name,
                kind=kind,
                status="in_progress",
                input=part.args_as_json_str(),
            )
        return ToolCall(
            tool_call_id=part.tool_call_id,
            title=part.tool_name,
            kind=kind,
            status="in_progress",
            input=to_json(args),
            paths=tool_paths(args),
        )
    if isinstance(part, ToolReturnPart):
        return ToolCall(tool_call_id=part.tool_call_id, status="completed", output=clip(part.model_response_str()))
    return ToolCall(tool_call_id=part.tool_call_id, status="failed", output=clip(part.model_response()))


class UsageLimitExceeded(AidError):
    def __init__(self, limit: int) -> None:
        super().__init__("request_limit", f"turn reached its {limit} model-request budget")


def model_name(agent: Agent) -> str | None:
    model = agent.model
    return model if isinstance(model, str) else model.model_name


class LiteLMBackend:
    def __init__(
        self,
        agent: Agent,
        history_file: anyio.Path,
        history: list[ModelMessage],
        toolsets: list[Toolset],
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
        self._scope: anyio.CancelScope | None = None
        self._model: Model = LiteLMModel(agent.model) if isinstance(agent.model, str) else agent.model
        tools: dict[str, Tool] = {}
        functions = [*agent.tools, *(fn for toolset in [*agent.toolsets, *toolsets] for fn in toolset.tools)]
        for fn in functions:
            tool = fn if isinstance(fn, Tool) else Tool(fn)
            if tool.name is None:
                raise ValueError("tool has no name")
            if tool.name in tools:
                raise ValueError(f"duplicate tool {tool.name!r}")
            tools[tool.name] = tool
        self._tools = tools
        self._coding = coding
        self._desired_max = desired_max
        self._max_requests = max_requests
        self._refresh_instructions = refresh_instructions
        self._history_lock = anyio.Lock()
        coding.compact = self.compact

    def started(self) -> Started:
        return self._started

    def _preserve_signed_thinking(self) -> bool:
        """Anthropic rejects a signed block whose signature the wire dropped, so keep them whole."""
        return self._model.model_name.startswith("anthropic/")

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
        self._coding.emit = emit
        token = CODING.set(self._coding)
        try:
            return await self._prompt(text, emit)
        finally:
            CODING.reset(token)
            self._coding.waits.cancel_all()
            self._coding.emit = None

    async def _prompt(self, text: str, emit: Emit) -> Output:
        messages: list[ModelMessage] = []
        if not self._history and self._agent.instructions:
            messages.append(ModelRequest(parts=[SystemPromptPart(self._agent.instructions)]))
        messages.append(ModelRequest(parts=[UserPromptPart(text)]))
        usage = Usage(requests=0, models=[])
        completed = False
        output: Any = None
        retries = 0
        scope = anyio.CancelScope()
        self._scope = scope
        try:
            with scope:
                while True:
                    await checkpoint()
                    if self._max_requests is not None and (usage.requests or 0) >= self._max_requests:
                        raise UsageLimitExceeded(self._max_requests)
                    usage = usage.model_copy(update={"requests": (usage.requests or 0) + 1})
                    response = ModelResponse(parts=[], model_name=self._model.model_name)
                    messages.append(response)
                    indexes: dict[int, int] = {}
                    request_usage = RequestUsage()
                    usage_observed = False
                    info = ModelInfo(list(self._tools.values()), self._agent.output_type, self._agent.model_settings)
                    wire = _send_policy(
                        len(self._history),
                        self._desired_max,
                        preserve_signed_thinking=self._preserve_signed_thinking(),
                    )([*self._history, *messages[:-1]])
                    try:
                        stream = self._model.stream(wire, info)
                        try:
                            async for delta in stream:
                                accumulate(response, indexes, delta)
                                if delta.model_name:
                                    response.model_name = delta.model_name
                                if delta.usage is not None:
                                    usage_observed = True
                                    for name in RequestUsage.__dataclass_fields__:
                                        setattr(
                                            request_usage,
                                            name,
                                            max(getattr(request_usage, name), getattr(delta.usage, name)),
                                        )
                                if isinstance(delta.part, TextPart) and delta.part.content:
                                    await emit(TextDelta(text=delta.part.content))
                                elif isinstance(delta.part, ThinkingPart) and delta.part.content:
                                    await emit(ThoughtDelta(text=delta.part.content))
                                await checkpoint()
                        finally:
                            with anyio.CancelScope(shield=True):
                                await stream.aclose()
                    finally:
                        if usage_observed:
                            usage = usage.model_copy(
                                update={
                                    "input_tokens": (usage.input_tokens or 0) + request_usage.input_tokens,
                                    "output_tokens": (usage.output_tokens or 0) + request_usage.output_tokens,
                                    "cache_read_tokens": (usage.cache_read_tokens or 0)
                                    + request_usage.cache_read_tokens,
                                    "cache_write_tokens": (usage.cache_write_tokens or 0)
                                    + request_usage.cache_write_tokens,
                                    "thought_tokens": (usage.thought_tokens or 0) + request_usage.reasoning_tokens,
                                }
                            )
                        resolved_model = response.model_name or self._model.model_name
                        if resolved_model not in usage.models:
                            usage = usage.model_copy(update={"models": [*usage.models, resolved_model]})
                    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
                    if not calls:
                        answer = "".join(part.content for part in response.parts if isinstance(part, TextPart))
                        try:
                            output = (
                                answer
                                if self._agent.output_type is str
                                else self._agent.output_adapter().validate_json(answer)
                            )
                        except ValidationError as error:
                            if retries >= self._agent.output_retries:
                                raise
                            retries += 1
                            messages.append(ModelRequest(parts=[RetryPromptPart(str(error))]))
                            continue
                        completed = True
                        break
                    ids = [call.tool_call_id for call in calls]
                    if any(not call_id for call_id in ids) or len(ids) != len(set(ids)):
                        raise AidError("invalid_tool_call", "response has missing or duplicate tool-call identities")
                    for call in calls:
                        try:
                            tool = self._tools.get(call.tool_name)
                            await emit(to_event(call, tool.kind if tool is not None else None))
                            await checkpoint()
                            if tool is None:
                                raise ModelRetry(f"unknown tool {call.tool_name!r}")
                            try:
                                arguments = call.args_as_dict()
                            except ValueError as error:
                                raise ModelRetry(str(error)) from error
                            result = await tool.call(arguments, call.tool_call_id)
                        except ModelRetry as error:
                            part = RetryPromptPart(str(error), tool_name=call.tool_name, tool_call_id=call.tool_call_id)
                        else:
                            part = ToolReturnPart(call.tool_name, to_jsonable_python(result), call.tool_call_id)
                        messages.append(ModelRequest(parts=[part]))
                        await emit(to_event(part))
            return Output(output=to_jsonable_python(output), stop_reason="end_turn" if completed else "cancelled")
        finally:
            self._scope = None
            with anyio.CancelScope(shield=True):
                if not completed:
                    messages = interrupted_messages(messages)
                async with self._history_lock:
                    self._history = [*self._history, *messages]
                    try:
                        await _write_atomic(self._history_file, ModelMessagesTypeAdapter.dump_json(self._history))
                    except OSError:
                        log.exception("history_write_failed", path=str(self._history_file))
                    # The turn's end state: what a request over the stored history would carry, against the
                    # gauge. The next turn sends exactly this (its boundary is the whole history) plus its prompt.
                    usage = usage.model_copy(
                        update={
                            "context_used": context_tokens(
                                self._history,
                                len(self._history),
                                preserve_signed_thinking=self._preserve_signed_thinking(),
                            ),
                            "context_size": self._desired_max,
                        }
                    )
                await emit(usage)

    async def cancel(self) -> None:
        self._coding.waits.cancel_all()
        if self._scope is not None:
            self._scope.cancel()

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
        model = self._model
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
        response = await model.request(request, ModelInfo(settings=self._agent.model_settings))
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
async def open_litelm(spec: LiteLMSpec, state_dir: anyio.Path) -> AsyncGenerator[LiteLMBackend]:
    # Imports and filesystem walks, run off the event loop. Instructions compose for every session: None
    # selects every provider, so the catalog is always needed.
    catalog = await anyio.to_thread.run_sync(_session_catalog)
    if spec.agent is not None:
        agent = await anyio.to_thread.run_sync(load_named, catalog, spec.agent)
    elif spec.target is not None:
        agent = await anyio.to_thread.run_sync(load_target, spec.target)
    else:
        raise ValueError("the spec names no agent")
    toolsets: list[Toolset] = [aid_toolset(catalog)] if spec.aid_tools and catalog.tools else []
    history_file = state_dir / HISTORY_FILE
    history: list[ModelMessage] = (
        ModelMessagesTypeAdapter.validate_json(await history_file.read_bytes()) if await history_file.exists() else []
    )
    # A loaded conversation, not the fresh system prompt seeded below: that seed is a start, not a resume.
    resumed = bool(history)

    def refresh_instructions() -> str:
        composed = compose(
            spec.cwd,
            catalog,
            select=spec.instruction_providers,
            skip=frozenset(spec.skip_instructions),
            session_name=state_dir.name,
        )
        return "\n\n".join(filter(None, [agent.instructions, composed, environment(spec.cwd, state_dir.name)]))

    if not history:
        composed = await anyio.to_thread.run_sync(refresh_instructions)
        if composed:
            history = [ModelRequest(parts=[SystemPromptPart(content=composed)])]
    log.info("history_loaded", agent=spec.agent or spec.target, messages=len(history))
    started = Started(pid=os.getpid(), resumed=resumed, agent=spec.agent or spec.target, model=model_name(agent))
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
        yield LiteLMBackend(
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
        with anyio.CancelScope(shield=True):
            try:
                await coding.close()
            finally:
                await litelm.close_litelm_async_clients()
