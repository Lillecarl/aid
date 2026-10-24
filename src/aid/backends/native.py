"""Aid's own agent loop, over the OpenAI and Anthropic SDKs (`aid.model`)."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio
import anyio.to_thread
import structlog
from anyio.lowlevel import checkpoint
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from aid import sandbox
from aid.agent import Agent, ModelRetry, Tool, Toolset
from aid.backends.permissions import PermissionWaits
from aid.catalog import session_catalog
from aid.client import connect
from aid.coding import CODING, OUTPUTS_DIR, Coding, asking_names
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
from aid.messaging import ARRIVED, arrived
from aid.model import Model, ModelInfo, OpenAIChatModel, RequestUsage, Transport, accumulate, model_for, split_name
from aid.prices import CURRENCY, Price, price_for
from aid.protocol import (
    AidError,
    Cost,
    Output,
    SessionTool,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    Usage,
    clip,
    to_json,
)
from aid.providers import catalog_model, resolve
from aid.tools import tool_paths

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from aid.backends.base import Emit, Notifier
    from aid.discovery import Catalog
    from aid.protocol import MessageEntry
    from aid.spec import NativeSpec, PermissionMode

log = structlog.get_logger(__name__)

HISTORY_FILE = "history.json"
RUNS_DIR = "runs"
"""pyrun's store for what the session's `python` tool runs."""
THOUGHT_LIMIT: Final = 500
"""Characters of a thinking part kept for replay: thought serves the turn, not the replay."""
FORKS_DIR: Final = "forks"
"""Snapshots `snapshot` writes for forks, under the session's state directory."""
FORK_SNAPSHOTS: Final = 16
"""Snapshots kept: a fork copies its seed when it opens, so only recent ones are still unread."""
NOT_FORKED: Final = "Not run in this fork: the parent session runs it."
DEFAULT_DESIRED_MAX: Final = 100_000
"""Tokens the context gauge assumes when the session names no `max_context`. Rule of thumb: half the model's
window; unknown models err toward compacting early, which fails safe."""
CONTINUE: Final = (
    "Your reply above stopped at the output token limit, mid-sentence. Continue it from exactly where it stopped, "
    "as if it had never been interrupted. Do not restart or summarize it."
)
"""The request after an answer cut at the output limit. DeepSeek's prefix completion, which would continue it in
place, refuses any history holding a tool call (measured 2026-10-23), and every session has tools. Not asked to
repeat its last words: Claude Haiku 5.5 refused that or restarted the answer in 2 of 3 tries, and continued this
plainly in 3 of 3 (2026-10-24). `join_continued` still overlaps a repeat a model gives on its own."""
OVERLAP: Final = 500
"""The longest overlap `join_continued` looks for."""
LEADERS: Final = " \t\n.\u2026"
"""What a continuation's repeat may open with before the repeated words."""
HOLD: Final = 200
"""Characters of a continuation held from the stream until its repeat of the answer is known and left out."""
SEAM_PUNCTUATION: Final = ",.;:!?\u2014"
"""Punctuation a space follows in prose, where `join_continued` restores one."""


METADATA: Final = "[Session metadata]"
"""The header of the metadata part each turn's opening request carries."""


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

    Rough but monotonic, and the same estimate a turn's `context_pct` uses, so the number a turn reports as
    `context_used` matches what it opened with.
    """
    return estimate_tokens(prune_history(history, boundary, preserve_signed_thinking=preserve_signed_thinking))


def join_continued(answer: str, more: str) -> str:
    """An answer and its continuation, joined at their longest overlap when the continuation repeats the answer's
    last words, as models do unasked.
    Measured on deepseek-flash, a repeat can open with "...", change case, or trade a no-break space for a plain one:
    the match ignores all three. A repeat starts at a word, so the overlap must too: a single repeated word counts,
    and a chance match inside a word does not. A repeat can also rewrite the word the cut split ("carbon-d", then
    "carbon dioxide"), so that word may give way to the repeat's. After the repeat, a paragraph break before a
    lowercase word is a space. Without a repeat, a continuation drops the space at the cut; every such seam
    measured fell between words, so one goes back between a word or punctuation and a word."""
    start = len(more) - len(more.lstrip(LEADERS))
    rest = more[start:]
    cut_word = max(answer.rfind(" "), answer.rfind("\n")) + 1
    for kept in (answer, answer[:cut_word] if 0 < cut_word < len(answer) else None):
        if kept is not None and (size := _overlap(kept, rest)):
            after = rest[size:]
            if after[:1].isspace() and "\n" in after[: len(after) - len(after.lstrip())]:
                stripped = after.lstrip()
                after = f" {stripped}" if stripped[:1].islower() else after
            return kept + after
    if any(char in ".\u2026" for char in more[:start]):
        more = rest
    if answer and more and more[0].isalnum() and (answer[-1].isalnum() or answer[-1] in SEAM_PUNCTUATION):
        return f"{answer} {more}"
    return answer + more


def unshown(answer: str, more: str) -> str:
    """What joining `more` adds to an `answer` already streamed. A rewritten cut word cannot be taken back, so the
    stream gets what follows the part both share."""
    joined = join_continued(answer, more)
    return joined[len(os.path.commonprefix([answer, joined])) :]


def _overlap(answer: str, more: str) -> int:
    """The longest start of `more` that ends `answer`, loosely compared, and starts at a word there or with
    punctuation (Claude Haiku 5.5 repeated the comma a cut ended on); 0 if none."""
    tail, head = _loose(answer[-OVERLAP:]), _loose(more[:OVERLAP])
    for size in range(min(len(tail), len(head)), 0, -1):
        boundary = size == len(tail) or tail[-size - 1] == " " or not head[0].isalnum()
        if boundary and tail.endswith(head[:size]):
            return size
    return 0


def _loose(text: str) -> str:
    """Case and whitespace folded, character for character, so lengths match the original."""
    return "".join(" " if char.isspace() else lower if len(lower := char.lower()) == 1 else char for char in text)


def open_turn(history: list[ModelMessage]) -> int:
    """How many messages at the end of `history` belong to its last turn: from the last user prompt on. A turn's
    first message is its prompt, a wake's included; messages a turn took between its steps (`ARRIVED`) are not."""
    starts = [
        i
        for i, message in enumerate(history)
        if any(isinstance(p, UserPromptPart) and not str(p.content).startswith(ARRIVED) for p in message.parts)
    ]
    return len(history) - starts[-1] if starts else 0


def turn_metadata(used: int, desired_max: int, cwd: str | None = None) -> UserPromptPart:
    """The metadata part of a turn's opening request, stored with it. Sent and stored once, never stripped: a
    provider's prefix cache and Anthropic's binding of thinking to the conversation before it both need the
    history to only grow (measured 2026-10-23 on claude-haiku-5-5: a stripped per-request part left the cache
    unread and flagged the replayed thinking `prefix_binding_mismatch`)."""
    meta: dict[str, object] = {"context_pct": round(used / desired_max * 100), "unixtime": int(time.time())}
    if cwd is not None:
        # The instructions name the directory once, at the start; a session can move since (`ChangeDirectory`).
        meta["cwd"] = cwd
    return UserPromptPart(content=f"{METADATA}\n{json.dumps(meta)}")


def interrupted_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Close unanswered calls without replaying them. A failed or cancelled tool can already have effects.
    A response the provider never started is dropped: there is nothing in it to keep."""
    normalized: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse) and not message.parts:
            continue
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
    return session_catalog()


def session_agent(agent: Agent, spec: NativeSpec) -> Agent:
    """The agent on the spec's model and settings. The agent's own settings follow only to the same provider."""
    if spec.model is None and not spec.model_settings:
        return agent
    model = spec.model or agent.model
    same = (
        isinstance(agent.model, str) and isinstance(model, str) and split_name(agent.model)[0] == split_name(model)[0]
    )
    inherited = agent.model_settings if spec.model is None or same else {}
    return dataclasses.replace(agent, model=model, model_settings={**inherited, **spec.model_settings})


def load_named(catalog: Catalog, name: str) -> Agent:
    if (cls := catalog.agents.get(name)) is None:
        known = ", ".join(sorted(catalog.agents)) or "none"
        problems = "".join(f"\n  {problem}" for problem in catalog.problems)
        raise LookupError(f"no agent {name!r} on the plugins path (found: {known}){problems}")
    return cls().build()


def aid_toolset(catalog: Catalog) -> Toolset:
    tools = [
        Tool(t.fn, takes_ctx=False, name=t.name, description=t.description, errors_to_model=True)
        for t in catalog.tools.values()
    ]
    return Toolset(tools)


def to_event(part: ToolCallPart | ToolReturnPart | RetryPromptPart, kind: str | None = None) -> ToolCall:
    if isinstance(part, ToolCallPart):
        try:
            args = part.args_as_dict()
        except ValueError:
            return ToolCall(
                tool_call_id=part.tool_call_id,
                tool=part.tool_name,
                title=part.tool_name,
                kind=kind,
                status="in_progress",
                input=part.args_as_json_str(),
            )
        return ToolCall(
            tool_call_id=part.tool_call_id,
            tool=part.tool_name,
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


TITLE_LIMIT: Final = 100
"""Characters of a gated call's arguments its permission request names in the title; `input` has them all."""


def tool_group(tool: Tool) -> str:
    """Where a tool comes from, by its function's module: `coding`, `aid`, or a plugin's top-level package."""
    module = getattr(tool.fn, "__module__", None) or ""
    if module == "aid.coding":
        return "coding"
    return module.split(".")[0] or "agent"


class NativeBackend:
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
        open_tail: int = 0,
        receive: Callable[[], Awaitable[list[MessageEntry]]] | None = None,
        transport: Transport | None = None,
        price_as: str | None = None,
    ) -> None:
        """`transport`: the agent's model, resolved from its name (`aid.providers.resolve`); the backend owns and
        prices it. Without one, a name resolves through `model_for` alone. `price_as`: the name models.dev prices it
        under, where the provider's prefix is not models.dev's id."""
        self._agent = agent
        self._history_file = history_file
        self._history = history
        self._toolsets = toolsets
        self._started = started
        self._scope: anyio.CancelScope | None = None
        self._owned: Transport | None = transport
        self._price_as = price_as
        model = agent.model if transport is None else transport
        if isinstance(model, str):
            model = self._owned = model_for(model)
        self._model: Model = model
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
        coding.asking = asking_names(tools.values())
        self._desired_max = desired_max
        self._max_requests = max_requests
        self._refresh_instructions = refresh_instructions
        # Messages at the history's end that the next turn sends unpruned: a fork's copy of the turn its parent was
        # in, which the parent's own requests sent whole. Pruned, the fork's prefix would part from the parent's at
        # the first long thought, and the prompt cache with it.
        self._open_tail = open_tail
        # Takes the messages waiting for the session (the daemon's inbox), between the steps of a turn.
        self._receive = receive
        self._history_lock = anyio.Lock()
        # A model object is its owner's to price; aid prices only the transports it builds from a name.
        self._priced = self._owned is None
        self._price: Price | None = None
        self._turn: list[ModelMessage] | None = None
        self._in_flight: tuple[list[ToolCallPart], int] | None = None
        coding.compact = self.compact
        coding.snapshot = self.snapshot

    def started(self) -> Started:
        return self._started

    def session_tools(self) -> list[SessionTool]:
        return [
            SessionTool(name=name, group=tool_group(tool), asks=tool.asks)
            for tool in self._tools.values()
            for name in (tool.name or "", *tool.permissions)
        ]

    def set_permissions(self, default: PermissionMode | None, tools: dict[str, PermissionMode | None]) -> None:
        if default is not None:
            self._coding.mode = default
        for name, mode in tools.items():
            if mode is None:
                self._coding.overrides.pop(name, None)
            else:
                self._coding.overrides[name] = mode

    async def aclose(self) -> None:
        """Closes the SDK clients of a model built from the agent's name; a model object is its owner's."""
        if self._owned is not None:
            await self._owned.aclose()

    async def _take_messages(self) -> list[MessageEntry]:
        """What waits for the session now. A failure costs the turn its messages, never the turn: they stay in the
        inbox, and the turn's end hands them to a turn of their own."""
        if self._receive is None:
            return []
        try:
            return await self._receive()
        except Exception:
            log.exception("receive_messages_failed")
            return []

    async def _request_price(self) -> Price | None:
        if not self._priced:
            self._priced = True
            self._price = await price_for(self._price_as or self._model.model_name)
        return self._price

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
        used = context_tokens(
            [*self._history, *messages, ModelRequest(parts=[UserPromptPart(text)])],
            max(0, len(self._history) - self._open_tail),
            preserve_signed_thinking=self._preserve_signed_thinking(),
        )
        messages.append(
            ModelRequest(parts=[UserPromptPart(text), turn_metadata(used, self._desired_max, str(self._coding.cwd))])
        )
        usage = Usage(requests=0, models=[])
        completed = False
        stop_reason = "end_turn"
        output: Any = None
        retries = 0
        so_far: str | None = None
        continued = 0
        scope = anyio.CancelScope()
        self._scope = scope
        self._turn = messages
        first = True
        try:
            with scope:
                while True:
                    await checkpoint()
                    # Between steps, as Claude Code hands a person's messages to a turn in progress: only before a
                    # request the turn sends anyway, so a message taken is always seen.
                    if not first and (taken := await self._take_messages()):
                        messages.append(ModelRequest(parts=[UserPromptPart(arrived(taken))]))
                    first = False
                    if self._max_requests is not None and (usage.requests or 0) >= self._max_requests:
                        raise UsageLimitExceeded(self._max_requests)
                    usage = usage.model_copy(update={"requests": (usage.requests or 0) + 1})
                    response = ModelResponse(parts=[], model_name=self._model.model_name)
                    messages.append(response)
                    indexes: dict[int, int] = {}
                    request_usage = RequestUsage()
                    usage_observed = False
                    price = await self._request_price()
                    truncated = False
                    info = ModelInfo(list(self._tools.values()), self._agent.output_type, self._agent.model_settings)
                    wire = prune_history(
                        [*self._history, *messages[:-1]],
                        max(0, len(self._history) - self._open_tail),
                        preserve_signed_thinking=self._preserve_signed_thinking(),
                    )
                    held = None if so_far is None else ""
                    try:
                        stream = self._model.stream(wire, info)
                        try:
                            async for delta in stream:
                                accumulate(response, indexes, delta)
                                if delta.model_name:
                                    response.model_name = delta.model_name
                                truncated = truncated or delta.truncated
                                if delta.usage is not None:
                                    usage_observed = True
                                    for name in RequestUsage.__dataclass_fields__:
                                        counts = [getattr(request_usage, name), getattr(delta.usage, name)]
                                        known = [count for count in counts if count is not None]
                                        setattr(request_usage, name, max(known) if known else None)
                                if isinstance(delta.part, TextPart) and delta.part.content:
                                    if held is None:
                                        await emit(TextDelta(text=delta.part.content))
                                    elif len(held := held + delta.part.content) >= HOLD:
                                        await emit(TextDelta(text=unshown(so_far or "", held)))
                                        held = None
                                elif isinstance(delta.part, ThinkingPart) and delta.part.content:
                                    await emit(ThoughtDelta(text=delta.part.content))
                                await checkpoint()
                            if held:
                                await emit(TextDelta(text=unshown(so_far or "", held)))
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
                                    "thought_tokens": usage.thought_tokens
                                    if request_usage.reasoning_tokens is None
                                    else (usage.thought_tokens or 0) + request_usage.reasoning_tokens,
                                }
                            )
                        if usage_observed and price is not None:
                            usage = usage.model_copy(
                                update={"cost": add_cost(usage.cost, price, request_usage, self._model)}
                            )
                        resolved_model = response.model_name or self._model.model_name
                        if resolved_model not in usage.models:
                            usage = usage.model_copy(update={"models": [*usage.models, resolved_model]})
                    calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
                    if not calls:
                        text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
                        answer = text if so_far is None else join_continued(so_far, text)
                        # A request that adds nothing to the answer (all thinking, or only the repeat) would be cut
                        # or stop the same way again.
                        grew = len(answer) > len(so_far or "")
                        limit = self._agent.max_continuations
                        if truncated and grew and (limit is None or continued < limit):
                            so_far, continued = answer, continued + 1
                            messages.append(ModelRequest(parts=[UserPromptPart(CONTINUE)]))
                            continue
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
                            so_far, continued = None, 0
                            messages.append(ModelRequest(parts=[RetryPromptPart(str(error))]))
                            continue
                        completed = True
                        if truncated or not grew:
                            stop_reason = "max_tokens"
                        break
                    so_far, continued = None, 0
                    ids = [call.tool_call_id for call in calls]
                    if any(not call_id for call_id in ids) or len(ids) != len(set(ids)):
                        raise AidError("invalid_tool_call", "response has missing or duplicate tool-call identities")
                    for index, call in enumerate(calls):
                        self._in_flight = (calls, index)
                        try:
                            tool = self._tools.get(call.tool_name)
                            await emit(to_event(call, tool.kind if tool is not None else None))
                            await checkpoint()
                            if tool is None:
                                raise ModelRetry(f"unknown tool {call.tool_name!r}")
                            try:
                                arguments = call.args_as_dict()
                            except ValueError as error:
                                if truncated:
                                    raise ModelRetry(f"{error}: the response hit its output token limit") from error
                                raise ModelRetry(str(error)) from error
                            if tool.asks or await self._coding.permit(
                                call.tool_call_id,
                                tool_name=call.tool_name,
                                title=f"{call.tool_name} {json.dumps(arguments, default=str)[:TITLE_LIMIT]}",
                                input=arguments,
                            ):
                                result = await tool.call(arguments, call.tool_call_id)
                            else:
                                result = f"not allowed: {call.tool_name} stays unrun"
                        except ModelRetry as error:
                            part = RetryPromptPart(str(error), tool_name=call.tool_name, tool_call_id=call.tool_call_id)
                        else:
                            part = ToolReturnPart(call.tool_name, to_jsonable_python(result), call.tool_call_id)
                        messages.append(ModelRequest(parts=[part]))
                        await emit(to_event(part))
            return Output(output=to_jsonable_python(output), stop_reason=stop_reason if completed else "cancelled")
        finally:
            self._scope = None
            self._turn = None
            self._in_flight = None
            with anyio.CancelScope(shield=True):
                if not completed:
                    messages = interrupted_messages(messages)
                async with self._history_lock:
                    self._history = [*self._history, *messages]
                    self._open_tail = 0
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

    async def snapshot(self, reply: str) -> Path:
        """Write the conversation as it stands inside the running tool call, for a fork to start from: the
        stored history, this turn so far, `reply` as this call's return, and the sibling calls after it closed
        as not run. Stored history alone lacks the turn: it is written when the turn ends."""
        if self._turn is None or self._in_flight is None:
            raise AidError("no_tool_call", "a snapshot is taken only inside a tool call")
        calls, index = self._in_flight
        call = calls[index]
        returns = [
            ToolReturnPart(call.tool_name, reply, call.tool_call_id),
            *(ToolReturnPart(later.tool_name, NOT_FORKED, later.tool_call_id) for later in calls[index + 1 :]),
        ]
        async with self._history_lock:
            messages = [*self._history, *self._turn, ModelRequest(parts=returns)]
        forks = self._history_file.parent / FORKS_DIR
        await forks.mkdir(exist_ok=True)
        path = forks / f"{uuid.uuid4().hex[:12]}.json"
        await _write_atomic(path, ModelMessagesTypeAdapter.dump_json(messages))
        kept = sorted([((await old.stat()).st_mtime, str(old)) async for old in forks.glob("*.json")])
        for _, old in kept[:-FORK_SNAPSHOTS]:
            await anyio.Path(old).unlink(missing_ok=True)
        return Path(path)

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


def add_cost(total: Cost | None, price: Price, usage: RequestUsage, model: Model) -> Cost:
    # Chat Completions counts cache reads inside prompt tokens; Anthropic's Messages API counts them apart.
    cached = usage.cache_read_tokens if isinstance(model, OpenAIChatModel) else 0
    amount = price.cost(
        max(0, usage.input_tokens - cached), usage.cache_read_tokens, usage.cache_write_tokens, usage.output_tokens
    )
    return Cost(amount=(total.amount if total is not None else 0.0) + amount, currency=CURRENCY)


async def _receive_messages(session: str) -> list[MessageEntry]:
    async with connect() as client:
        return await client.receive_messages(session, wait=0)


@asynccontextmanager
async def open_native(spec: NativeSpec, state_dir: anyio.Path) -> AsyncGenerator[NativeBackend]:
    # Imports and filesystem walks, run off the event loop. Instructions compose for every session: None
    # selects every provider, so the catalog is always needed.
    catalog = await anyio.to_thread.run_sync(_session_catalog)
    if spec.agent is not None:
        agent = await anyio.to_thread.run_sync(load_named, catalog, spec.agent)
    elif spec.target is not None:
        agent = await anyio.to_thread.run_sync(load_target, spec.target)
    else:
        raise ValueError("the spec names no agent")
    agent = session_agent(agent, spec)
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

    open_tail = 0
    if not history and spec.seed_history is not None:
        seed = await anyio.Path(spec.seed_history).read_bytes()
        history = ModelMessagesTypeAdapter.validate_json(seed)
        open_tail = open_turn(history)
        await _write_atomic(history_file, seed)
    elif not history:
        composed = await anyio.to_thread.run_sync(refresh_instructions)
        if composed:
            history = [ModelRequest(parts=[SystemPromptPart(content=composed)])]
    log.info("history_loaded", agent=spec.agent or spec.target, messages=len(history))
    started = Started(pid=os.getpid(), resumed=resumed, agent=spec.agent or spec.target, model=model_name(agent))
    if spec.sandbox is not None:
        await sandbox.check(spec.sandbox, spec.cwd)
    coding = Coding(
        cwd=Path(spec.cwd),
        store=Path(state_dir) / RUNS_DIR,
        outputs=Path(state_dir) / OUTPUTS_DIR,
        mode=spec.permission,
        timeout=spec.permission_timeout,
        autoselect_after=spec.ask_autoselect_after,
        hooks=tuple(spec.hooks),
        overrides=dict(spec.tool_permissions),
        waits=PermissionWaits(),
        sandbox=spec.sandbox,
    )
    transport = await resolve(agent.model, catalog.model_providers) if isinstance(agent.model, str) else None
    backend: NativeBackend | None = None
    try:
        backend = NativeBackend(
            agent,
            history_file,
            history,
            toolsets,
            started,
            coding,
            desired_max=spec.max_context if spec.max_context is not None else DEFAULT_DESIRED_MAX,
            max_requests=spec.max_requests,
            refresh_instructions=refresh_instructions,
            open_tail=open_tail,
            receive=functools.partial(_receive_messages, state_dir.name) if spec.aid_tools else None,
            transport=transport,
            price_as=catalog_model(agent.model, catalog.model_providers) if isinstance(agent.model, str) else None,
        )
        yield backend
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await coding.close()
            finally:
                if backend is not None:
                    await backend.aclose()
