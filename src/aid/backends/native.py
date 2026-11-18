"""Aid's own agent loop, over the OpenAI and Anthropic SDKs (`aid_agent.model`)."""

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
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import structlog
from pydantic_core import to_jsonable_python

from aid import sandbox
from aid.backends.permissions import PermissionWaits
from aid.catalog import session_catalog
from aid.client import connect
from aid.coding import CODING, OUTPUTS_DIR, Coding, asking_names, coding_tools
from aid.instructions import compose, environment
from aid.messaging import ARRIVED, arrived
from aid.protocol import (
    AidError,
    Output,
    SessionTool,
    Started,
)
from aid.providers import catalog_model, resolve
from aid.removals import PLANS_FILE
from aid.slots import Slotted, limited_hosts
from aid.supervision import supervisor_tools
from aid_agent.agent import Agent, Tool, Toolset
from aid_agent.attachments import AttachmentStore, Kind, checked, kind_of
from aid_agent.loop import Turn, context_tokens, run, signed_thinking, takes
from aid_agent.messages import (
    AttachmentPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid_agent.model import Model, ModelInfo, Transport, model_for
from aid_models.models_dev import catalog as models_catalog
from aid_models.models_dev import inputs, window
from aid_models.prices import Price, lookup
from aid_models.transport import Hosted, check_host, split_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from pydantic import JsonValue

    from aid.backends.base import Emit, Notifier
    from aid.discovery import Catalog
    from aid.protocol import MessageEntry
    from aid.spec import NativeSpec, PermissionMode

log = structlog.get_logger(__name__)

HISTORY_FILE = "history.json"
RUNS_DIR = "runs"
"""pyrun's store for what the session's `python` tool runs."""
FORKS_DIR: Final = "forks"
"""Snapshots `snapshot` writes for forks, under the session's state directory."""
FORK_SNAPSHOTS: Final = 16
"""Snapshots kept: a fork copies its seed when it opens, so only recent ones are still unread."""
NOT_FORKED: Final = "Not run in this fork: the parent session runs it."
DEFAULT_DESIRED_MAX: Final = 100_000
"""Tokens the context gauge assumes when the session names no `max_context` and models.dev knows no window for the
model; a known window gives half of it. Unknown models err toward compacting early, which fails safe."""
AUTO_COMPACT: Final = "keep what the next turns need to go on with the work, and every open request."


METADATA: Final = "[Session metadata]"
"""The header of the metadata part each turn's opening request carries."""


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
    meta: dict[str, JsonValue] = {"context_pct": round(used / desired_max * 100), "unixtime": int(time.time())}
    if cwd is not None:
        # The instructions name the directory once, at the start; a session can move since (`ChangeDirectory`).
        meta["cwd"] = cwd
    return UserPromptPart(content=f"{METADATA}\n{json.dumps(meta)}")


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
        Tool(t.fn, takes_ctx=False, name=t.name, description=t.description or "", errors_to_model=True)
        for t in catalog.tools.values()
    ]
    return Toolset(tools)


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


def session_tools(agent: Agent, toolsets: list[Toolset], offers: Callable[[str], bool] | None) -> dict[str, Tool]:
    """The tools the model is offered, by name: the agent's and `toolsets`', less what `offers` refuses."""
    tools: dict[str, Tool] = {}
    for fn in [*agent.tools, *(fn for toolset in [*agent.toolsets, *toolsets] for fn in toolset.tools)]:
        tool = fn if isinstance(fn, Tool) else Tool(fn)
        if offers is not None and not offers(tool.name):
            continue
        if tool.name in tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        tools[tool.name] = tool
    return tools


class NativeBackend:
    def __init__(
        self,
        agent: Agent,
        history_file: anyio.Path,
        history: list[ModelMessage],
        toolsets: list[Toolset],
        started: Started,
        coding: Coding,
        desired_max: int | None = None,
        max_requests: int | None = None,
        refresh_instructions: Callable[[], str] | None = None,
        open_tail: int = 0,
        receive: Callable[[], Awaitable[list[MessageEntry]]] | None = None,
        transport: Transport | None = None,
        price_as: str | None = None,
        fresh_turns: bool = False,
        offers: Callable[[str], bool] | None = None,
        attachments: AttachmentStore | None = None,
    ) -> None:
        """`transport`: the agent's model, resolved from its name (`aid.providers.resolve`); the backend owns and
        prices it. Without one, a name resolves through `model_for` alone. `price_as`: the name models.dev prices it
        under, where the provider's prefix is not models.dev's id. `desired_max`: the context gauge; None takes half
        the model's window from models.dev. `offers`: the session's tool ceiling; a tool it refuses is never offered."""
        self._agent = agent
        self._history_file = history_file
        self._attachments = attachments
        self._history = history
        self._fresh_turns = fresh_turns
        self._toolsets = toolsets
        self._started = started
        self._scope: anyio.CancelScope | None = None
        self._owned: Transport | None = transport
        self._price_as = price_as
        model = agent.model if transport is None else transport
        if isinstance(model, str):
            model = self._owned = model_for(model, agent.model_settings)
        self._model: Model = model
        tools = self._tools = session_tools(agent, toolsets, offers)
        self._coding = coding
        coding.asking = asking_names(tools.values())
        self._max_context = desired_max
        self._desired_max = desired_max or DEFAULT_DESIRED_MAX
        self._max_requests = max_requests
        self._refresh_instructions = refresh_instructions
        # Messages at the history's end that the next turn sends unpruned: a fork's copy of the turn its parent was
        # in, which the parent's own requests sent whole. Pruned, the fork's prefix would part from the parent's at
        # the first long thought, and the prompt cache with it.
        self._open_tail = open_tail
        # Takes the messages waiting for the session (the daemon's inbox), between the steps of a turn.
        self._receive = receive
        self._history_lock = anyio.Lock()
        self.compacted: Callable[[str, str], Awaitable[None]] | None = None
        """Told of a compaction the backend chose (detail, digest): the worker records it."""
        # A model object is its owner's to price; aid prices only the transports it builds from a name.
        self._looked_up = self._owned is None
        self._price: Price | None = None
        self._inputs: list[str] | None = None
        self._turn: Turn | None = None
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

    async def _look_up(self) -> None:
        """Price and context window from models.dev, once, for a model the backend resolved from a name."""
        if self._looked_up:
            return
        self._looked_up = True
        catalog = await models_catalog()
        if catalog is None:
            return
        name = self._price_as or self._model.model_name
        self._price = lookup(catalog, name)
        self._inputs = inputs(catalog, name)
        if self._max_context is None and (size := window(catalog, name)):
            self._desired_max = size // 2

    async def _takes(self, kind: Kind) -> bool:
        await self._look_up()
        return takes(self._model, self._inputs, kind)

    async def _permit(self, tool: Tool, call: ToolCallPart, arguments: dict[str, JsonValue]) -> str | None:
        verdict = await self._coding.permit(
            call.tool_call_id,
            tool_name=call.tool_name,
            title=f"{call.tool_name} {json.dumps(arguments, default=str)[:TITLE_LIMIT]}",
            input=arguments,
        )
        return None if verdict.allowed else verdict.refusal(f"{tool.name} stays unrun")

    async def _received(self) -> str | None:
        return arrived(taken) if (taken := await self._take_messages()) else None

    def answer_permission(
        self,
        request_id: str,
        option_id: str | None,
        plugin: str | None = None,
        text: str | None = None,
        supervisor: str | None = None,
    ) -> bool:
        return self._coding.waits.answer(request_id, option_id, plugin, text, supervisor)

    async def prompt(self, text: str, emit: Emit, attachments: Sequence[bytes] = ()) -> Output:
        if attachments and self._attachments is None:
            raise AidError("attachments_unsupported", "this session keeps no attachments")
        kinds: dict[Kind, None] = dict.fromkeys(kind_of(checked(attachment)) for attachment in attachments)
        for kind in kinds:
            if not await self._takes(kind):
                name = self._price_as or self._model.model_name
                listed = ", ".join(self._inputs or [])
                raise AidError("attachments_unsupported", f"{name} takes no {kind} (models.dev input: {listed})")
        parts = [await self._attachments.add(image) for image in attachments] if self._attachments else []
        self._coding.emit = emit
        token = CODING.set(self._coding)
        try:
            return await self._prompt(text, emit, parts)
        finally:
            CODING.reset(token)
            self._coding.waits.cancel_all()
            self._coding.emit = None

    async def _prompt(self, text: str, emit: Emit, attachments: Sequence[AttachmentPart]) -> Output:
        if self._fresh_turns:
            # Only the instructions before the first prompt stay: the same cached prefix for every turn.
            lead = next(
                (i for i, m in enumerate(self._history) if any(p.part_kind == "user-prompt" for p in m.parts)),
                len(self._history),
            )
            del self._history[lead:]
            self._open_tail = 0
        await self._auto_compact(text)
        await self._look_up()
        instructions = "" if self._history else self._agent.instructions
        opening = [ModelRequest(parts=[SystemPromptPart(instructions)])] if instructions else []
        used = context_tokens(
            [*self._history, *opening, ModelRequest(parts=[UserPromptPart(text)])],
            max(0, len(self._history) - self._open_tail),
            preserve_signed_thinking=signed_thinking(self._model),
        )
        prompt = [UserPromptPart(text), *attachments, turn_metadata(used, self._desired_max, str(self._coding.cwd))]
        turn = self._turn = Turn()
        scope = self._scope = anyio.CancelScope()
        try:
            with scope:
                await run(
                    self._model,
                    lambda: self._history,
                    prompt,
                    turn=turn,
                    instructions=instructions,
                    tools=list(self._tools.values()),
                    emit=emit,
                    permit=self._permit,
                    receive=self._received,
                    settings=self._agent.model_settings,
                    output_type=self._agent.output_type,
                    output_retries=self._agent.output_retries,
                    max_continuations=self._agent.max_continuations,
                    max_requests=self._max_requests,
                    attachments=self._attachments,
                    price=self._price,
                    inputs=self._inputs,
                    open_tail=self._open_tail,
                )
            return Output(output=to_jsonable_python(turn.output), stop_reason=turn.stop_reason or "cancelled")
        finally:
            self._scope = None
            self._turn = None
            with anyio.CancelScope(shield=True):
                async with self._history_lock:
                    self._history = [*self._history, *turn.messages]
                    self._open_tail = 0
                    try:
                        await _write_atomic(self._history_file, ModelMessagesTypeAdapter.dump_json(self._history))
                    except OSError:
                        log.exception("history_write_failed", path=str(self._history_file))
                    # The turn's end state: what a request over the stored history would carry, against the
                    # gauge. The next turn sends exactly this (its boundary is the whole history) plus its prompt.
                    usage = turn.usage.model_copy(
                        update={
                            "context_used": context_tokens(
                                self._history,
                                len(self._history),
                                preserve_signed_thinking=signed_thinking(self._model),
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
        if self._turn is None or self._turn.in_flight is None:
            raise AidError("no_tool_call", "a snapshot is taken only inside a tool call")
        calls, index = self._turn.in_flight
        call = calls[index]
        returns = [
            ToolReturnPart(call.tool_name, reply, call.tool_call_id),
            *(ToolReturnPart(later.tool_name, NOT_FORKED, later.tool_call_id) for later in calls[index + 1 :]),
        ]
        async with self._history_lock:
            messages = [*self._history, *self._turn.messages, ModelRequest(parts=returns)]
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
            digest = await self._compact(instructions)
        return "history is short; nothing compacted" if digest is None else digest

    async def _auto_compact(self, prompt: str) -> None:
        """Compact before a turn whose opening request would reach the gauge. Measured (campaign supervisor,
        2026-10-30): a session nothing compacts grows past the model's window, and then every turn fails."""
        if self._fresh_turns:
            return
        opening = [*self._history, ModelRequest(parts=[UserPromptPart(prompt)])]
        used = context_tokens(opening, len(self._history), preserve_signed_thinking=signed_thinking(self._model))
        if used < self._desired_max:
            return
        try:
            async with self._history_lock:
                digest = await self._compact(AUTO_COMPACT)
        except Exception:
            # The turn may still fit the model's window: the gauge is half of it by default.
            log.exception("auto_compact_failed")
            return
        if digest is not None and self.compacted is not None:
            await self.compacted("auto", digest)

    async def _compact(self, instructions: str) -> str | None:
        tail_at = max(
            (i for i, m in enumerate(self._history) if any(p.part_kind == "user-prompt" for p in m.parts)),
            default=None,
        )
        lead = next(
            (i for i, m in enumerate(self._history) if any(p.part_kind == "user-prompt" for p in m.parts)),
            len(self._history),
        )
        if tail_at is None or not self._history[lead:tail_at]:
            return None
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
        response = await model.request(
            request, ModelInfo(settings=self._agent.model_settings, attachments=self._attachments)
        )
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
    elif spec.instructions is not None and spec.model is not None:
        agent = Agent(spec.model, instructions=spec.instructions, toolsets=[coding_tools, supervisor_tools])
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

    definition = catalog.agents.get(spec.agent) if spec.agent is not None else None
    select = spec.instruction_providers
    if select is None and definition is not None and definition.instruction_providers is not None:
        select = list(definition.instruction_providers)

    offered = session_tools(agent, toolsets, spec.offers)
    runs_programs = any(tool.kind == "execute" for tool in offered.values())
    moves = "change_directory" in offered

    def refresh_instructions() -> str:
        composed = compose(
            spec.cwd,
            catalog,
            select=select,
            skip=frozenset(spec.skip_instructions),
            session_name=state_dir.name,
            tools=frozenset(offered),
        )
        context = environment(spec.cwd, state_dir.name, runs_programs=runs_programs, moves=moves)
        return "\n\n".join(filter(None, [agent.instructions, composed, context]))

    attachments = AttachmentStore(Path(state_dir))
    open_tail = 0
    if not history and spec.seed_history is not None:
        seed = await anyio.Path(spec.seed_history).read_bytes()
        history = ModelMessagesTypeAdapter.validate_json(seed)
        open_tail = open_turn(history)
        if spec.parent is not None:
            await attachments.take(AttachmentStore(Path(state_dir.parent / spec.parent)), history)
        await _write_atomic(history_file, seed)
    elif not history:
        composed = await anyio.to_thread.run_sync(refresh_instructions)
        if composed:
            history = [ModelRequest(parts=[SystemPromptPart(content=composed)])]
    log.info("history_loaded", agent=spec.runs, messages=len(history))
    started = Started(pid=os.getpid(), resumed=resumed, agent=spec.runs, model=model_name(agent))
    if spec.sandbox is not None:
        await sandbox.check(spec.sandbox, spec.cwd)
    coding = Coding(
        cwd=Path(spec.cwd),
        store=Path(state_dir) / RUNS_DIR,
        outputs=Path(state_dir) / OUTPUTS_DIR,
        plans_file=Path(state_dir) / PLANS_FILE,
        mode=spec.permission,
        timeout=spec.permission_timeout,
        autoselect_after=spec.ask_autoselect_after,
        hooks=tuple(spec.hooks),
        overrides=dict(spec.tool_permissions),
        waits=PermissionWaits(),
        sandbox=spec.sandbox,
    )
    transport = (
        await resolve(agent.model, catalog.model_providers, state_dir.name, agent.model_settings)
        if isinstance(agent.model, str)
        else None
    )
    backend: NativeBackend | None = None
    try:
        if spec.private is not None:
            # Before the first request, on the settings every request and compaction use.
            check_host(transport or agent.model, agent.model_settings, spec.private.hosts)
        # Every request to a host the daemon limits waits for one of its slots: compaction too, through the model.
        model: object = transport if transport is not None else agent.model
        if (
            (limited := limited_hosts())
            and isinstance(model, Hosted)
            and (host := model.host(agent.model_settings)) in limited
        ):
            if transport is not None:
                transport = Slotted(transport, host)
            elif not isinstance(agent.model, str):
                agent = dataclasses.replace(agent, model=Slotted(agent.model, host))
        backend = NativeBackend(
            agent,
            history_file,
            history,
            toolsets,
            started,
            coding,
            desired_max=spec.max_context,
            max_requests=spec.max_requests,
            refresh_instructions=refresh_instructions,
            open_tail=open_tail,
            receive=functools.partial(_receive_messages, state_dir.name) if spec.aid_tools else None,
            transport=transport,
            price_as=catalog_model(agent.model, catalog.model_providers) if isinstance(agent.model, str) else None,
            fresh_turns=spec.fresh_turns or (definition is not None and definition.fresh_turns),
            offers=spec.offers,
            attachments=attachments,
        )
        yield backend
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await coding.close()
            finally:
                if backend is not None:
                    await backend.aclose()
                elif transport is not None:
                    await transport.aclose()
