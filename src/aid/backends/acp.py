"""Backend for an external ACP agent, spoken to over the agent's stdio."""

from __future__ import annotations

import contextlib
import math
import os
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, cast

import acp
import anyio
import structlog
from acp.connection import StreamDirection
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    ConfigOptionUpdate,
    ContentToolCallContent,
    DeniedOutcome,
    FileEditToolCallContent,
    Implementation,
    RequestPermissionResponse,
    SessionConfigOptionSelect,
    SessionNotification,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)
from pydantic import BaseModel, Field, ValidationError

from aid.backends.permissions import PermissionWaits
from aid.env import agent_environment
from aid.mcp import AID_TOOL_PREFIX, session_servers, to_acp
from aid.protocol import (
    Cost,
    Output,
    PermissionChoice,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    SessionEvent,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolDiff,
    Usage,
    clip,
    to_json,
    tool_text,
)
from aid.spec import PermissionMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp.client.connection import ClientSideConnection
    from acp.connection import StreamEvent
    from acp.schema import (
        AcpMcpServer,
        HttpMcpServer,
        InitializeResponse,
        McpCapabilities,
        McpServerStdio,
        PermissionOption,
        PromptResponse,
        SessionConfigOptionBoolean,
        SseMcpServer,
        ToolCallUpdate,
    )
    from anyio.streams.memory import MemoryObjectSendStream

    from aid.backends.base import Emit
    from aid.spec import AcpSpec

log = structlog.get_logger(__name__)

SESSION_ID_FILE = "acp-session-id"
START_TIMEOUT = 60

_ALLOW_KINDS = ("allow_once", "allow_always")
_REJECT_KINDS = ("reject_once", "reject_always")


def choose_permission(mode: PermissionMode, options: list[PermissionChoice]) -> str | None:
    """The option a policy picks; None cancels the request. `ask` refuses: it is the answer when nobody does."""
    kinds = _ALLOW_KINDS if mode is PermissionMode.ALLOW else _REJECT_KINDS
    for kind in kinds:
        for option in options:
            if option.kind == kind:
                return option.option_id
    return None


def permission_response(option_id: str | None) -> RequestPermissionResponse:
    if option_id is None:
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
    return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option_id))


def to_event(update: object) -> SessionEvent | None:
    match update:
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return TextDelta(text=text)
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return ThoughtDelta(text=text)
        case ToolCallStart() | ToolCallProgress():
            content = update.content or []
            texts = [
                c.content.text
                for c in content
                if isinstance(c, ContentToolCallContent) and isinstance(c.content, TextContentBlock)
            ]
            diffs = [
                ToolDiff(path=c.path, old=clip(c.old_text) if c.old_text is not None else None, new=clip(c.new_text))
                for c in content
                if isinstance(c, FileEditToolCallContent)
            ]
            output = "\n".join(texts) if texts else tool_text(update.raw_output)
            return ToolCall(
                tool_call_id=update.tool_call_id,
                title=update.title,
                kind=update.kind,
                status=update.status,
                input=to_json(update.raw_input),
                output=clip(output) if output else None,
                diffs=diffs,
                paths=[f"{loc.path}:{loc.line}" if loc.line else loc.path for loc in update.locations or []],
            )
        case _:
            return None


type ConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean


def model_of(options: list[ConfigOption] | None) -> str | None:
    """The current value of the session's model option: ACP's `model` category."""
    for option in options or []:
        if isinstance(option, SessionConfigOptionSelect) and option.category == "model":
            return option.current_value
    return None


class _ModelTokens(BaseModel):
    model: str


class _Quota(BaseModel):
    model_usage: list[_ModelTokens] = Field(default_factory=list[_ModelTokens])


class _PromptMeta(BaseModel):
    """claude-agent-acp's `_meta.quota`, outside the ACP schema: the turn's tokens per model it resolved."""

    quota: _Quota = Field(default_factory=_Quota)


def usage_of(response: PromptResponse, context: UsageUpdate | None, model: str | None) -> Usage:
    """The turn's Usage, from the prompt response and the last `usage_update` the turn sent."""
    try:
        models = [m.model for m in _PromptMeta.model_validate(response.field_meta or {}).quota.model_usage]
    except ValidationError:
        models = []
    tokens = response.usage
    return Usage(
        input_tokens=tokens.input_tokens if tokens else None,
        output_tokens=tokens.output_tokens if tokens else None,
        cache_read_tokens=tokens.cached_read_tokens if tokens else None,
        cache_write_tokens=tokens.cached_write_tokens if tokens else None,
        thought_tokens=tokens.thought_tokens if tokens else None,
        models=models or ([model] if model else []),
        context_used=context.used if context else None,
        context_size=context.size if context else None,
        session_cost=Cost(amount=context.cost.amount, currency=context.cost.currency)
        if context and context.cost
        else None,
    )


def permission_for(spec: AcpSpec, tool_name: str | None) -> PermissionMode:
    """aid's own tools are approved: whoever put them on the agents path chose them."""
    if spec.aid_tools and tool_name is not None and tool_name.startswith(AID_TOOL_PREFIX):
        return PermissionMode.ALLOW
    return spec.permission


def requested_tool(message: dict[str, Any]) -> tuple[str, str] | None:
    """(tool call id, tool name) of a raw `session/request_permission`.

    The name is claude-agent-acp's `toolCall.name`, outside the ACP schema, so the parsed request lacks it. The
    title is no substitute: for Bash it is the description the model wrote.
    """
    if message.get("method") != "session/request_permission":
        return None
    try:
        tool_call = _PermissionParams.model_validate(message.get("params")).tool_call
    except ValidationError:
        return None
    return tool_call.tool_call_id, tool_call.name


class _NamedToolCall(BaseModel):
    tool_call_id: str = Field(alias="toolCallId")
    name: str


class _PermissionParams(BaseModel):
    tool_call: _NamedToolCall = Field(alias="toolCall")


class _Client:
    """The `acp.Client` side. Session updates arrive through `AcpBackend.observe`, not here."""

    def __init__(self, spec: AcpSpec) -> None:
        self._spec = spec
        self._tool_names: dict[str, str] = {}
        self.waits = PermissionWaits()
        self.events: MemoryObjectSendStream[SessionEvent] | None = None
        """The running prompt's events; permission requests only come during one."""

    def observe(self, event: StreamEvent) -> None:
        """Runs in the receive loop, before the library dispatches the request to `request_permission`."""
        if event.direction is StreamDirection.INCOMING and (found := requested_tool(event.message)):
            self._tool_names[found[0]] = found[1]

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        tool_name = self._tool_names.pop(tool_call.tool_call_id, None)
        mode = permission_for(self._spec, tool_name)
        request = PermissionRequest(
            request_id=uuid.uuid4().hex,
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_name,
            title=tool_call.title,
            kind=tool_call.kind,
            input=to_json(tool_call.raw_input),
            options=[PermissionChoice(option_id=o.option_id, name=o.name, kind=o.kind) for o in options],
        )
        events = self.events
        if events is not None:
            events.send_nowait(request)
        if mode is PermissionMode.ASK and events is not None:
            refused = choose_permission(PermissionMode.DENY, request.options)
            decision = PermissionDecision(
                request_id=request.request_id, option_id=refused, by=PermissionDecider.TIMEOUT
            )
            with anyio.move_on_after(self._spec.permission_timeout):
                decision = await self.waits.wait(request)
        else:
            chosen = choose_permission(mode, request.options)
            decision = PermissionDecision(request_id=request.request_id, option_id=chosen, by=PermissionDecider.POLICY)
        log.info("permission_decided", tool=tool_name, title=tool_call.title, option=decision.option_id, by=decision.by)
        if events is not None:
            with contextlib.suppress(anyio.ClosedResourceError):  # The prompt ended while the request waited.
                events.send_nowait(decision)
        return permission_response(decision.option_id)

    async def session_update(self, session_id: str, update: object, **kwargs: Any) -> None:
        return None

    def on_connect(self, conn: object) -> None:
        return None


class AcpBackend:
    def __init__(self, conn: ClientSideConnection, client: _Client, session_id: str, started: Started) -> None:
        self._conn = conn
        self._client = client
        self._session_id = session_id
        self._started = started
        self._model = started.model
        self._context: UsageUpdate | None = None
        self._events: MemoryObjectSendStream[SessionEvent] | None = None

    def started(self) -> Started:
        return self._started

    def observe(self, event: StreamEvent) -> None:
        """Runs synchronously in the receive loop, so updates keep wire order and precede the prompt response.

        The acp library dispatches `session_update` to the client as separate tasks, which can run after
        `prompt()` has already returned.
        """
        if event.message.get("method") != "session/update":
            return
        try:
            notification = SessionNotification.model_validate(event.message.get("params"))
        except ValidationError:
            log.exception("malformed_session_update")
            return
        if notification.session_id != self._session_id:
            return
        match notification.update:
            case UsageUpdate() as usage:
                self._context = usage
            case ConfigOptionUpdate(config_options=options):
                self._model = model_of(options) or self._model
            case update:
                if self._events is not None and (session_event := to_event(update)) is not None:
                    self._events.send_nowait(session_event)

    async def prompt(self, text: str, emit: Emit) -> Output:
        send, receive = anyio.create_memory_object_stream[SessionEvent](math.inf)
        chunks: list[str] = []

        async def forward() -> None:
            async with receive:
                async for event in receive:
                    if isinstance(event, TextDelta):
                        chunks.append(event.text)
                    await emit(event)

        response: PromptResponse | None = None
        self._context = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(forward)
            response = await self._send_prompt(text, send)
        if response is None:
            raise RuntimeError("ACP prompt returned no response")
        if response.usage is not None or self._context is not None:
            await emit(usage_of(response, self._context, self._model))
        return Output(output="".join(chunks), stop_reason=response.stop_reason)

    async def _send_prompt(self, text: str, events: MemoryObjectSendStream[SessionEvent]) -> PromptResponse:
        self._events = self._client.events = events
        try:
            return await self._conn.prompt(session_id=self._session_id, prompt=[acp.text_block(text)])
        finally:
            self._client.waits.cancel_all()
            self._events = self._client.events = None
            events.close()

    async def cancel(self) -> None:
        # ACP: a client that cancels a turn answers its pending permission requests `cancelled`.
        self._client.waits.cancel_all()
        await self._conn.cancel(session_id=self._session_id)

    def answer_permission(
        self, request_id: str, option_id: str | None, plugin: str | None = None, text: str | None = None
    ) -> bool:
        return self._client.waits.answer(request_id, option_id, plugin, text)


@asynccontextmanager
async def open_acp(spec: AcpSpec, state_dir: anyio.Path) -> AsyncGenerator[AcpBackend]:
    id_file = state_dir / SESSION_ID_FILE
    backend: AcpBackend | None = None
    client = _Client(spec)

    def observe(event: StreamEvent) -> None:
        client.observe(event)
        if backend is not None:
            backend.observe(event)

    async with acp.spawn_agent_process(
        # Partial on purpose: the router answers method_not_found for what `ClientCapabilities()` does not advertise.
        cast("acp.Client", client),
        spec.command[0],
        *spec.command[1:],
        env=agent_environment(spec.env, inherit=spec.inherit_env),
        cwd=spec.cwd,
        observers=[observe],
    ) as (conn, _process):
        with anyio.fail_after(START_TIMEOUT):
            init = await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="aid", version=version("aid")),
            )
            caps = init.agent_capabilities
            if missing := unsupported_transports(spec, caps.mcp_capabilities if caps else None):
                raise RuntimeError(f"the agent does not take {' or '.join(missing)} MCP servers")
            servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] = [
                to_acp(server) for server in session_servers(spec, state_dir.name)
            ]
            session_id, resumed, options = await _resume(
                conn, spec, id_file, servers, can_load=bool(caps and caps.load_session)
            )
        await id_file.write_text(session_id)
        started = Started(
            pid=os.getpid(), agent_session=session_id, resumed=resumed, agent=agent_name(init), model=model_of(options)
        )
        backend = AcpBackend(conn, client, session_id, started)
        yield backend


def agent_name(init: InitializeResponse) -> str | None:
    info = init.agent_info
    return None if info is None else f"{info.name} {info.version}" if info.version else info.name


def unsupported_transports(spec: AcpSpec, caps: McpCapabilities | None) -> list[str]:
    """Stdio is mandatory in ACP; http and sse are capabilities the agent advertises."""
    wanted = {server.type for server in spec.mcp_servers} - {"stdio"}
    offered = {kind for kind in ("http", "sse") if caps and getattr(caps, kind)}
    return sorted(wanted - offered)


async def _resume(
    conn: ClientSideConnection,
    spec: AcpSpec,
    id_file: anyio.Path,
    servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio],
    *,
    can_load: bool,
) -> tuple[str, bool, list[ConfigOption] | None]:
    """The session id, whether it is the earlier one, and the session's config options."""
    # The same list on load as on new: claude-agent-acp restarts its query process when they differ.
    if can_load and await id_file.exists():
        previous = (await id_file.read_text()).strip()
        try:
            loaded = await conn.load_session(cwd=spec.cwd, session_id=previous, mcp_servers=servers)
        except acp.RequestError:
            log.warning("session_load_failed", session=previous, exc_info=True)
        else:
            return previous, True, loaded.config_options if loaded else None
    session = await conn.new_session(cwd=spec.cwd, mcp_servers=servers)
    return session.session_id, False, session.config_options
