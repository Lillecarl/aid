"""Backend for an external ACP agent, spoken to over the agent's stdio."""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, cast

import acp
import anyio
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    Implementation,
    RequestPermissionResponse,
    SessionNotification,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)
from pydantic import ValidationError

from aid.protocol import Output, SessionEvent, TextDelta, ThoughtDelta, ToolCall
from aid.spec import PermissionMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp.client.connection import ClientSideConnection
    from acp.connection import StreamEvent
    from acp.schema import PermissionOption, PromptResponse, ToolCallUpdate
    from anyio.streams.memory import MemoryObjectSendStream

    from aid.backends.base import Emit
    from aid.spec import AcpSpec

log = logging.getLogger(__name__)

SESSION_ID_FILE = "acp-session-id"
START_TIMEOUT = 60

_ALLOW_KINDS = ("allow_once", "allow_always")
_REJECT_KINDS = ("reject_once", "reject_always")


def choose_permission(mode: PermissionMode, options: list[PermissionOption]) -> RequestPermissionResponse:
    kinds = _ALLOW_KINDS if mode is PermissionMode.ALLOW else _REJECT_KINDS
    for kind in kinds:
        for option in options:
            if option.kind == kind:
                return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option.option_id))
    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def to_event(update: object) -> SessionEvent | None:
    match update:
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return TextDelta(text=text)
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return ThoughtDelta(text=text)
        case ToolCallStart() | ToolCallProgress():
            return ToolCall(
                tool_call_id=update.tool_call_id,
                title=update.title,
                kind=update.kind,
                status=update.status,
            )
        case _:
            return None


class _Client:
    """The `acp.Client` side. Session updates arrive through `AcpBackend.observe`, not here."""

    def __init__(self, permission: PermissionMode) -> None:
        self._permission = permission

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        response = choose_permission(self._permission, options)
        log.info("permission %s for %r: %s", self._permission, tool_call.title, response.outcome)
        return response

    async def session_update(self, session_id: str, update: object, **kwargs: Any) -> None:
        return None

    def on_connect(self, conn: object) -> None:
        return None


class AcpBackend:
    def __init__(self, conn: ClientSideConnection, session_id: str) -> None:
        self._conn = conn
        self._session_id = session_id
        self._events: MemoryObjectSendStream[SessionEvent] | None = None

    def observe(self, event: StreamEvent) -> None:
        """Runs synchronously in the receive loop, so updates keep wire order and precede the prompt response.

        The acp library dispatches `session_update` to the client as separate tasks, which can run after
        `prompt()` has already returned.
        """
        if self._events is None or event.message.get("method") != "session/update":
            return
        try:
            notification = SessionNotification.model_validate(event.message.get("params"))
        except ValidationError:
            log.exception("malformed session/update")
            return
        if notification.session_id != self._session_id:
            return
        if (session_event := to_event(notification.update)) is not None:
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
        async with anyio.create_task_group() as tg:
            tg.start_soon(forward)
            response = await self._send_prompt(text, send)
        if response is None:
            raise RuntimeError("ACP prompt returned no response")
        return Output(output="".join(chunks), stop_reason=response.stop_reason)

    async def _send_prompt(self, text: str, events: MemoryObjectSendStream[SessionEvent]) -> PromptResponse:
        self._events = events
        try:
            return await self._conn.prompt(session_id=self._session_id, prompt=[acp.text_block(text)])
        finally:
            self._events = None
            events.close()

    async def cancel(self) -> None:
        await self._conn.cancel(session_id=self._session_id)


def agent_environment(spec: AcpSpec) -> dict[str, str]:
    return {**os.environ, **spec.env} if spec.inherit_env else dict(spec.env)


@asynccontextmanager
async def open_acp(spec: AcpSpec, state_dir: anyio.Path) -> AsyncGenerator[AcpBackend]:
    id_file = state_dir / SESSION_ID_FILE
    backend: AcpBackend | None = None

    def observe(event: StreamEvent) -> None:
        if backend is not None:
            backend.observe(event)

    async with acp.spawn_agent_process(
        # Partial on purpose: the router answers method_not_found for what `ClientCapabilities()` does not advertise.
        cast("acp.Client", _Client(spec.permission)),
        spec.command[0],
        *spec.command[1:],
        env=agent_environment(spec),
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
            session_id = await _resume(conn, spec, id_file, can_load=bool(caps and caps.load_session))
        await id_file.write_text(session_id)
        backend = AcpBackend(conn, session_id)
        yield backend


async def _resume(conn: ClientSideConnection, spec: AcpSpec, id_file: anyio.Path, *, can_load: bool) -> str:
    if can_load and await id_file.exists():
        previous = (await id_file.read_text()).strip()
        try:
            await conn.load_session(cwd=spec.cwd, session_id=previous, mcp_servers=[])
        except acp.RequestError:
            log.warning("agent could not load session %s; starting a new one", previous, exc_info=True)
        else:
            return previous
    session = await conn.new_session(cwd=spec.cwd, mcp_servers=[])
    return session.session_id
