"""Async Python API for the aid daemon.

async with aid.connect() as client:
    session = await client.create("reviewer", NativeSpec(cwd=".", target="agents:reviewer"))
    result = await session.run("Review this diff", output_type=Review)
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, overload

import anyio
import zmq
import zmq.asyncio
from pydantic import JsonValue, TypeAdapter

from aid import events, zap
from aid.control import ENV_AGENT_CONNECTION, AgentConnection
from aid.history import MAX_PAGE
from aid.paths import ENV_SESSION, default_paths
from aid.plugins import (
    INPROC_PLUGIN_EVENTS,
    INPROC_PLUGIN_PRIVATE_EVENTS,
    INPROC_PLUGINS,
    PLUGIN_FILE,
    SECRET_FILE,
    Grant,
    PluginSpec,
)
from aid.protocol import (
    AddPlugin,
    AgentCatalog,
    AidError,
    AnswerDirectory,
    AnswerPermission,
    CallPlugin,
    CallResult,
    Cancel,
    ChangeDirectory,
    CompactSession,
    CreateSession,
    DeleteSession,
    Delivery,
    Done,
    Event,
    Failure,
    GetHistory,
    GetPane,
    GetScreen,
    GetStatus,
    GetSummary,
    HistoryEntry,
    HistoryPage,
    Invocation,
    ListAgents,
    ListModels,
    ListPlugins,
    ListSessions,
    MessageEntry,
    ModelListing,
    Output,
    PaneAddress,
    PaneView,
    Prompt,
    ReceiveMessages,
    RemovePlugin,
    Reply,
    SendMessage,
    ServeCalls,
    SessionInfo,
    SessionInfosAdapter,
    SessionStatus,
    SessionSummary,
    SetPermissions,
    SetSandbox,
    SetSupervisor,
    SetTags,
    StartSession,
    StopSession,
    TextDelta,
    decode_reply,
    encode,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from aid.paths import Paths
    from aid.protocol import Request, SessionEvent, Usage
    from aid.spec import AgentSpec, PermissionMode, Sandbox

_MESSAGES = TypeAdapter(list[MessageEntry])
_PLUGINS = TypeAdapter(list[PluginSpec])


@dataclass(frozen=True)
class RunResult[T]:
    output: T
    text: str
    stop_reason: str


@dataclass(frozen=True)
class Endpoints:
    """Where a client's sockets connect: the control socket and events, or with `curve`, a plugin's sockets."""

    control: str
    events: str
    curve: tuple[str, zap.Keypair] | None = None
    """The daemon's public key and the caller's keypair."""
    trust_pem: str = ""

    def connect(self, sock: zmq.asyncio.Socket, endpoint: str) -> None:
        if self.curve is None:
            sock.connect(endpoint)
        else:
            zap.connect_curve(sock, endpoint, *self.curve, self.trust_pem)


async def plugin_endpoints(paths: Paths, name: str) -> Endpoints:
    """Plugin `name`'s sockets, with the secret key `aid plugin add` wrote and the running daemon's public key."""
    try:
        secret = (await anyio.Path(paths.plugin_dir(name), SECRET_FILE).read_text()).strip()
    except FileNotFoundError:
        raise AidError("not_found", f"no key for plugin {name!r} here: register it with `aid plugin add`") from None
    try:
        server = (await anyio.Path(paths.server_key).read_text()).strip()
    except FileNotFoundError:
        raise AidError("not_running", f"no {paths.server_key}: the daemon is not running") from None
    keys = zap.Keypair(public=zmq.curve_public(secret.encode()).decode(), secret=secret)
    try:
        grants = PluginSpec.model_validate_json(
            await anyio.Path(paths.plugin_dir(name), PLUGIN_FILE).read_bytes()
        ).grants
    except FileNotFoundError:
        grants = frozenset[Grant]()
    # The two events sockets publish different session lists: a plugin follows exactly one.
    events = paths.plugin_private_events if Grant.PRIVATE in grants else paths.plugin_events
    return Endpoints(paths.plugins, events, (server, keys))


def local_plugin_endpoints(*, private: bool = False) -> Endpoints:
    """A plugin's sockets inside the daemon: the inproc pair, no keys. Only tasks on the daemon's context can
    connect, and the daemon names each task it spawns, so identity holds by construction."""
    return Endpoints(INPROC_PLUGINS, INPROC_PLUGIN_PRIVATE_EVENTS if private else INPROC_PLUGIN_EVENTS)


@asynccontextmanager
async def spawn_plugin_client(ctx: zmq.asyncio.Context, name: str, *, private: bool = False) -> AsyncGenerator[Client]:
    """A plugin client for a task the daemon spawned as `name`: a DEALER on the daemon's context, whose
    identity the plugin loop reads as the plugin. Mirrors `connect`, minus the context, which the daemon
    owns: no `ctx.term()` here, and sockets it opens close with it. `private`: follow private sessions too."""
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, name.encode())
    endpoints = local_plugin_endpoints(private=private)
    sock.connect(endpoints.control)
    client = Client(sock, ctx, endpoints)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.read_replies)
            try:
                yield client
            finally:
                tg.cancel_scope.cancel()
    except BaseExceptionGroup as group:
        # The reader's task group wraps the caller's own exception; hand it back as the caller raised it.
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        raise
    finally:
        sock.close(linger=0)
        for opened in list(client.sockets):  # A socket still open would hold the daemon's context forever.
            opened.close(linger=0)


DAEMON_CALL_TIMEOUT: Final = 30.0
"""A daemon call slower than this means the daemon is gone: exit loudly and let the supervisor restart into a
fresh baseline, rather than hang on a reply a restarted daemon never sends."""


async def daemon_call[T](call: Awaitable[T], *, name: str = "aid") -> T:
    """Await a daemon call, exiting loudly when the daemon answers nothing: a restarted daemon never replies
    to requests sent before its restart, so hanging would outlive the outage silently."""
    try:
        with anyio.fail_after(DAEMON_CALL_TIMEOUT):
            return await call
    except TimeoutError:
        raise SystemExit(f"{name}: the daemon answered nothing for 30s") from None


SERVE_RENEW: Final = 30.0
"""Seconds between a serving plugin's `ServeCalls`: a restarted daemon routes its calls again within this."""

type CallHandler = Callable[[Invocation], Awaitable[JsonValue]]


async def register_plugin(
    client: Client, paths: Paths, name: str, grants: frozenset[Grant], *, public_key: str | None = None
) -> str:
    """Register plugin `name` with `grants`; return its public key. Without `public_key` a keypair is made, and the
    secret kept where `connect(plugin=name)` reads it."""
    keys = None if public_key else zap.Keypair.new()
    public = keys.public if keys is not None else public_key or ""
    await client.add_plugin(PluginSpec(name=name, public_key=public, grants=grants))
    secret = anyio.Path(paths.plugin_dir(name), SECRET_FILE)
    if keys is None:
        await secret.unlink(missing_ok=True)  # A secret made here before would not match the new key.
    else:
        await secret.touch(mode=0o600)
        await secret.write_text(keys.secret)
    return public


@asynccontextmanager
async def connect(paths: Paths | None = None, *, plugin: str | None = None) -> AsyncGenerator[Client]:
    """Connect with launch credentials inside an agent, plugin grants with `plugin`, or local administrator access."""
    paths = paths or default_paths()
    connection = os.environ.get(ENV_AGENT_CONNECTION)
    if connection is not None:
        if plugin is not None:
            raise AidError("forbidden", "an agent connection cannot select a plugin identity")
        try:
            credentials = AgentConnection.model_validate_json(connection)
        except ValueError:
            raise AidError("unauthenticated", "invalid agent control credentials") from None
        endpoints = Endpoints(
            credentials.endpoint,
            paths.events,
            (credentials.server_key, zap.Keypair(public=credentials.public_key, secret=credentials.secret_key)),
            credentials.trust_pem,
        )
    elif os.environ.get(ENV_SESSION):
        raise AidError("unauthenticated", "agent control credentials are missing")
    else:
        endpoints = await plugin_endpoints(paths, plugin) if plugin else Endpoints(paths.control, paths.events)
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    endpoints.connect(sock, endpoints.control)
    client = Client(sock, ctx, endpoints)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.read_replies)
            try:
                yield client
            finally:
                tg.cancel_scope.cancel()
    except BaseExceptionGroup as group:
        # The reader's task group wraps the caller's own exception; hand it back as the caller raised it.
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        raise
    finally:
        sock.close(linger=0)
        for opened in list(client.sockets):  # A socket still open would hold ctx.term() forever.
            opened.close(linger=0)
        ctx.term()


class Client:
    def __init__(self, sock: zmq.asyncio.Socket, ctx: zmq.asyncio.Context, endpoints: Endpoints) -> None:
        self._sock = sock
        self._ctx = ctx
        self._endpoints = endpoints
        self._send_lock = anyio.Lock()
        self._pending: dict[str, MemoryObjectSendStream[Reply]] = {}
        self._invocations: MemoryObjectSendStream[Invocation] | None = None
        self.sockets: set[zmq.asyncio.Socket] = set()

    @asynccontextmanager
    async def open_socket(self, socket_type: int) -> AsyncGenerator[zmq.asyncio.Socket]:
        """A socket of its own on the daemon, while the block runs: `zmq.DEALER` on the control socket, `zmq.SUB`
        on the events. For a relay that speaks the protocol itself: this client neither routes its replies nor
        checks what it sends."""
        endpoints = self._endpoints
        endpoint = endpoints.control if socket_type == zmq.DEALER else endpoints.events
        sock = self._ctx.socket(socket_type)
        endpoints.connect(sock, endpoint)
        self.sockets.add(sock)
        try:
            yield sock
        finally:
            self.sockets.discard(sock)
            sock.close(linger=0)

    @asynccontextmanager
    async def subscribed(
        self, *prefixes: bytes, idle: float | None = None
    ) -> AsyncGenerator[AsyncIterator[tuple[bytes, bytes] | None]]:
        """The daemon's published `[topic, payload]` messages under `prefixes` (`aid.events`), while the block runs;
        with `idle`, a None after each `idle` seconds with no message.

        A subscription reaches the daemon a moment after this returns, and what is published before then is
        missed: fetch a baseline after subscribing, as the `follow_*` methods do."""
        async with self.open_socket(zmq.SUB) as sock:
            for prefix in prefixes:
                sock.setsockopt(zmq.SUBSCRIBE, prefix)

            async def messages() -> AsyncIterator[tuple[bytes, bytes] | None]:
                while True:
                    # A poll with a timeout, not a cancelled receive: cancelling inside an async generator ends it.
                    if idle is not None and not await sock.poll(int(idle * 1000)):
                        yield None
                        continue
                    topic, payload = await sock.recv_multipart()
                    yield topic, payload

            yield messages()

    @overload
    def follow_sessions(self, *, idle: None = None) -> AsyncGenerator[list[SessionInfo]]: ...
    @overload
    def follow_sessions(self, *, idle: float) -> AsyncGenerator[list[SessionInfo] | None]: ...
    async def follow_sessions(self, *, idle: float | None = None) -> AsyncGenerator[list[SessionInfo] | None]:
        """The session list now, then each time it changes; with `idle`, None after each quiet `idle` seconds."""
        async with self.subscribed(events.SESSIONS, idle=idle) as published:
            yield await self.sessions()
            async for message in published:
                yield None if message is None else SessionInfosAdapter.validate_json(message[1])

    async def read_replies(self) -> None:
        while True:
            reply = decode_reply(await self._sock.recv())
            if isinstance(reply, Event | Done | Failure) and (stream := self._pending.get(reply.id)) is not None:
                stream.send_nowait(reply)
            elif isinstance(reply, Invocation) and self._invocations is not None:
                self._invocations.send_nowait(reply)

    async def serve_calls(self, handler: CallHandler, hooks: Sequence[Literal["message"]] = ()) -> None:
        """As a plugin, answer the calls and `hooks` the daemon routes to it with `handler`, each in a task of its
        own, until cancelled. A handler's `AidError` fails the call with its code; any other exception, as
        `plugin_failed`."""
        send, receive = anyio.create_memory_object_stream[Invocation](math.inf)
        self._invocations = send
        try:
            async with anyio.create_task_group() as tg, receive:
                tg.start_soon(self._renew_serving, list(hooks))
                async for invocation in receive:
                    tg.start_soon(self._answer, handler, invocation)
        finally:
            self._invocations = None
            send.close()

    async def _renew_serving(self, hooks: list[Literal["message"]]) -> None:
        while True:
            await daemon_call(self.call(ServeCalls(hooks=hooks)))
            await anyio.sleep(SERVE_RENEW)

    async def _answer(self, handler: CallHandler, invocation: Invocation) -> None:
        try:
            result = CallResult(call=invocation.id, data=await handler(invocation))
        except AidError as error:
            result = CallResult(call=invocation.id, code=error.code, message=error.message)
        except Exception as error:
            result = CallResult(call=invocation.id, code="plugin_failed", message=f"{type(error).__name__}: {error}")
        await daemon_call(self.call(result))

    @asynccontextmanager
    async def _exchange(self, request: Request) -> AsyncGenerator[MemoryObjectReceiveStream[Reply]]:
        send, receive = anyio.create_memory_object_stream[Reply](math.inf)
        self._pending[request.id] = send
        try:
            async with self._send_lock:
                await self._sock.send(encode(request))
            yield receive
        finally:
            del self._pending[request.id]
            send.close()
            receive.close()

    async def _events(self, request: Request) -> AsyncIterator[SessionEvent | Done]:
        async with self._exchange(request) as replies:
            async for reply in replies:
                match reply:
                    case Event():
                        yield reply.event
                    case Done():
                        yield reply
                        return
                    case Failure():
                        raise AidError(reply.code, reply.message)
                    case _:
                        pass

    async def call(self, request: Request) -> JsonValue:
        async for item in self._events(request):
            if isinstance(item, Done):
                return item.data
        raise AidError("closed", "daemon closed the exchange without a reply")

    async def create(self, name: str, spec: AgentSpec) -> Session:
        await self.call(CreateSession(name=name, spec=spec))
        return Session(self, name)

    async def sessions(self) -> list[SessionInfo]:
        return SessionInfosAdapter.validate_python(await self.call(ListSessions()))

    async def agents(self) -> AgentCatalog:
        """The `aid.AgentDefinition`s on the daemon's plugins path, and what failed to load."""
        return AgentCatalog.model_validate(await self.call(ListAgents()))

    async def models(self, provider: str | None = None) -> ModelListing:
        """Model providers, and `provider`'s models (`ListModels`)."""
        return ModelListing.model_validate(await self.call(ListModels(provider=provider)))

    def session(self, name: str) -> Session:
        return Session(self, name)

    async def add_plugin(self, spec: PluginSpec) -> None:
        """Register a plugin, or replace its key and grants. Only the control socket may."""
        await self.call(AddPlugin(spec=spec))

    async def plugins(self) -> list[PluginSpec]:
        return _PLUGINS.validate_python(await self.call(ListPlugins()))

    async def remove_plugin(self, name: str) -> None:
        await self.call(RemovePlugin(name=name))

    async def send_message(
        self,
        to: str,
        text: str,
        *,
        sender: str | None = None,
        usage: Usage | None = None,
        delivery: Delivery = Delivery.INTERRUPT,
    ) -> None:
        """Leave a message for session `to`; the daemon wakes it. `sender` names the session it is from, or None
        for a person; `usage` what the sender's work for it cost (`SendMessage.usage`); `delivery` whether a
        running turn takes it between its steps. Returns once the message is recorded, not answered."""
        await self.call(SendMessage(to=to, text=text, sender=sender, usage=usage, delivery=delivery))

    async def call_plugin(self, plugin: str, method: str, /, **args: JsonValue) -> JsonValue:
        """Call `method` on `plugin` (`Plugin.call`), as this connection's session, or as the operator."""
        return await self.call(CallPlugin(plugin=plugin, method=method, args=args))

    async def receive_messages(self, session: str, *, wait: float = 25) -> list[MessageEntry]:
        """Take the messages waiting for `session`, waiting up to `wait` seconds (`ReceiveMessages`)."""
        return _MESSAGES.validate_python(await self.call(ReceiveMessages(session=session, wait=wait)))


class Session:
    def __init__(self, client: Client, name: str) -> None:
        self._client = client
        self.name = name

    async def stream(self, text: str) -> AsyncIterator[SessionEvent]:
        """Yield the prompt's events. The last one is an `Output`."""
        async for item in self._client._events(Prompt(session=self.name, text=text)):  # pyright: ignore[reportPrivateUsage] -- Session is Client's own facade
            if not isinstance(item, Done):
                yield item

    @overload
    async def run(self, text: str) -> RunResult[JsonValue]: ...
    @overload
    async def run[T](self, text: str, output_type: type[T]) -> RunResult[T]: ...
    async def run[T](self, text: str, output_type: type[T] | None = None) -> RunResult[T] | RunResult[JsonValue]:
        chunks: list[str] = []
        output: Output | None = None
        async for event in self.stream(text):
            match event:
                case TextDelta():
                    chunks.append(event.text)
                case Output():
                    output = event
                case _:
                    pass
        if output is None:
            raise AidError("no_output", f"prompt to {self.name!r} ended without output")
        if output_type is None:
            return RunResult(output.output, "".join(chunks), output.stop_reason)
        typed = TypeAdapter(output_type).validate_python(output.output)
        return RunResult(typed, "".join(chunks), output.stop_reason)

    async def summary(self) -> SessionSummary:
        """The whole history totalled: tokens, models, agent sessions, cost, and the files tool calls touched."""
        return SessionSummary.model_validate(await self._client.call(GetSummary(session=self.name)))

    async def history(
        self, *, before: int | None = None, after: int | None = None, limit: int = 100, wait: float = 0
    ) -> HistoryPage:
        """A page of this session's history: the newest entries before `before`, the oldest after `after`, or
        the newest of all. Entry `seq` numbers are what `before` and `after` take. With `after`, `wait` waits up
        to that many seconds for a newer entry to be recorded."""
        request = GetHistory(session=self.name, before=before, after=after, limit=limit, wait=wait)
        return HistoryPage.model_validate(await self._client.call(request))

    async def screen(self, *, stylesheet: bool = False, since: int | None = None, wait: float = 25) -> PaneView:
        """What interactive Claude's pane shows, as HTML; with `stylesheet`, the CSS it is written against.

        With `since`, a revision already drawn, it answers once the pane has changed from it, or after `wait`
        seconds with the same revision back.
        """
        request = GetScreen(session=self.name, stylesheet=stylesheet, since=since, wait=wait)
        return PaneView.model_validate(await self._client.call(request))

    async def pane(self) -> PaneAddress:
        """Where interactive Claude's pane lives, for a relay that streams it with `libpymux.PaneStream`."""
        return PaneAddress.model_validate(await self._client.call(GetPane(session=self.name)))

    async def status(self) -> SessionStatus:
        return SessionStatus.model_validate(await self._client.call(GetStatus(session=self.name)))

    @overload
    def follow_status(self, *, idle: None = None) -> AsyncGenerator[SessionStatus]: ...
    @overload
    def follow_status(self, *, idle: float) -> AsyncGenerator[SessionStatus | None]: ...
    async def follow_status(self, *, idle: float | None = None) -> AsyncGenerator[SessionStatus | None]:
        """The status now, then each time it changes; with `idle`, None after each quiet `idle` seconds. A missing
        session raises, as `status` does."""
        async with self._client.subscribed(events.status_topic(self.name), idle=idle) as published:
            yield await self.status()
            async for message in published:
                yield None if message is None else SessionStatus.model_validate_json(message[1])

    @overload
    def follow_history(self, after: int = -1, *, idle: None = None) -> AsyncGenerator[HistoryEntry]: ...
    @overload
    def follow_history(self, after: int = -1, *, idle: float) -> AsyncGenerator[HistoryEntry | None]: ...
    async def follow_history(
        self, after: int = -1, *, idle: float | None = None
    ) -> AsyncGenerator[HistoryEntry | None]:
        """Every entry after seq `after`, in order and once each: what is recorded now, then each as it is
        recorded; with `idle`, None after each quiet `idle` seconds. An entry the subscription dropped shows as a
        jump in seq, and is fetched."""
        async with self._client.subscribed(events.history_topic(self.name), idle=idle) as published:
            last = after
            async for entry in self._entries_after(last):
                last = entry.seq
                yield entry
            async for message in published:
                if message is None:
                    yield None
                    continue
                entry = HistoryEntry.model_validate_json(message[1])
                if entry.seq > last + 1:
                    async for missed in self._entries_after(last):
                        last = missed.seq
                        yield missed
                if entry.seq == last + 1:
                    last = entry.seq
                    yield entry

    async def _entries_after(self, after: int) -> AsyncGenerator[HistoryEntry]:
        while True:
            page = await self.history(after=after, limit=MAX_PAGE)
            for entry in page.entries:
                yield entry
            if not page.has_newer or not page.entries:
                return
            after = page.entries[-1].seq

    async def cancel(self) -> None:
        await self._client.call(Cancel(session=self.name))

    async def compact(self, instructions: str | None = None) -> str:
        """Summarize the session's history into a digest, outside a turn; defaults to continuing the work."""
        if instructions is None:
            request = CompactSession(session=self.name)
        else:
            request = CompactSession(session=self.name, instructions=instructions)
        result = await self._client.call(request)
        if not isinstance(result, str):
            raise AidError("compact_failed", f"compaction returned {result!r}")
        return result

    async def answer(self, request_id: str, option_id: str | None, text: str | None = None) -> None:
        """Answer a PermissionRequest the session waits on: one of its options, or None to cancel it. `text`
        is your own words: with an option, alongside the pick; alone, the whole answer to an `ask_user`
        question."""
        await self._client.call(
            AnswerPermission(session=self.name, request_id=request_id, option_id=option_id, text=text)
        )

    async def set_permissions(
        self, tools: dict[str, PermissionMode | None] | None = None, default: PermissionMode | None = None
    ) -> None:
        """Change a native session's modes (`SetPermissions`): by tool name, None returning one to its default;
        `default` for the tools that ask for themselves."""
        await self._client.call(SetPermissions(session=self.name, default=default, tools=tools or {}))

    async def set_sandbox(self, sandbox: Sandbox | None) -> None:
        """Change a native session's sandbox (`SetSandbox`); None turns it off. A running worker restarts under
        it, when its turn ends if one runs."""
        await self._client.call(SetSandbox(session=self.name, sandbox=sandbox))

    async def set_supervisor(self, supervisor: str | None) -> None:
        """Change the session supervising this one (`SetSupervisor`); None ends supervision."""
        await self._client.call(SetSupervisor(session=self.name, supervisor=supervisor))

    async def set_tags(self, tags: Mapping[str, str | None]) -> None:
        """Set each tag to its value, or remove it when the value is None (`SetTags`); others stay."""
        await self._client.call(SetTags(session=self.name, tags=dict(tags)))

    async def change_directory(self, cwd: str) -> str:
        """Move the session to `cwd` (`ChangeDirectory`), relative to its current one; returns the absolute path."""
        result = await self._client.call(ChangeDirectory(session=self.name, cwd=cwd))
        if not isinstance(result, dict) or not isinstance(moved := result.get("cwd"), str):
            raise AidError("cd_failed", f"the move returned {result!r}")
        return moved

    async def answer_directory(self, approve: bool) -> None:
        """Approve or decline the agent's request to move (`SessionStatus.directory_request`)."""
        await self._client.call(AnswerDirectory(session=self.name, approve=approve))

    async def start(self) -> None:
        """Start the worker if it is stopped, with no turn. A prompt or a message starts it too."""
        await self._client.call(StartSession(session=self.name))

    async def stop(self) -> None:
        await self._client.call(StopSession(session=self.name))

    async def delete(self) -> None:
        await self._client.call(DeleteSession(session=self.name))
