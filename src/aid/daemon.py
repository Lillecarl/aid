"""The daemon: owns sessions, starts workers, and routes requests and replies between clients and workers."""

from __future__ import annotations

import contextlib
import logging
import shlex
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import zmq
import zmq.asyncio
from pydantic import ValidationError

from aid import events, plugins, zap
from aid.history import HISTORY_FILE, HistoryLog, Recorder
from aid.launcher import WorkerArgs
from aid.plugins import PLUGIN_FILE, PluginSpec
from aid.protocol import (
    Activity,
    AddPlugin,
    AgentCatalog,
    AidError,
    AnswerPermission,
    Cancel,
    CreateSession,
    DeleteSession,
    Done,
    Event,
    Failure,
    GetHistory,
    GetPane,
    GetScreen,
    GetStatus,
    GetSummary,
    Hello,
    Hook,
    ListAgents,
    ListPlugins,
    ListSessions,
    MessageEntry,
    Observed,
    PermissionDecision,
    PermissionRequest,
    Prompt,
    ReceiveMessages,
    RemovePlugin,
    SendMessage,
    SessionInfo,
    SessionInfosAdapter,
    SessionStatus,
    Started,
    StartFailed,
    StartSession,
    StopSession,
    TextDelta,
    ThoughtDelta,
    Usage,
    decode_reply,
    decode_request,
    encode,
)
from aid.spec import BUILTIN_MCP_SERVER, AcpSpec, AgentKind, AgentSpecAdapter, ClaudeTtySpec, PydanticAISpec
from aid.summary import summarize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyio.abc import TaskGroup, TaskStatus

    from aid.launcher import Launcher, WorkerHandle
    from aid.paths import Paths
    from aid.protocol import HistoryEntry, HistoryItem, Request
    from aid.spec import AgentSpec

log = logging.getLogger(__name__)

START_TIMEOUT: Final = 60.0
START_ERROR_GRACE: Final = 0.5
CATALOG_TIMEOUT: Final = 60.0
STOP_TIMEOUT: Final = 10.0
SPEC_FILE: Final = "spec.json"
# A spec holds env values and MCP headers, which are often credentials.
SESSION_DIR_MODE: Final = 0o700


@dataclass
class _Session:
    name: str
    spec: AgentSpec
    history: HistoryLog
    handle: WorkerHandle | None = None
    key: str | None = None
    """The CURVE public key issued to the running worker."""
    peer: bytes | None = None
    """The routing id of the worker's connection, from its Hello."""
    ready: anyio.Event = field(default_factory=anyio.Event)
    exited: anyio.Event = field(default_factory=anyio.Event)
    exit_code: int | None = None
    start_error: str | None = None
    start_failed: anyio.Event = field(default_factory=anyio.Event)
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    turn: str | None = None
    """The prompt running now, a client's or a wake. The worker takes one at a time."""
    inbox: list[MessageEntry] = field(default_factory=list[MessageEntry])
    """Messages not yet handed over: to a wake turn, or to the channel server of interactive Claude."""
    mail: anyio.Event = field(default_factory=anyio.Event)
    """Set when `inbox` gains a message, for a channel server waiting on it."""
    started: Started | None = None
    """What the last worker said about its agent. Kept after it exits: the agent session carries on."""
    models: list[str] = field(default_factory=list[str])
    """What the latest turn's Usage named."""
    permissions: dict[str, PermissionRequest] = field(default_factory=dict[str, PermissionRequest])
    """Requests the agent waits on, by request id."""
    working: bool = False
    """In a turn aid did not start, as the worker's Activity says."""
    attention: str | None = None

    @property
    def running(self) -> bool:
        return self.handle is not None and self.ready.is_set() and not self.exited.is_set()

    def note(self, item: HistoryItem) -> None:
        """What a turn's entry changes in the status, whoever started the turn."""
        match item:
            case Usage(models=models, agent=None) if models:
                self.models = models
            case PermissionRequest():
                self.permissions[item.request_id] = item
            case PermissionDecision():
                self.permissions.pop(item.request_id, None)
            case Started():
                self.started = item
            case _:
                pass

    def info(self) -> SessionInfo:
        return SessionInfo(
            name=self.name,
            kind=self.spec.kind,
            running=self.running,
            permissions=len(self.permissions),
            working=self.working or self.turn is not None,
            attention=self.attention,
        )

    @property
    def uses_channel(self) -> bool:
        return self.spec.kind is AgentKind.CLAUDE_TTY

    def status(self) -> SessionStatus:
        match self.spec:
            case AcpSpec():
                runs, servers = shlex.join(self.spec.command), [s.name for s in self.spec.mcp_servers]
            case ClaudeTtySpec():
                runs = shlex.join([*self.spec.command, *self.spec.args])
                servers = [s.name for s in self.spec.mcp_servers]
            case PydanticAISpec():
                runs, servers = self.spec.agent or self.spec.target or "", []
        if self.spec.aid_tools and not isinstance(self.spec, PydanticAISpec):
            servers.append(BUILTIN_MCP_SERVER)
        return SessionStatus(
            name=self.name,
            kind=self.spec.kind,
            running=self.running,
            busy=self.turn is not None,
            pending=len(self.inbox),
            pid=self.handle.pid if self.handle is not None and self.running else None,
            cwd=self.spec.cwd,
            runs=runs,
            mcp_servers=servers,
            aid_tools=self.spec.aid_tools,
            agent_session=self.started.agent_session if self.started else None,
            agent=self.started.agent if self.started else None,
            model=", ".join(self.models) or (self.started.model if self.started else None),
            permissions=list(self.permissions.values()),
            working=self.working or self.turn is not None,
            attention=self.attention,
        )


def wake_prompt(messages: list[MessageEntry]) -> str:
    """The turn that hands messages to an ACP or pydantic-ai session."""
    parts = [
        f"Message from {f'aid session {m.sender!r}' if m.sender else 'a person using aid'}:\n\n{m.text}"
        for m in messages
    ]
    if any(m.sender for m in messages):
        parts.append(
            "Answer a session with the send_message tool, addressed to its name, when its message asks for "
            "something. Do not answer a message that only acknowledges or thanks."
        )
    return "\n\n---\n\n".join(parts)


class _Listener:
    """A ROUTER socket the daemon answers requests on, and the lock its sends take."""

    def __init__(self, sock: zmq.asyncio.Socket) -> None:
        self.sock = sock
        self._lock = anyio.Lock()

    async def send(self, peer: bytes, payload: bytes) -> None:
        async with self._lock:
            await self.sock.send_multipart([peer, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence


@dataclass(frozen=True)
class _Peer:
    """Who sent a request: a connection on one of the daemon's ROUTER sockets."""

    listener: _Listener
    routing_id: bytes
    plugin: str | None = None
    """The plugin whose key the connection holds; None on the control socket."""

    async def send(self, payload: bytes) -> None:
        await self.listener.send(self.routing_id, payload)


@dataclass(frozen=True)
class _Route:
    client: _Peer | None
    """None for a turn the daemon started itself, which only history sees."""
    session: str
    recorder: Recorder | None = None
    """Set for prompts: writes the turn to the session's history as it passes."""


class Daemon:
    def __init__(self, paths: Paths, launcher: Launcher, *, workers_listen: Sequence[str] = ()) -> None:
        """`workers_listen`: endpoints for workers besides `paths.workers`, such as `ws://` for remote ones."""
        self._paths = paths
        self._launcher = launcher
        self._workers_listen = list(workers_listen)
        self._sessions: dict[str, _Session] = {}
        self._routes: dict[str, _Route] = {}
        self._handling: set[str] = set()
        """Ids of requests still in `_dispatch`, which a route does not name yet."""
        self._ctx = zmq.asyncio.Context()
        self._clients = _Listener(self._ctx.socket(zmq.ROUTER))
        self._workers = self._ctx.socket(zmq.ROUTER)
        self._workers.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._keys = zap.Keys()
        zap.serve_curve(self._workers, self._keys)
        self._zap = self._ctx.socket(zmq.REP)
        self._events = self._ctx.socket(zmq.PUB)
        self._plugins = _Listener(self._ctx.socket(zmq.ROUTER))
        zap.serve_curve(self._plugins.sock, self._keys, zap.PLUGINS_DOMAIN)
        self._plugin_events = self._ctx.socket(zmq.PUB)
        zap.serve_curve(self._plugin_events, self._keys, zap.PLUGIN_EVENTS_DOMAIN)
        self._worker_lock = anyio.Lock()
        self._events_lock = anyio.Lock()
        self._published: dict[bytes, bytes] = {}
        """The last status or list published per topic: publish only what changed."""
        self._tg: TaskGroup | None = None

    async def serve(self, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
        await anyio.Path(self._paths.runtime_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        await self._load_sessions()
        await self._load_plugins()
        await anyio.Path(self._paths.server_key).write_text(self._keys.server.public)
        self._published = self._state()  # As loaded: a reader's first fetch sees this, so it is no change.
        self._events.bind(self._paths.events)
        self._clients.sock.bind(self._paths.control)
        # Before the workers socket: libzmq accepts every CURVE client while no handler is bound.
        self._zap.bind(zap.ZAP_ENDPOINT)
        self._workers.bind(self._paths.workers)
        self._plugins.sock.bind(self._paths.plugins)
        self._plugin_events.bind(self._paths.plugin_events)
        for endpoint in self._workers_listen:
            self._workers.bind(endpoint)
            log.info("workers can connect on %s", endpoint)
        log.info("listening on %s", self._paths.control)
        try:
            async with anyio.create_task_group() as tg:
                self._tg = tg
                tg.start_soon(zap.handle, self._zap, self._keys)
                tg.start_soon(self._client_loop)
                tg.start_soon(self._worker_loop)
                tg.start_soon(self._plugin_loop)
                task_status.started()
                try:
                    await anyio.sleep_forever()
                finally:
                    # Inside the task group: its shielded watchers must see the workers exit before it can close.
                    with anyio.CancelScope(shield=True):
                        await self._stop_all()
        finally:
            self._clients.sock.close(linger=0)
            self._workers.close(linger=0)
            self._zap.close(linger=0)
            self._events.close(linger=0)
            self._plugins.sock.close(linger=0)
            self._plugin_events.close(linger=0)
            self._ctx.term()

    async def _load_sessions(self) -> None:
        sessions_dir = anyio.Path(self._paths.state_dir) / "sessions"
        if not await sessions_dir.exists():
            return
        async for session_dir in sessions_dir.iterdir():
            spec_file = session_dir / SPEC_FILE
            if not await spec_file.exists():
                continue
            await session_dir.chmod(SESSION_DIR_MODE)
            try:
                spec = AgentSpecAdapter.validate_json(await spec_file.read_bytes())
            except ValidationError:
                log.exception("skipping session %s with an invalid spec", session_dir.name)
                continue
            self._sessions[session_dir.name] = self._new_session(session_dir.name, spec)
        log.info("restored %d sessions", len(self._sessions))

    async def _load_plugins(self) -> None:
        plugins_dir = anyio.Path(self._paths.state_dir) / "plugins"
        if not await plugins_dir.exists():
            return
        async for plugin_dir in plugins_dir.iterdir():
            spec_file = plugin_dir / PLUGIN_FILE
            if not await spec_file.exists():
                continue
            try:
                spec = PluginSpec.model_validate_json(await spec_file.read_bytes())
            except ValidationError:
                log.exception("skipping plugin %s with an invalid spec", plugin_dir.name)
                continue
            self._keys.plugins[spec.name] = spec
        log.info("registered %d plugins", len(self._keys.plugins))

    async def _stop_all(self) -> None:
        async with anyio.create_task_group() as tg:
            for session in self._sessions.values():
                tg.start_soon(self._stop, session)

    async def _publish(self, topic: bytes, payload: bytes) -> None:
        async with self._events_lock:
            await self._events.send_multipart([topic, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
            await self._plugin_events.send_multipart([topic, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

    def _state(self) -> dict[bytes, bytes]:
        current = {events.status_topic(s.name): s.status().model_dump_json().encode() for s in self._sessions.values()}
        current[events.SESSIONS] = SessionInfosAdapter.dump_json([s.info() for s in self._sessions.values()])
        return current

    async def _publish_changes(self) -> None:
        """Publish each status and the list if it differs from what was published last."""
        current = self._state()
        for topic, payload in current.items():
            if self._published.get(topic) != payload:
                self._published[topic] = payload
                await self._publish(topic, payload)
        for gone in self._published.keys() - current.keys():
            del self._published[gone]

    async def _publish_entry(self, name: str, entry: HistoryEntry) -> None:
        await self._publish(events.history_topic(name), entry.model_dump_json().encode())

    async def _send_worker(self, session: _Session, request: Request) -> None:
        peer = session.peer
        if peer is None:
            raise zmq.ZMQError(zmq.EHOSTUNREACH, f"no worker connected for {session.name!r}")
        async with self._worker_lock:
            await self._workers.send_multipart([peer, encode(request)])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

    async def _client_loop(self) -> None:
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        while True:
            frames = await self._clients.sock.recv_multipart()
            client = _Peer(self._clients, frames[0])
            if len(frames) != 2:  # A DEALER's request is one frame; anything else would end this loop.
                await client.send(encode(Failure(id="", code="invalid_request", message="a request is one frame")))
                continue
            try:
                request = decode_request(frames[1])
            except ValidationError as error:
                await client.send(encode(Failure(id="", code="invalid_request", message=str(error))))
                continue
            self._tg.start_soon(self._handle, client, request)

    async def _plugin_loop(self) -> None:
        """Requests from plugins: each checked against the grants its plugin holds now."""
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        while True:
            frames = await self._plugins.sock.recv_multipart(copy=False)
            # The plugin the ZAP handler named for this connection's key.
            name = frames[-1].get("User-Id")  # pyright: ignore[reportArgumentType] -- pyzmq's stub knows only the int options; libzmq also takes metadata names
            if not isinstance(name, str):
                raise TypeError(f"User-Id is {name!r}")
            client = _Peer(self._plugins, frames[0].bytes, plugin=name)
            if len(frames) != 2:
                await client.send(encode(Failure(id="", code="invalid_request", message="a request is one frame")))
                continue
            try:
                request = decode_request(frames[1].bytes)
            except ValidationError as error:
                await client.send(encode(Failure(id="", code="invalid_request", message=str(error))))
                continue
            spec = self._keys.plugins.get(name)
            if spec is None or not spec.allows(request.op):
                message = f"plugin {name!r} may not send {request.op!r}"
                await client.send(encode(Failure(id=request.id, code="forbidden", message=message)))
                continue
            if isinstance(request, SendMessage):
                request = request.model_copy(update={"sender": plugins.sender(name)})
            self._tg.start_soon(self._handle, client, request)

    async def _handle(self, client: _Peer, request: Request) -> None:
        try:
            # Replies route by request id, whichever client sent it: a second request under an id in flight would
            # take the first one's replies.
            if request.id in self._routes or request.id in self._handling:
                raise AidError("duplicate_id", f"a request {request.id!r} is in flight")
            self._handling.add(request.id)
            try:
                reply = await self._dispatch(client, request)
            finally:
                self._handling.discard(request.id)
        except AidError as error:
            reply = Failure(id=request.id, code=error.code, message=error.message)
        except Exception as error:
            log.exception("request %s failed", request.id)
            reply = Failure(id=request.id, code="internal", message=f"{type(error).__name__}: {error}")
        if reply is not None:
            await client.send(encode(reply))
        await self._publish_changes()

    async def _dispatch(self, client: _Peer, request: Request) -> Done | None:
        match request:
            case CreateSession():
                await self._create(request)
                return Done(id=request.id, data={"name": request.name})
            case ListSessions():
                infos = [s.info() for s in self._sessions.values()]
                return Done(id=request.id, data=[info.model_dump(mode="json") for info in infos])
            case ListAgents():
                return Done(id=request.id, data=(await list_agents()).model_dump(mode="json"))
            case Prompt():
                session = self._session(request.session)
                if session.turn is not None:
                    raise AidError("busy", f"{session.name!r} is in a turn")
                session.turn = request.id
                try:
                    await self._ensure_running(session)
                    recorder = Recorder(session.history, request.id)
                    await recorder.prompt(request.text)
                    self._routes[request.id] = _Route(client, session.name, recorder)
                    await self._send_worker(session, request)
                except BaseException:
                    self._routes.pop(request.id, None)
                    session.turn = None
                    raise
                return None
            case GetScreen() | GetPane():
                session = self._session(request.session)
                if not session.uses_channel:
                    raise AidError("no_screen", f"{session.name!r} has no terminal")
                if not session.running:
                    raise AidError("not_running", f"{session.name!r} is stopped, so it has no screen")
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case GetStatus():
                return Done(id=request.id, data=self._session(request.session).status().model_dump(mode="json"))
            case SendMessage():
                await self._send_message(request)
                return Done(id=request.id)
            case ReceiveMessages():
                session = self._session(request.session)
                if not session.uses_channel:
                    raise AidError("not_channel", f"{session.name!r} takes its messages as turns, not by channel")
                if not session.inbox:
                    with anyio.move_on_after(request.wait):
                        await session.mail.wait()
                taken, session.inbox = session.inbox, []
                session.mail = anyio.Event()
                return Done(id=request.id, data=[m.model_dump(mode="json") for m in taken])
            case Cancel():
                session = self._session(request.session)
                if not session.running:
                    return Done(id=request.id)
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case Hook():
                session = self._session(request.session)
                if not session.running or session.peer is None:
                    return Done(id=request.id)  # A hook from before Hello, or after the worker went: nobody to tell.
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case AnswerPermission():
                session = self._session(request.session)
                if (pending := session.permissions.get(request.request_id)) is None:
                    raise AidError(
                        "no_pending", f"{session.name!r} waits on no permission request {request.request_id}"
                    )
                if request.option_id is not None and request.option_id not in {o.option_id for o in pending.options}:
                    raise AidError("no_option", f"the request has no option {request.option_id!r}")
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request.model_copy(update={"plugin": client.plugin}))
                return None
            case StartSession():
                await self._ensure_running(self._session(request.session))
                return Done(id=request.id)
            case StopSession():
                await self._stop(self._session(request.session))
                return Done(id=request.id)
            case DeleteSession():
                session = self._session(request.session)
                await self._stop(session)
                del self._sessions[session.name]
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.session_dir(session.name), True)
                return Done(id=request.id)
            case AddPlugin():
                holder = self._keys.plugin(request.spec.public_key)
                if holder is not None and holder.name != request.spec.name:
                    raise AidError("exists", f"plugin {holder.name!r} holds that key")
                plugin_dir = anyio.Path(self._paths.plugin_dir(request.spec.name))
                await plugin_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                await (plugin_dir / PLUGIN_FILE).write_text(request.spec.model_dump_json())
                self._keys.plugins[request.spec.name] = request.spec
                return Done(id=request.id)
            case ListPlugins():
                return Done(id=request.id, data=[s.model_dump(mode="json") for s in self._keys.plugins.values()])
            case RemovePlugin():
                if self._keys.plugins.pop(request.name, None) is None:
                    raise AidError("not_found", f"no plugin named {request.name!r}")
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.plugin_dir(request.name), True)
                return Done(id=request.id)
            case GetSummary():
                entries = await self._session(request.session).history.entries()
                summary = await anyio.to_thread.run_sync(summarize, entries)
                return Done(id=request.id, data=summary.model_dump(mode="json"))
            case GetHistory():
                session = self._session(request.session)
                if request.after is not None and request.wait:
                    with anyio.move_on_after(request.wait):
                        await session.history.wait_after(request.after)
                page = await session.history.page(before=request.before, after=request.after, limit=request.limit)
                return Done(id=request.id, data=page.model_dump(mode="json"))

    def _new_session(self, name: str, spec: AgentSpec) -> _Session:
        history = HistoryLog(
            anyio.Path(self._paths.session_dir(name)) / HISTORY_FILE,
            on_append=lambda entry: self._publish_entry(name, entry),
        )
        return _Session(name, spec, history)

    def _session(self, name: str) -> _Session:
        if (session := self._sessions.get(name)) is None:
            raise AidError("not_found", f"no session named {name!r}")
        return session

    async def _create(self, request: CreateSession) -> None:
        if request.name in self._sessions:
            raise AidError("exists", f"session {request.name!r} already exists")
        session = self._new_session(request.name, request.spec)
        self._sessions[session.name] = session
        session_dir = anyio.Path(self._paths.session_dir(session.name))
        await session_dir.mkdir(mode=SESSION_DIR_MODE, parents=True, exist_ok=True)
        await (session_dir / SPEC_FILE).write_bytes(AgentSpecAdapter.dump_json(request.spec))
        try:
            await self._ensure_running(session)
        except BaseException:
            del self._sessions[session.name]
            await anyio.to_thread.run_sync(shutil.rmtree, session_dir, True)
            raise

    async def _send_message(self, request: SendMessage) -> None:
        session = self._session(request.to)
        if request.sender == session.name:
            raise AidError("to_self", f"{session.name!r} cannot message itself")
        if session.uses_channel and not session.spec.aid_tools:
            raise AidError("cannot_receive", f"{session.name!r} has aid_tools off, so nothing reads its messages")
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        message = MessageEntry(sender=request.sender, text=request.text)
        await Recorder(session.history, request.id).message(message)
        session.inbox.append(message)
        session.mail.set()
        # The sender is often an agent mid-turn, waiting on its tool call: it gets its answer before the wake.
        self._tg.start_soon(self._wake, session)

    async def _wake(self, session: _Session) -> None:
        if not session.uses_channel:
            await self._deliver(session)
            return
        # Interactive Claude reads its inbox through the channel server it starts.
        try:
            await self._ensure_running(session)
        except AidError:
            log.exception("could not start %s for its messages", session.name)
        await self._publish_changes()

    async def _deliver(self, session: _Session) -> None:
        """Hand the inbox to a turn of its own, unless a turn is running: its end calls this again."""
        if session.turn is not None or not session.inbox:
            return
        messages, session.inbox = session.inbox, []
        request = Prompt(session=session.name, text=wake_prompt(messages))
        session.turn = request.id
        recorder = Recorder(session.history, request.id)
        try:
            await self._ensure_running(session)
            self._routes[request.id] = _Route(None, session.name, recorder)
            await self._send_worker(session, request)
        except (AidError, zmq.ZMQError) as error:
            log.exception("could not deliver %d messages to %s", len(messages), session.name)
            self._routes.pop(request.id, None)
            session.turn = None
            await self._record(
                _Route(None, session.name, recorder), Failure(id=request.id, code="undelivered", message=str(error))
            )
        await self._publish_changes()

    def _end_turn(self, name: str, turn: str) -> None:
        session = self._sessions.get(name)
        if session is None or session.turn != turn:
            return
        session.turn = None
        session.permissions.clear()
        if session.inbox and not session.uses_channel and self._tg is not None:
            self._tg.start_soon(self._deliver, session)

    async def _ensure_running(self, session: _Session) -> None:
        async with session.lock:
            if session.running:
                return
            if self._tg is None:
                raise RuntimeError("daemon is not serving")
            session.ready = anyio.Event()
            session.exited = anyio.Event()
            session.start_failed = anyio.Event()
            session.start_error = None
            session.peer = None
            keys = self._keys.issue(session.name)
            session.key = keys.public
            args = WorkerArgs(
                endpoint=self._paths.workers,
                trust_pem="",
                server_key=self._keys.server.public,
                public_key=keys.public,
                secret_key=keys.secret,
                name=session.name,
                spec_json=AgentSpecAdapter.dump_json(session.spec).decode(),
                state_dir=str(self._paths.session_dir(session.name)),
                daemon_runtime_dir=str(self._paths.runtime_dir),
                daemon_state_dir=str(self._paths.state_dir),
            )
            handle = await self._launcher.launch(args)
            session.handle = handle
            self._tg.start_soon(self._watch, session, handle)
            try:
                with anyio.move_on_after(START_TIMEOUT) as scope:
                    await session.ready.wait()
            except anyio.get_cancelled_exc_class():
                # `_create` forgets a session whose start is cancelled, so `_stop_all` can miss it, and the
                # shielded `_watch` would wait on this worker forever.
                handle.kill()
                raise
            if scope.cancelled_caught:
                handle.kill()
                raise AidError("start_timeout", f"worker for {session.name!r} sent no hello in {START_TIMEOUT}s")
            if session.exited.is_set() and not session.start_failed.is_set():
                # The worker's StartFailed and its exit take different paths here, so the exit can win.
                with anyio.move_on_after(START_ERROR_GRACE):
                    await session.start_failed.wait()
            if session.start_error is not None:
                raise AidError("start_failed", f"{session.name!r} did not start: {session.start_error}")
            if session.exited.is_set():
                raise AidError("worker_exited", f"worker for {session.name!r} exited with {session.exit_code}")

    async def _watch(self, session: _Session, handle: WorkerHandle) -> None:
        # Shielded: shutdown cancels this task group, but `_stop_all` still waits on `exited`.
        # `_stop` kills a worker that does not exit, so the wait always ends.
        with anyio.CancelScope(shield=True):
            code = await handle.wait()
            log.info("worker %s (pid %s) exited with %s", session.name, handle.pid, code)
            session.exit_code = code
            if session.handle is handle:
                session.handle = None
                session.peer = None
                if session.key is not None:
                    self._keys.revoke(session.name, session.key)
            for request_id, route in list(self._routes.items()):
                if route.session == session.name:
                    del self._routes[request_id]
                    failure = Failure(id=request_id, code="worker_exited", message=f"worker exited with {code}")
                    if route.client is not None:
                        await route.client.send(encode(failure))
                    await self._record(route, failure)
            # Messages still in the inbox wait for the next turn to end or the next message: a wake now could
            # start a crashing worker again and again.
            session.turn = None
            session.permissions.clear()
            session.working, session.attention = False, None
            with contextlib.suppress(zmq.ZMQError):  # Shutting down: nobody is subscribed any more.
                await self._publish_changes()
            session.exited.set()
            session.ready.set()

    async def _stop(self, session: _Session) -> None:
        handle = session.handle
        if handle is None or session.exited.is_set():
            return
        try:
            await self._send_worker(session, StopSession(session=session.name))
        except zmq.ZMQError:
            log.warning("could not ask worker %s to stop", session.name, exc_info=True)
        with anyio.move_on_after(STOP_TIMEOUT):
            await session.exited.wait()
        if not session.exited.is_set():
            log.warning("killing worker %s after %ss", session.name, STOP_TIMEOUT)
            handle.kill()
            await session.exited.wait()

    async def _worker_loop(self) -> None:
        while True:
            frames = await self._workers.recv_multipart(copy=False)
            if len(frames) != 2:
                log.warning("dropped a message of %d frames from a worker", len(frames))
                continue
            peer, frame = frames
            # The session the ZAP handler named for this connection's key.
            name = frame.get("User-Id")  # pyright: ignore[reportArgumentType] -- pyzmq's stub knows only the int options; libzmq also takes metadata names
            if not isinstance(name, str):
                raise TypeError(f"User-Id is {name!r}")
            payload = frame.bytes
            try:
                reply = decode_reply(payload)
            except ValidationError:
                log.exception("invalid reply from worker %s", name)
                continue
            if (
                isinstance(reply, Event | Done | Failure)
                and (route := self._routes.get(reply.id)) is not None
                and route.session != name
            ):
                log.warning("worker %s replied to %s's request %s", name, route.session, reply.id)
                continue
            match reply:
                case Hello():
                    if (session := self._sessions.get(name)) is not None:
                        log.info("worker %s ready (pid %d)", name, reply.started.pid)
                        session.peer = peer.bytes
                        session.started = reply.started
                        session.ready.set()
                        try:
                            await session.history.append(reply.started, uuid.uuid4().hex)
                        except OSError:
                            log.exception("could not record %s's start", name)
                case StartFailed():
                    if (session := self._sessions.get(name)) is not None:
                        log.warning("worker %s did not start: %s", name, reply.message)
                        session.start_error = reply.message
                        session.start_failed.set()
                        session.ready.set()
                case Observed():
                    if (session := self._sessions.get(name)) is not None:
                        session.note(reply.item)
                        try:
                            await session.history.append(reply.item, reply.turn)
                        except OSError:
                            log.exception("could not record %s's turn %s", name, reply.turn)
                case Activity():
                    if (session := self._sessions.get(name)) is not None:
                        session.working, session.attention = reply.working, reply.attention
                case Event():
                    if (route := self._routes.get(reply.id)) is not None:
                        if route.client is not None:
                            await route.client.send(payload)
                        if (session := self._sessions.get(name)) is not None:
                            session.note(reply.event)
                        await self._record(route, reply)
                case Done() | Failure():
                    if (route := self._routes.pop(reply.id, None)) is not None:
                        if route.client is not None:
                            await route.client.send(payload)
                        await self._record(route, reply)
                        self._end_turn(route.session, reply.id)
            # A turn streams many deltas a second, and they change no status.
            if not (isinstance(reply, Event) and isinstance(reply.event, TextDelta | ThoughtDelta)):
                await self._publish_changes()

    async def _record(self, route: _Route, reply: Event | Done | Failure) -> None:
        """Write a reply to the turn's history, after the client has it. A disk error loses history, not the turn."""
        if route.recorder is None:
            return
        try:
            match reply:
                case Event():
                    await route.recorder.event(reply.event)
                case Done():
                    await route.recorder.flush()
                case Failure():
                    await route.recorder.error(reply.code, reply.message)
        except OSError:
            log.exception("could not write history for %s", route.session)


async def list_agents() -> AgentCatalog:
    """Discover agents in a fresh process: agent modules are user code, and edits since the last call count."""
    with anyio.fail_after(CATALOG_TIMEOUT):
        result = await anyio.run_process([sys.executable, "-m", "aid.catalog"], check=False)
    if result.returncode != 0:
        raise AidError("catalog_failed", result.stderr.decode(errors="replace").strip()[-2000:])
    return AgentCatalog.model_validate_json(result.stdout)


async def run(paths: Paths, launcher: Launcher, *, workers_listen: Sequence[str] = ()) -> None:
    await Daemon(paths, launcher, workers_listen=workers_listen).serve()
