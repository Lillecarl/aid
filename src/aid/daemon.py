"""The daemon: owns sessions, starts workers, and routes requests and replies between clients and workers."""

from __future__ import annotations

import contextlib
import shlex
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import anyio
import anyio.to_thread
import structlog
import zmq
import zmq.asyncio
from pydantic import ValidationError

from aid import events, plugin_host, plugins, zap
from aid.catalog import listing
from aid.client import spawn_plugin_client
from aid.control import AGENT_ID_PREFIX, authorize
from aid.history import HISTORY_FILE, HistoryLog, Recorder
from aid.inbox import INBOX_FILE, Inbox
from aid.launcher import CommandLauncher, WorkerArgs
from aid.messaging import BACKGROUND_SENDER, wake_prompt
from aid.paths import hook_socket
from aid.plugins import PLUGIN_FILE, Grant, PluginSpec
from aid.protocol import (
    Activity,
    AddPlugin,
    AidError,
    AnswerPermission,
    CallPlugin,
    CallResult,
    Cancel,
    CompactSession,
    CreateSession,
    DeleteSession,
    Done,
    Event,
    Failure,
    GetHistory,
    GetLive,
    GetPane,
    GetScreen,
    GetSpec,
    GetStatus,
    GetSummary,
    Hello,
    Hook,
    Invocation,
    ListAgents,
    ListPlugins,
    ListSessions,
    Live,
    LiveChunk,
    MessageEntry,
    MessageHook,
    Notify,
    Observed,
    PermissionDecision,
    PermissionEvent,
    PermissionRequest,
    Prompt,
    PublishSupervision,
    ReceiveMessages,
    RemovePlugin,
    SendMessage,
    ServeCalls,
    SessionInfo,
    SessionInfosAdapter,
    SessionStatus,
    SetAttention,
    Started,
    StartFailed,
    StartSession,
    StopSession,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolEvent,
    TurnEvent,
    Usage,
    decode_reply,
    decode_request,
    encode,
    request_id,
    to_json,
)
from aid.spec import (
    BUILTIN_MCP_SERVER,
    AcpSpec,
    AgentKind,
    AgentSpecAdapter,
    ClaudeSdkSpec,
    ClaudeTtySpec,
    NativeSpec,
)
from aid.summary import summarize

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from anyio.abc import TaskGroup, TaskStatus
    from anyio.streams.memory import MemoryObjectSendStream
    from pydantic import JsonValue

    from aid.client import Client
    from aid.launcher import Launcher, WorkerHandle
    from aid.paths import Paths
    from aid.plugins import Plugin
    from aid.protocol import HistoryEntry, HistoryItem, Request
    from aid.spec import AgentSpec

log = structlog.get_logger(__name__)

START_TIMEOUT: Final = 60.0
START_ERROR_GRACE: Final = 0.5
STOP_TIMEOUT: Final = 10.0
PLUGIN_CALL_TIMEOUT: Final = 30.0
"""Seconds a `CallPlugin` waits for the plugin's answer: a plugin gone mid-call never sends one."""
LIVE_LIMIT: Final = 65536
"""Characters of `usertee` output a session keeps for a page opened mid-command (`GetLive`)."""
SPEC_FILE: Final = "spec.json"
# A spec holds env values and MCP headers, which are often credentials.
SESSION_DIR_MODE: Final = 0o700


@dataclass
class _Session:
    name: str
    spec: AgentSpec
    history: HistoryLog
    inbox: Inbox
    handle: WorkerHandle | None = None
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
    mail: anyio.Event = field(default_factory=anyio.Event)
    """Set when `inbox` gains a message, for a channel server waiting on it."""
    started: Started | None = None
    """What the last worker said about its agent. Kept after it exits: the agent session carries on."""
    models: list[str] = field(default_factory=list[str])
    """What the latest turn's Usage named."""
    cost: dict[str, float] = field(default_factory=dict[str, float])
    """Banked cost per currency: each closed agent session's last reported figure, as SessionSummary.cost."""
    latest: dict[str, float] = field(default_factory=dict[str, float])
    """The running agent session's latest cost per currency: folded into `cost` when the next session starts."""
    permissions: dict[str, PermissionRequest] = field(default_factory=dict[str, PermissionRequest])
    """Requests the agent waits on, by request id."""
    working: bool = False
    """In a turn aid did not start, as the worker's Activity says."""
    live: str = ""
    """The newest `usertee` output, at most LIVE_LIMIT characters, of the turn `live_turn`."""
    live_turn: str | None = None
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
                self.cost.update(self.latest)
                self.latest.clear()
            case _:
                pass
        # Apart from the match: every Usage counts toward cost, like SessionSummary.cost, whether or not it
        # named models.
        if isinstance(item, Usage) and item.session_cost is not None:
            self.latest[item.session_cost.currency] = item.session_cost.amount

    def info(self) -> SessionInfo:
        return SessionInfo(
            name=self.name,
            kind=self.spec.kind,
            running=self.running,
            permissions=len(self.permissions),
            working=self.working or self.turn is not None,
            attention=self.attention,
            parent=self.spec.parent,
        )

    @property
    def busy(self) -> bool:
        return self.working or self.turn is not None

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
            case NativeSpec():
                runs, servers = self.spec.agent or self.spec.target or "", []
            case ClaudeSdkSpec():
                runs, servers = shlex.join([*self.spec.command, "-p", "--model", self.spec.model]), []
        if self.spec.aid_tools and not isinstance(self.spec, NativeSpec):
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
            parent=self.spec.parent,
            cost={**self.cost, **self.latest},
        )


def supervision_cycle(links: dict[str, str | None], name: str, supervisor: str) -> bool:
    """Whether pointing `name` at `supervisor` closes a cycle through the existing links. Sessions are
    created oldest-first, so a cycle is unformable through the API; this guards restored specs."""
    seen = {name}
    current: str | None = supervisor
    while current is not None and current not in seen:
        seen.add(current)
        current = links.get(current)
    return current is not None


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
    agent_key: str | None = None

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
    def __init__(
        self,
        paths: Paths,
        launcher: Launcher,
        *,
        workers_listen: Sequence[str] = (),
        allow_worker_command: bool = False,
        plugin_dirs: Sequence[Path] | None = None,
    ) -> None:
        """`workers_listen`: endpoints for workers besides `paths.workers`, such as `ws://` for remote ones.
        `allow_worker_command`: let sessions name their own worker command. It runs as the daemon with the
        session's worker credentials, so only allow it where session creators are trusted.
        `plugin_dirs`: where in-process plugins load from; None means the standard tiers
        (`aid.plugins.plugins_path`). Tests pass `[]` for no hosted plugins."""
        self._paths = paths
        self._launcher = launcher
        self._workers_listen = list(workers_listen)
        self._allow_worker_command = allow_worker_command
        self._plugin_dirs = list(plugin_dirs) if plugin_dirs is not None else None
        self._hosted: dict[str, anyio.CancelScope] = {}
        """Supervised plugin tasks running now, by name: removing the registration cancels the scope."""
        self._disabled_plugins: set[str] = set()
        self._autostarted: set[str] = set()
        """Plugins registered from their manifest's `autostart`, in memory only: each start registers them anew."""
        self._unwanted: set[str] = set()
        """Autostart plugins removed this daemon's lifetime: they stay off until added again."""
        """Plugins exhausting their restart budget stay disabled for this daemon's lifetime."""
        self._plugin_lock = anyio.Lock()
        self._sessions: dict[str, _Session] = {}
        self._routes: dict[str, _Route] = {}
        self._handling: set[str] = set()
        """Ids of requests still in `_dispatch`, which a route does not name yet."""
        self._ctx = zmq.asyncio.Context()
        self._clients = _Listener(self._ctx.socket(zmq.ROUTER))
        self._workers = self._ctx.socket(zmq.ROUTER)
        self._worker_clients = _Listener(self._workers)
        self._workers.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._keys = zap.Keys()
        zap.serve_curve(self._workers, self._keys)
        self._zap = self._ctx.socket(zmq.REP)
        self._events = self._ctx.socket(zmq.PUB)
        self._plugins = _Listener(self._ctx.socket(zmq.ROUTER))
        zap.serve_curve(self._plugins.sock, self._keys, zap.PLUGINS_DOMAIN)
        self._plugin_events = self._ctx.socket(zmq.PUB)
        zap.serve_curve(self._plugin_events, self._keys, zap.PLUGIN_EVENTS_DOMAIN)
        self._local_plugins = _Listener(self._ctx.socket(zmq.ROUTER))
        self._local_plugin_events = self._ctx.socket(zmq.PUB)
        """The same protocol over inproc, for tasks the daemon spawned: no CURVE, since no other process can
        connect, and the task's DEALER identity names the plugin."""
        self._events_lock = anyio.Lock()
        self._published: dict[bytes, bytes] = {}
        """The last status or list published per topic: publish only what changed."""
        self._tg: TaskGroup | None = None
        self._serving = False
        """True while the task group runs its loops: no wakes or firings during shutdown."""
        self._call_targets: dict[str, _Peer] = {}
        """The connection each plugin serves its calls on, from its latest `ServeCalls`."""
        self._hooks: dict[str, frozenset[str]] = {}
        """What each serving plugin hooks, from its latest `ServeCalls`."""
        self._calls: dict[str, tuple[str, MemoryObjectSendStream[CallResult]]] = {}
        """Calls waiting on a plugin's answer, by invocation id: the plugin and where its answer goes."""

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
        self._local_plugins.sock.bind(plugins.INPROC_PLUGINS)
        self._local_plugin_events.bind(plugins.INPROC_PLUGIN_EVENTS)
        for endpoint in self._workers_listen:
            self._workers.bind(endpoint)
            log.info("listening", socket=endpoint)
        log.info("listening", socket=self._paths.control)
        try:
            async with anyio.create_task_group() as tg:
                self._tg = tg
                tg.start_soon(zap.handle, self._zap, self._keys)
                tg.start_soon(self._client_loop)
                tg.start_soon(self._worker_loop)
                tg.start_soon(self._plugin_loop)
                tg.start_soon(self._plugin_loop, self._local_plugins, True)
                task_status.started()
                self._serving = True
                for session in self._sessions.values():
                    if session.inbox:
                        tg.start_soon(self._wake, session)
                tg.start_soon(self._reconcile_plugins)
                try:
                    await anyio.sleep_forever()
                finally:
                    # Inside the task group: its shielded watchers must see the workers exit before it can close.
                    self._serving = False
                    with anyio.CancelScope(shield=True):
                        await self._stop_all()
        finally:
            self._clients.sock.close(linger=0)
            self._workers.close(linger=0)
            self._zap.close(linger=0)
            self._events.close(linger=0)
            self._plugins.sock.close(linger=0)
            self._plugin_events.close(linger=0)
            self._local_plugins.sock.close(linger=0)
            self._local_plugin_events.close(linger=0)
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
                log.exception("invalid_session_spec", session=session_dir.name)
                continue
            session = self._new_session(session_dir.name, spec)
            session.attention = await session.inbox.restore()
            if session.inbox:
                session.mail.set()
            self._sessions[session_dir.name] = session
        log.info("sessions_restored", count=len(self._sessions))
        for session in self._sessions.values():
            try:
                self._check_supervisor(session.name, session.spec)
            except AidError as error:
                log.warning("invalid_supervision_link", session=session.name, error=error.message)
                session.spec = session.spec.model_copy(update={"supervisor": None})

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
                log.exception("invalid_plugin_spec", plugin=plugin_dir.name)
                continue
            self._keys.plugins[spec.name] = spec
        log.info("plugins_registered", count=len(self._keys.plugins))

    async def _stop_all(self) -> None:
        async with anyio.create_task_group() as tg:
            for session in self._sessions.values():
                tg.start_soon(self._stop, session)

    async def _publish(self, topic: bytes, payload: bytes) -> None:
        async with self._events_lock:
            await self._events.send_multipart([topic, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
            await self._plugin_events.send_multipart([topic, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
            await self._local_plugin_events.send_multipart([topic, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

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
        await self._worker_clients.send(peer, encode(request))

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
                await client.send(encode(Failure(id=request_id(frames[1]), code="invalid_request", message=str(error))))
                continue
            self._tg.start_soon(self._handle, client, request)

    @contextlib.asynccontextmanager
    async def plugin_client(self, name: str) -> AsyncGenerator[Client]:
        """A plugin client for a task spawned as `name`: inproc sockets on this daemon's context, which inproc
        peers must share, with the task's name as its identity."""
        async with spawn_plugin_client(self._ctx, name) as client:
            yield client

    async def _reconcile_plugins(self) -> None:
        """Start supervised tasks for hosted plugins missing one, and stop tasks whose plugin left the registry
        or lost the grants its code declares. Runs at startup and after every registry change."""
        async with self._plugin_lock:
            await self._reconcile_plugins_locked()

    async def _reconcile_plugins_locked(self) -> None:
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        try:
            catalog = await plugin_host.hosted(self._plugin_dirs)
        except (AidError, TimeoutError) as error:
            # Running tasks keep running: a failed listing says nothing about them.
            log.error("plugin_catalog_failed", error=f"{type(error).__name__}: {error}")
            return
        for problem in catalog.problems:
            log.warning("plugin_problem", problem=problem)
        for name, reason in catalog.declined.items():
            log.info("plugin_declined", plugin=name, reason=reason)
        for name, spec in plugin_host.autostart(catalog).items():
            if name not in self._keys.plugins and name not in self._unwanted:
                self._keys.plugins[name] = spec
                self._autostarted.add(name)
        wanted = dict(plugin_host.enabled(catalog, self._keys.plugins))
        for name in self._hosted.keys() - wanted.keys():
            log.info("plugin_stopped", plugin=name)
            self._hosted.pop(name).cancel()
        for name, cls in wanted.items():
            if name not in self._hosted and name not in self._disabled_plugins:
                try:
                    plugin = cls()
                except Exception as error:
                    log.warning("plugin_not_instantiated", plugin=name, error=f"{type(error).__name__}: {error}")
                    continue
                scope = anyio.CancelScope()
                self._hosted[name] = scope
                self._tg.start_soon(self._host_one, name, plugin, scope)

    async def _host_one(self, name: str, plugin: Plugin, scope: anyio.CancelScope) -> None:
        """Supervise one plugin task under `scope`, which removing the registration cancels."""
        log.info("plugin_started", plugin=name)
        with scope:
            try:
                await plugin_host.supervise(self._ctx, name, plugin)
                if self._hosted.get(name) is scope:
                    self._disabled_plugins.add(name)
            finally:
                # Removal/re-registration can install a replacement before this cancelled task unwinds.
                if self._hosted.get(name) is scope:
                    self._hosted.pop(name, None)

    async def _plugin_loop(self, listener: _Listener | None = None, local: bool = False) -> None:
        """Requests from plugins: each checked against the grants its plugin holds now. `local` reads the name
        from the connection's routing id — the DEALER identity the daemon set where it spawned the task, which
        no other process can hold, since inproc peers are process-private. Anything else proves its name with
        its key, through the ZAP handler's User-Id."""
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        listener = self._plugins if listener is None else listener
        while True:
            frames = await listener.sock.recv_multipart(copy=False)
            if local:
                try:
                    name = frames[0].bytes.decode("ascii")
                except UnicodeDecodeError as error:
                    raise TypeError(f"routing id is {frames[0].bytes!r}") from error
            else:
                # The plugin the ZAP handler named for this connection's key.
                name = frames[-1].get("User-Id")  # pyright: ignore[reportArgumentType] -- pyzmq's stub knows only the int options; libzmq also takes metadata names  # pyrefly: ignore[bad-argument-type]
                if not isinstance(name, str):
                    raise TypeError(f"User-Id is {name!r}")
            client = _Peer(listener, frames[0].bytes, plugin=name)
            if len(frames) != 2:
                await client.send(encode(Failure(id="", code="invalid_request", message="a request is one frame")))
                continue
            try:
                request = decode_request(frames[1].bytes)
            except ValidationError as error:
                failure = Failure(id=request_id(frames[1].bytes), code="invalid_request", message=str(error))
                await client.send(encode(failure))
                continue
            spec = self._keys.plugins.get(name)
            if spec is None or not spec.allows(request.op):
                message = f"plugin {name!r} may not send {request.op!r}"
                await client.send(encode(Failure(id=request.id, code="forbidden", message=message)))
                continue
            if isinstance(request, SendMessage) and Grant.SUPERVISE not in spec.grants:
                request = request.model_copy(update={"sender": plugins.sender(name), "usage": None})
            elif isinstance(request, SendMessage) and not request.sender:
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
            log.exception("request_failed", id=request.id)
            reply = Failure(id=request.id, code="internal", message=f"{type(error).__name__}: {error}")
        if reply is not None:
            await client.send(encode(reply))
        await self._publish_changes()

    async def _dispatch(self, client: _Peer, request: Request) -> Done | None:
        if client.agent_key is not None:
            actor = self._keys.agent_session(client.agent_key)
            if actor is None:
                raise AidError("unauthenticated", "agent control credentials were revoked")
            request = authorize(request, actor)
        match request:
            case CreateSession():
                await self._create(request)
                return Done(id=request.id, data={"name": request.name})
            case ListSessions():
                infos = [s.info() for s in self._sessions.values()]
                return Done(id=request.id, data=[info.model_dump(mode="json") for info in infos])
            case ListAgents():
                return Done(id=request.id, data=(await listing()).model_dump(mode="json"))
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
                    await self._publish_turn(session.name, request.id, "started", "prompt")
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
                # Outside a turn the inbox wakes a turn of its own; inside one, the turn may take them between steps.
                if not session.uses_channel and session.turn is None:
                    raise AidError("not_in_turn", f"{session.name!r} takes its messages as turns: none is running")
                with anyio.move_on_after(request.wait):
                    while not session.inbox.ready(interrupts_only=session.busy):
                        mail = session.mail
                        await mail.wait()
                        if session.mail is mail:
                            session.mail = anyio.Event()
                if client.agent_key is not None and self._keys.agent_session(client.agent_key) != session.name:
                    raise AidError("unauthenticated", "agent control credentials were revoked")
                taken = await session.inbox.claim(request.id, interrupts_only=session.busy)
                await session.inbox.finish(request.id)
                session.mail = anyio.Event()
                if session.inbox:
                    session.mail.set()
                return Done(id=request.id, data=[m.model_dump(mode="json") for m in taken])
            case Cancel():
                session = self._session(request.session)
                if not session.running:
                    return Done(id=request.id)
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case CompactSession():
                session = self._session(request.session)
                if session.turn is not None:
                    raise AidError("busy", f"{session.name!r} is in a turn")
                await self._ensure_running(session)
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case Hook():
                session = self._session(request.session)
                if not session.running or session.peer is None:
                    return Done(id=request.id)  # A hook from before Hello, or after the worker went: nobody to tell.
                # hook.py's fallback: its fast path asks the worker's own hook socket first.
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session, request)
                return None
            case AnswerPermission():
                session = self._session(request.session)
                self._checked_pending(session, request.request_id, request.option_id, request.text)
                self._routes[request.id] = _Route(client, session.name)
                supervisor = request.supervisor if self._granted(client, Grant.SUPERVISE) else None
                await self._send_worker(
                    session, request.model_copy(update={"plugin": client.plugin, "supervisor": supervisor})
                )
                return None
            case StartSession():
                session = self._session(request.session)
                await self._ensure_running(session)
                if session.inbox and not session.uses_channel and self._tg is not None:
                    self._tg.start_soon(self._deliver, session)
                return Done(id=request.id)
            case StopSession():
                await self._stop(self._session(request.session))
                return Done(id=request.id)
            case DeleteSession():
                session = self._session(request.session)
                await self._stop(session)
                del self._sessions[session.name]
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.session_dir(session.name), True)
                await anyio.to_thread.run_sync(
                    Path(hook_socket(self._paths.runtime_dir, session.name).removeprefix("ipc://")).unlink, True
                )
                return Done(id=request.id)
            case AddPlugin():
                holder = self._keys.plugin(request.spec.public_key)
                if holder is not None and holder.name != request.spec.name:
                    raise AidError("exists", f"plugin {holder.name!r} holds that key")
                plugin_dir = anyio.Path(self._paths.plugin_dir(request.spec.name))
                await plugin_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                await (plugin_dir / PLUGIN_FILE).write_text(request.spec.model_dump_json())
                self._keys.plugins[request.spec.name] = request.spec
                self._autostarted.discard(request.spec.name)
                self._unwanted.discard(request.spec.name)
                self._call_targets.pop(request.spec.name, None)
                self._hooks.pop(request.spec.name, None)
                await self._reconcile_plugins()
                return Done(id=request.id)
            case ListPlugins():
                return Done(id=request.id, data=[s.model_dump(mode="json") for s in self._keys.plugins.values()])
            case RemovePlugin():
                if self._keys.plugins.pop(request.name, None) is None:
                    raise AidError("not_found", f"no plugin named {request.name!r}")
                if request.name in self._autostarted:
                    self._autostarted.discard(request.name)
                    self._unwanted.add(request.name)
                self._call_targets.pop(request.name, None)
                self._hooks.pop(request.name, None)
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.plugin_dir(request.name), True)
                await self._reconcile_plugins()
                return Done(id=request.id)
            case GetLive():
                session = self._session(request.session)
                chunk = LiveChunk(turn=session.live_turn, text=session.live)
                return Done(id=request.id, data=chunk.model_dump(mode="json"))
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
            case CallPlugin():
                return Done(id=request.id, data=await self._call_plugin(request))
            case ServeCalls():
                if client.plugin is None:
                    raise AidError("forbidden", "only a plugin serves calls")
                if request.hooks and not self._granted(client, Grant.SUPERVISE):
                    raise AidError("forbidden", f"plugin {client.plugin!r} hooks nothing without supervise")
                self._call_targets[client.plugin] = client
                self._hooks[client.plugin] = frozenset(request.hooks)
                return Done(id=request.id)
            case GetSpec():
                return Done(id=request.id, data=to_json(self._session(request.session).spec))
            case SetAttention():
                self._session(request.session).attention = request.text
                return Done(id=request.id)
            case PublishSupervision():
                self._session(request.event.session)
                await self._publish(
                    events.supervision_topic(request.event.session), request.event.model_dump_json().encode()
                )
                return Done(id=request.id)
            case CallResult():
                waiting = self._calls.get(request.call)
                if waiting is None or waiting[0] != client.plugin:
                    raise AidError("unknown_call", f"no call {request.call!r} waits on this plugin")
                del self._calls[request.call]
                waiting[1].send_nowait(request)
                return Done(id=request.id)

    def _granted(self, client: _Peer, grant: Grant) -> bool:
        spec = self._keys.plugins.get(client.plugin) if client.plugin is not None else None
        return spec is not None and grant in spec.grants

    async def _call_plugin(self, request: CallPlugin) -> JsonValue:
        return await self._invoke(
            request.plugin, Invocation(id="", caller=request.caller, method=request.method, args=request.args)
        )

    async def _gate_message(self, request: SendMessage) -> None:
        """Ask every plugin hooking messages before accepting `request`: one refusal refuses it. A plugin that is
        gone or silent lets it through, so a broken plugin never stops messages."""
        hook = MessageHook(to=request.to, sender=request.sender, text=request.text)
        for plugin, hooked in list(self._hooks.items()):
            if "message" not in hooked:
                continue
            invocation = Invocation(
                id="", caller=None, method="message", args=hook.model_dump(mode="json"), hook="message"
            )
            try:
                await self._invoke(plugin, invocation)
            except AidError as error:
                if error.code not in {"no_plugin", "plugin_timeout"}:
                    raise
                log.warning("message_hook_skipped", plugin=plugin, error=error.message)

    async def _invoke(self, plugin: str, invocation: Invocation) -> JsonValue:
        """Send `invocation` to `plugin` under a fresh id and wait for its answer."""
        target = self._call_targets.get(plugin)
        if target is None or plugin not in self._keys.plugins:
            raise AidError("no_plugin", f"no plugin {plugin!r} serves calls")
        call = uuid.uuid4().hex
        send, receive = anyio.create_memory_object_stream[CallResult](1)
        self._calls[call] = (plugin, send)
        try:
            await target.send(encode(invocation.model_copy(update={"id": call})))
            with anyio.fail_after(PLUGIN_CALL_TIMEOUT):
                result = await receive.receive()
        except TimeoutError:
            raise AidError("plugin_timeout", f"plugin {plugin!r} did not answer {invocation.method!r}") from None
        finally:
            self._calls.pop(call, None)
            send.close()
            receive.close()
        if result.code is not None:
            raise AidError(result.code, result.message)
        return result.data

    def _new_session(self, name: str, spec: AgentSpec) -> _Session:
        history = HistoryLog(
            anyio.Path(self._paths.session_dir(name)) / HISTORY_FILE,
            on_append=lambda entry: self._publish_entry(name, entry),
        )
        return _Session(name, spec, history, Inbox(anyio.Path(self._paths.session_dir(name)) / INBOX_FILE, history))

    def _session(self, name: str) -> _Session:
        if (session := self._sessions.get(name)) is None:
            raise AidError("not_found", f"no session named {name!r}")
        return session

    def _check_supervisor(self, name: str, spec: AgentSpec) -> None:
        """The supervision link holds: the supervisor exists, is not this session, and closes no cycle."""
        supervisor = spec.supervisor
        if supervisor is None:
            if spec.supervisor_permissions:
                raise AidError(
                    "supervisor_permissions_need_supervisor",
                    f"session {name!r} allows supervisor permissions but names no supervisor",
                )
            return
        if supervisor == name:
            raise AidError("self_supervised", f"session {name!r} cannot supervise itself")
        if supervisor not in self._sessions:
            raise AidError("unknown_supervisor", f"session {name!r} names no supervisor {supervisor!r}")
        links = {other: session.spec.supervisor for other, session in self._sessions.items()}
        if supervision_cycle(links, name, supervisor):
            raise AidError("supervisor_cycle", f"supervising {name!r} by {supervisor!r} closes a cycle")

    async def _create(self, request: CreateSession) -> None:
        if request.name in self._sessions:
            raise AidError("exists", f"session {request.name!r} already exists")
        self._check_supervisor(request.name, request.spec)
        session = self._new_session(request.name, request.spec)
        self._sessions[session.name] = session
        session_dir = anyio.Path(self._paths.session_dir(session.name))
        await session_dir.mkdir(mode=SESSION_DIR_MODE, parents=True, exist_ok=True)
        await (session_dir / SPEC_FILE).write_bytes(AgentSpecAdapter.dump_json(request.spec))
        try:
            await session.inbox.initialize()
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
        await self._gate_message(request)
        await self._accept_message(
            session,
            MessageEntry(sender=request.sender, text=request.text, delivery=request.delivery),
            request.id,
            request.usage,
        )

    async def _live(self, session: _Session, text: str) -> None:
        """Hold the newest `usertee` output of the running turn and publish it. A new turn starts afresh: the
        output a page opened mid-command needs is this turn's."""
        if session.live_turn != session.turn:
            session.live, session.live_turn = "", session.turn
        session.live = (session.live + text)[-LIVE_LIMIT:]
        chunk = LiveChunk(turn=session.turn, text=text)
        await self._publish(events.live_topic(session.name), chunk.model_dump_json().encode())

    async def _notify(self, session: _Session, reply: Notify) -> None:
        """A worker's watch fired: record its report and wake the session, like a message but from no person
        or session. A worker whose session is gone notifies nothing."""
        await self._accept_message(
            session, MessageEntry(sender=f"{BACKGROUND_SENDER}{reply.task_id}", text=reply.text), uuid.uuid4().hex
        )

    async def _accept_message(
        self, session: _Session, message: MessageEntry, turn: str, usage: Usage | None = None
    ) -> None:
        """Record a message for the session and wake it: a turn of its own when idle, the next one when a
        turn runs. `usage`, the sender's tokens for it, follows the message in the history."""
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        await session.inbox.accept(message, turn)
        if usage is not None:
            await session.history.append(usage, turn)
        session.mail.set()
        # The sender is often an agent mid-turn, waiting on its tool call: it gets its answer before the wake.
        self._tg.start_soon(self._wake, session)

    async def _wake(self, session: _Session) -> None:
        if not self._serving:
            return
        if not session.uses_channel:
            await self._deliver(session)
            return
        # Interactive Claude reads its inbox through the channel server it starts.
        try:
            await self._ensure_running(session)
        except AidError:
            log.exception("worker_start_failed", session=session.name)
        await self._publish_changes()

    async def _deliver(self, session: _Session) -> None:
        """Hand the inbox to a turn of its own, unless a turn is running: its end calls this again."""
        if not self._serving or session.turn is not None or not session.inbox:
            return
        request = Prompt(session=session.name, text="")
        session.turn = request.id
        recorder = Recorder(session.history, request.id)
        claimed = False
        sent = False
        try:
            await self._ensure_running(session)
            messages = await session.inbox.claim(request.id)
            claimed = True
            request = request.model_copy(update={"text": wake_prompt(messages)})
            self._routes[request.id] = _Route(None, session.name, recorder)
            await self._send_worker(session, request)
            sent = True
            await self._publish_turn(session.name, request.id, "started", "wake")
        except (AidError, zmq.ZMQError) as error:
            log.exception("deliver_failed", session=session.name)
            self._routes.pop(request.id, None)
            session.turn = None
            rejected = claimed and not sent and isinstance(error, zmq.ZMQError)
            if rejected:
                await session.inbox.release(request.id)
            await self._record(
                _Route(None, session.name, recorder),
                Failure(
                    id=request.id,
                    code="delivery_uncertain" if claimed and not rejected else "undelivered",
                    message=str(error),
                ),
            )
            await session.inbox.finish(request.id)
            session.attention = "Message delivery failed; inspect history before retrying"
        except BaseException:
            self._routes.pop(request.id, None)
            session.turn = None
            raise
        await self._publish_changes()

    async def _publish_turn(self, name: str, turn: str, phase: Literal["started", "ended"], reason: str) -> None:
        """A turn's boundary on its session's turn topic. Started publishes once the worker holds the prompt —
        a turn that never reaches a worker never starts — and every started has its ended."""
        await self._publish(
            events.turn_topic(name), TurnEvent(turn=turn, phase=phase, reason=reason).model_dump_json().encode()
        )

    async def _end_turn(self, name: str, turn: str, reason: str) -> None:
        session = self._sessions.get(name)
        if session is None or session.turn != turn:
            return
        await session.inbox.finish(turn)
        session.turn = None
        session.permissions.clear()
        await self._publish_turn(name, turn, "ended", reason)
        if session.inbox and not session.uses_channel and self._tg is not None:
            self._tg.start_soon(self._deliver, session)

    def _checked_pending(
        self, session: _Session, request_id: str, option_id: str | None, text: str | None
    ) -> PermissionRequest:
        """The permission request an answer may settle: still waited on, with one of its own options. Text
        with no pick is the whole answer, and only the ask_user tool reads answer text: a permission gate
        answers allow or deny, and anywhere else the text alone would read as a cancel while its words went
        nowhere."""
        if (pending := session.permissions.get(request_id)) is None:
            raise AidError("no_pending", f"{session.name!r} waits on no permission request {request_id}")
        if option_id is not None and option_id not in {o.option_id for o in pending.options}:
            raise AidError("no_option", f"the request has no option {option_id!r}")
        if (
            option_id is None
            and text is not None
            and (not isinstance(session.spec, NativeSpec) or pending.tool_name != "ask_user")
        ):
            raise AidError("no_text", f"{session.name!r} takes an option, not free text; pick one or cancel")
        return pending

    def _launcher_for(self, session: _Session) -> Launcher:
        """Who starts this session's worker: its own command wins, else the daemon's launcher."""
        if (command := session.spec.worker_command) is not None:
            if not self._allow_worker_command:
                raise AidError(
                    "forbidden",
                    "this daemon does not allow per-session worker commands (aid daemon --allow-worker-command)",
                )
            try:
                trust_pem = Path(session.spec.worker_ca).read_text() if session.spec.worker_ca else ""
            except OSError as error:
                raise AidError("worker_ca", f"cannot read {session.spec.worker_ca}: {error}") from error
            return CommandLauncher(command, session.spec.worker_endpoint or self._paths.workers, trust_pem=trust_pem)
        return self._launcher

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
            agent_keys = self._keys.issue_agent(session.name)
            args = WorkerArgs(
                endpoint=self._paths.workers,
                trust_pem="",
                server_key=self._keys.server.public,
                public_key=keys.public,
                secret_key=keys.secret,
                agent_public_key=agent_keys.public,
                agent_secret_key=agent_keys.secret,
                name=session.name,
                spec_json=AgentSpecAdapter.dump_json(session.spec).decode(),
                state_dir=str(self._paths.session_dir(session.name)),
                daemon_runtime_dir=str(self._paths.runtime_dir),
                daemon_state_dir=str(self._paths.state_dir),
            )
            try:
                handle = await self._launcher_for(session).launch(args)
            except BaseException:
                self._keys.revoke(session.name, keys.public)
                self._keys.revoke_agent(session.name, agent_keys.public)
                raise
            session.handle = handle
            self._tg.start_soon(self._watch, session, handle, keys.public, agent_keys.public)
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

    async def _watch(self, session: _Session, handle: WorkerHandle, public_key: str, agent_key: str) -> None:
        # Shielded: shutdown cancels this task group, but `_stop_all` still waits on `exited`.
        # `_stop` kills a worker that does not exit, so the wait always ends.
        with anyio.CancelScope(shield=True):
            code = await handle.wait()
            self._keys.revoke(session.name, public_key)
            self._keys.revoke_agent(session.name, agent_key)
            log.info("worker_exited", session=session.name, pid=handle.pid, code=code)
            session.exit_code = code
            if session.handle is handle:
                session.handle = None
                session.peer = None
            for request_id, route in list(self._routes.items()):
                if route.session == session.name:
                    del self._routes[request_id]
                    failure = Failure(id=request_id, code="worker_exited", message=f"worker exited with {code}")
                    if route.client is not None:
                        await route.client.send(encode(failure))
                    await self._record(route, failure)
            # Messages still in the inbox wait for the next turn to end or the next message: a wake now could
            # start a crashing worker again and again.
            if session.turn is not None:
                with contextlib.suppress(zmq.ZMQError):
                    await self._publish_turn(session.name, session.turn, "ended", "worker_exited")
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
            log.warning("worker_stop_failed", session=session.name, exc_info=True)
        with anyio.move_on_after(STOP_TIMEOUT):
            await session.exited.wait()
        if not session.exited.is_set():
            log.warning("worker_kill", session=session.name, after=STOP_TIMEOUT)
            handle.kill()
            await session.exited.wait()

    async def _worker_loop(self) -> None:
        while True:
            frames = await self._workers.recv_multipart(copy=False)
            if len(frames) != 2:
                log.warning("worker_frames_dropped", frames=len(frames))
                continue
            peer, frame = frames
            # The session the ZAP handler named for this connection's key.
            name = frame.get("User-Id")  # pyright: ignore[reportArgumentType] -- pyzmq's stub knows only the int options; libzmq also takes metadata names  # pyrefly: ignore[bad-argument-type]
            if not isinstance(name, str):
                raise TypeError(f"User-Id is {name!r}")
            payload = frame.bytes
            if name.startswith(AGENT_ID_PREFIX):
                client = _Peer(self._worker_clients, peer.bytes, agent_key=name.removeprefix(AGENT_ID_PREFIX))
                try:
                    request = decode_request(payload)
                except ValidationError:
                    failure = Failure(id=request_id(payload), code="invalid_request", message="invalid agent request")
                    await client.send(encode(failure))
                    continue
                if self._tg is None:
                    raise RuntimeError("daemon is not serving")
                self._tg.start_soon(self._handle, client, request)
                continue
            try:
                reply = decode_reply(payload)
            except ValidationError:
                log.exception("invalid_worker_reply", session=name)
                continue
            if (
                isinstance(reply, Event | Done | Failure)
                and (route := self._routes.get(reply.id)) is not None
                and route.session != name
            ):
                log.warning("worker_reply_mismatch", worker=name, session=route.session, id=reply.id)
                continue
            match reply:
                case Hello():
                    if (session := self._sessions.get(name)) is not None:
                        log.info("worker_ready", session=name, pid=reply.started.pid)
                        session.peer = peer.bytes
                        session.started = reply.started
                        session.ready.set()
                        try:
                            await session.history.append(reply.started, uuid.uuid4().hex)
                        except OSError:
                            log.exception("record_start_failed", session=name)
                case StartFailed():
                    if (session := self._sessions.get(name)) is not None:
                        log.warning("worker_start_failed", session=name, error=reply.message)
                        session.start_error = reply.message
                        session.start_failed.set()
                        session.ready.set()
                case Observed():
                    if (session := self._sessions.get(name)) is not None:
                        session.note(reply.item)
                        try:
                            await session.history.append(reply.item, reply.turn)
                        except OSError:
                            log.exception("record_turn_failed", session=name, turn=reply.turn)
                case Activity():
                    if (session := self._sessions.get(name)) is not None:
                        session.working, session.attention = reply.working, reply.attention
                        if not session.busy and session.inbox:
                            session.mail.set()  # The channel server holds what waits for the turn to end.
                case Notify():
                    if (session := self._sessions.get(name)) is not None:
                        await self._notify(session, reply)
                case Live():
                    if (session := self._sessions.get(name)) is not None:
                        await self._live(session, reply.text)
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
                        reason = "done" if isinstance(reply, Done) else reply.code
                        await self._end_turn(route.session, reply.id, reason)
                case Invocation():
                    log.warning("worker_sent_invocation", worker=name)
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
            log.exception("history_write_failed", session=route.session)
            return
        if isinstance(reply, Event):
            match reply.event:
                case ToolCall() as call:
                    topic, envelope = events.tool_topic(route.session), ToolEvent(turn=reply.id, call=call)
                case PermissionRequest() as request:
                    topic, envelope = (
                        events.permission_topic(route.session),
                        PermissionEvent(turn=reply.id, request=request),
                    )
                case PermissionDecision() as decision:
                    topic, envelope = (
                        events.permission_topic(route.session),
                        PermissionEvent(turn=reply.id, decision=decision),
                    )
                case _:
                    return
            await self._publish(topic, envelope.model_dump_json().encode())


async def run(
    paths: Paths, launcher: Launcher, *, workers_listen: Sequence[str] = (), allow_worker_command: bool = False
) -> None:
    await Daemon(paths, launcher, workers_listen=workers_listen, allow_worker_command=allow_worker_command).serve()
