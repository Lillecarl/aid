"""The daemon: owns sessions, starts workers, and routes requests and replies between clients and workers."""

from __future__ import annotations

import contextlib
import difflib
import shlex
import shutil
import sys
import time
import uuid
from collections import deque
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
from aid.client import spawn_plugin_client
from aid.control import AGENT_ID_PREFIX, authorize
from aid.history import HISTORY_FILE, HistoryLog, Recorder
from aid.inbox import INBOX_FILE, Inbox
from aid.launcher import CommandLauncher, WorkerArgs
from aid.loops import LoopEvidence, ToolLoopDetector
from aid.paths import hook_socket
from aid.plugins import PLUGIN_FILE, PluginSpec
from aid.protocol import (
    Activity,
    AddGoal,
    AddPlugin,
    AgentCatalog,
    AidError,
    AnswerPermission,
    AnswerSupervisedPermission,
    Cancel,
    CancelSupervised,
    CompactSession,
    CreateSession,
    DeleteSession,
    Done,
    Event,
    Failure,
    GetGoals,
    GetHistory,
    GetPane,
    GetScreen,
    GetStatus,
    GetSummary,
    Goal,
    GoalsAdapter,
    GoalState,
    Hello,
    Hook,
    ListAgents,
    ListPlugins,
    ListSessions,
    MessageEntry,
    Notify,
    Observed,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionEvent,
    PermissionRequest,
    Prompt,
    ReceiveMessages,
    RemovePlugin,
    SearchResults,
    SendMessage,
    SessionInfo,
    SessionInfosAdapter,
    SessionStatus,
    SetGoal,
    Started,
    StartFailed,
    StartSession,
    StopSession,
    SupervisedHistory,
    SupervisedSearch,
    SupervisionEvent,
    SupervisionFiring,
    SupervisionGoals,
    SupervisionNudge,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolEvent,
    TurnEvent,
    Usage,
    clip,
    decode_reply,
    decode_request,
    encode,
    history_line,
)
from aid.spec import (
    BUILTIN_MCP_SERVER,
    AcpSpec,
    AgentKind,
    AgentSpecAdapter,
    ClaudeSdkSpec,
    ClaudeTtySpec,
    LiteLMSpec,
)
from aid.summary import summarize

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from anyio.abc import TaskGroup, TaskStatus

    from aid.client import Client
    from aid.launcher import Launcher, WorkerHandle
    from aid.paths import Paths
    from aid.plugins import Plugin
    from aid.protocol import HistoryEntry, HistoryItem, HistoryPage, Request
    from aid.spec import AgentSpec

log = structlog.get_logger(__name__)

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
    stuck_turns: int = 0
    """Consecutive supervised turn ends scoring stuck: the stuck episode's length."""
    nudges: int = 0
    """Supervisor nudges accepted this stuck episode."""
    last_nudge: str | None = None
    """The last accepted nudge's text: a nudge too like it is refused."""
    recent_outputs: deque[Output] | None = None
    """Loaded lazily from history, then maintained from events; independent of the clipped review tail."""
    tool_loops: ToolLoopDetector = field(default_factory=ToolLoopDetector)
    loop_suspected: bool = False
    supervisor_interrupted: bool = False
    working: bool = False
    """In a turn aid did not start, as the worker's Activity says."""
    attention: str | None = None

    @property
    def running(self) -> bool:
        return self.handle is not None and self.ready.is_set() and not self.exited.is_set()

    def note(self, item: HistoryItem) -> None:
        """What a turn's entry changes in the status, whoever started the turn."""
        match item:
            case Output():
                if self.recent_outputs is not None:
                    self.recent_outputs.append(item)
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
            case LiteLMSpec():
                runs, servers = self.spec.agent or self.spec.target or "", []
            case ClaudeSdkSpec():
                runs, servers = shlex.join([*self.spec.command, "-p", "--model", self.spec.model]), []
        if self.spec.aid_tools and not isinstance(self.spec, LiteLMSpec):
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
            cost={**self.cost, **self.latest},
        )


BACKGROUND_SENDER: Final = "background task "
"""Prefix of a MessageEntry sender naming a background task report, `background task bg1`. The worker sends
the task id; the daemon builds the sender. A session named this way would also match, and render as a
report: absurd, and harmless if it ever happens."""


def wake_prompt(messages: list[MessageEntry]) -> str:
    """The turn that hands messages to an ACP or LiteLM session. Background task reports ride as
    written; person and session messages render as quotes, answered with the send_message tool."""
    parts = [
        message.text
        if message.sender is not None and message.sender.startswith(BACKGROUND_SENDER)
        else (
            f"Message from {f'aid session {message.sender!r}' if message.sender else 'a person using aid'}:"
            f"\n\n{message.text}"
        )
        for message in messages
    ]
    if any(m.sender and not m.sender.startswith(BACKGROUND_SENDER) for m in messages):
        parts.append(
            "Answer a session with the send_message tool, addressed to its name, when its message asks for "
            "something. Do not answer a message that only acknowledges or thanks."
        )
    return "\n\n---\n\n".join(parts)


SUPERVISION_ITEMS: Final = 10
"""History entries a supervision firing carries: the supervised session's recent tail, newest last."""
SUPERVISION_ITEM_CHARS: Final = 400
"""Characters of one tail line; longer bodies clip with their excess counted."""
SUPERVISION_SCORE_OUTPUTS: Final = 2
"""Latest output pair the loop score compares: old repetition must not conceal new progress."""
SUPERVISION_LOOP_THRESHOLD: Final = 0.85
"""Loop score at or above this reads as stuck; advisory, the supervisor judges."""


def supervision_cycle(links: dict[str, str | None], name: str, supervisor: str) -> bool:
    """Whether pointing `name` at `supervisor` closes a cycle through the existing links. Sessions are
    created oldest-first, so a cycle is unformable through the API; this guards restored specs."""
    seen = {name}
    current: str | None = supervisor
    while current is not None and current not in seen:
        seen.add(current)
        current = links.get(current)
    return current is not None


def _supervision_line(item: HistoryItem) -> str | None:
    """One clipped line of a supervised session's tail."""
    if (body := history_line(item)) is None:
        return None
    if len(body) > SUPERVISION_ITEM_CHARS:
        body = f"{body[:SUPERVISION_ITEM_CHARS]}\n… {len(body) - SUPERVISION_ITEM_CHARS} more characters"
    return body


def _normalized(body: str) -> str:
    """One line, one spacing, one case: what similarity compares."""
    return " ".join(body.split()).casefold()


def loop_score(items: Sequence[HistoryItem]) -> float | None:
    """Similarity of the latest two answers, never an older repeated pair. Missing/cancelled answers reset
    the signal rather than scoring the string 'None'. Advisory — the supervisor judges."""
    outputs = [item for item in items if isinstance(item, Output)][-SUPERVISION_SCORE_OUTPUTS:]
    if len(outputs) < SUPERVISION_SCORE_OUTPUTS:
        return None
    if any(item.output is None or item.stop_reason != "end_turn" or not str(item.output).strip() for item in outputs):
        return None
    first, second = (_normalized(str(item.output)) for item in outputs)
    return difflib.SequenceMatcher(None, first, second).ratio()


def supervision_text(
    name: str, reason: str, page: HistoryPage, goals: list[Goal], *, outputs: list[Output] | None = None
) -> str:
    """What the supervisor is told when a supervised turn ends: the session, how the turn closed, its open
    goals, its recent history clipped small, and the loop score over recent outputs. Cheap by design: the
    supervisor reads this inline, and reaches for more history only when it needs it. Bounded by
    construction: open goals plus at most `SUPERVISION_ITEMS` lines of `SUPERVISION_ITEM_CHARS` characters."""
    lines = [f"Supervised session {name!r} ended a turn ({reason})."]
    if goals:
        lines.append(f"Open goals ({len(goals)}):")
        lines.extend(f"- {goal.id[:8]}: {goal.text}" for goal in goals)
    else:
        lines.append("No open goals.")
    if page.total > len(page.entries):
        lines.append(f"Showing the newest {len(page.entries)} of {page.total} history entries:")
    rendered = [line for entry in page.entries if (line := _supervision_line(entry.item)) is not None]
    if rendered:
        lines.extend(rendered)
    else:
        lines.append("(Its history holds nothing readable yet.)")
    score = loop_score(outputs if outputs is not None else [entry.item for entry in page.entries])
    if score is None:
        lines.append("Loop score: n/a (fewer than two recent outputs).")
    else:
        lines.append(f"Loop score: {score:.2f} over recent outputs (stuck at {SUPERVISION_LOOP_THRESHOLD:.2f}).")
    return "\n".join(lines)


def permission_backstop_text(name: str, request: PermissionRequest) -> str:
    """What the supervisor is told when a supervised session's permission request outwaits the person: the
    session, the tool, the question, and every option with its id. Small by design: the supervisor answers
    inline with the answer_permission tool, and stays silent to leave the request for the person."""
    lines = [
        f"Supervised session {name!r} waits on permission (request {request.request_id}"
        f", tool {request.tool_name or 'a question'}): {request.title or 'no question given'}."
    ]
    options = [
        f"- {option.option_id}: {option.name}{' (recommended)' if option.recommended else ''}"
        for option in request.options
    ]
    lines.extend(options or ["(It offers no options: answer only with words, or not at all.)"])
    lines.append(
        f"Answer with the answer_permission tool, naming {name!r} and the request id. "
        "A person answering first wins: then the request is gone, and silence is the right answer."
    )
    return "\n".join(lines)


def nudge_repeats(last: str | None, text: str) -> bool:
    """The nudge says what the last one said: the loop-score check turned around, over normalized texts.
    With no last nudge, anything goes."""
    if last is None:
        return False
    return difflib.SequenceMatcher(None, _normalized(last), _normalized(text)).ratio() >= SUPERVISION_LOOP_THRESHOLD


def supervision_stuck(
    name: str, supervisor: str, stuck_turns: int, nudges: int, budget: int, score: float | None
) -> tuple[int, int, str | None]:
    """The stuck episode after a supervised turn end scoring `score`: the next stuck count, the nudges kept,
    and attention for a person. Evidence of progress ends the episode; missing evidence does not renew budgets.
    At `budget` stuck turns after `budget` nudges the session needs a person instead of another nudge."""
    if score is not None and score < SUPERVISION_LOOP_THRESHOLD:
        return 0, 0, None
    stuck = stuck_turns + int(score is not None)
    if stuck > 0 and stuck >= budget and nudges >= budget:
        return (
            stuck,
            nudges,
            f"{name} is stuck for {stuck} turns after {nudges} nudges from {supervisor}; needs a person",
        )
    return stuck, nudges, None


GOALS_FILE: Final = "goals.json"
"""A supervised session's goals, beside its `history.jsonl`: the supervisor names them, the daemon keeps them."""

SEARCH_PAGE: Final = 1000
"""History entries one search step scans: page-sized reads, newest first."""


def add_goal(goals: list[Goal], text: str) -> Goal:
    """Name a goal. Pure: the caller saves."""
    goal = Goal(id=uuid.uuid4().hex, text=text, updated_at=time.time())
    goals.append(goal)
    return goal


def set_goal(goals: list[Goal], goal_id: str, state: GoalState) -> Goal:
    """Mark a goal done or dropped. Pure: the caller saves."""
    for i, goal in enumerate(goals):
        if goal.id == goal_id:
            updated = goal.model_copy(update={"state": state, "updated_at": time.time()})
            goals[i] = updated
            return updated
    raise AidError("unknown_goal", f"no goal {goal_id!r}")


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
        self._backstops: dict[str, tuple[str, anyio.CancelScope]] = {}
        """Permission backstops still waiting, by request id: the supervised session and their cancel scope."""

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
                await client.send(encode(Failure(id="", code="invalid_request", message=str(error))))
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
        catalog = plugin_host.discover_hosted(self._plugin_dirs)
        for problem in catalog.problems:
            log.warning("plugin_problem", problem=problem)
        for name, reason in catalog.declined.items():
            log.info("plugin_declined", plugin=name, reason=reason)
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
                return Done(id=request.id, data=(await list_agents()).model_dump(mode="json"))
            case Prompt():
                session = self._session(request.session)
                if session.turn is not None:
                    raise AidError("busy", f"{session.name!r} is in a turn")
                session.turn = request.id
                session.tool_loops = ToolLoopDetector()
                session.loop_suspected = session.supervisor_interrupted = False
                # An explicit prompt is a new instruction, not another automatic recovery attempt.
                session.stuck_turns = session.nudges = 0
                session.last_nudge = None
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
                if not session.uses_channel:
                    raise AidError("not_channel", f"{session.name!r} takes its messages as turns, not by channel")
                if not session.inbox:
                    with anyio.move_on_after(request.wait):
                        await session.mail.wait()
                if client.agent_key is not None and self._keys.agent_session(client.agent_key) != session.name:
                    raise AidError("unauthenticated", "agent control credentials were revoked")
                taken = await session.inbox.claim(request.id)
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
            case CancelSupervised():
                session = self._supervised_session(request.supervisor, request.session)
                if session.turn != request.turn or not session.running:
                    raise AidError("stale_turn", "the reviewed turn is no longer running")
                self._routes[request.id] = _Route(client, session.name)
                session.supervisor_interrupted = True
                await self._send_worker(session, Cancel(id=request.id, session=session.name, turn=request.turn))
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
                pending = self._checked_pending(session, request.request_id, request.option_id, request.text)
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(
                    session, request.model_copy(update={"plugin": client.plugin, "supervisor": None})
                )
                return None
            case AnswerSupervisedPermission():
                session = self._supervised_session(request.supervisor, request.session)
                pending = self._checked_pending(session, request.request_id, request.option_id, request.text)
                if pending.tool_name not in session.spec.supervisor_permissions:
                    raise AidError(
                        "unlisted_tool",
                        f"{request.supervisor!r} may not answer {pending.tool_name!r} for {session.name!r}",
                    )
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(
                    session,
                    AnswerPermission(
                        # The worker's Done carries this id back: it must be the caller's, like the
                        # person path's model_copy, so the reply routes to who asked.
                        id=request.id,
                        session=session.name,
                        request_id=request.request_id,
                        option_id=request.option_id,
                        text=request.text,
                        supervisor=request.supervisor,
                    ),
                )
                return None
            case SupervisedHistory():
                session = self._supervised_session(request.supervisor, request.session)
                page = await session.history.page(limit=request.limit)
                return Done(id=request.id, data=page.model_dump(mode="json"))
            case SupervisedSearch():
                if not request.query.strip():
                    raise AidError("empty_query", "a supervision search needs a query")
                session = self._supervised_session(request.supervisor, request.session)
                results = await self._search_history(session, request.query, request.limit)
                return Done(id=request.id, data=results.model_dump(mode="json"))
            case GetGoals():
                session = self._supervised_session(request.supervisor, request.session)
                goals = await self._read_goals(session.name)
                return Done(id=request.id, data=[goal.model_dump(mode="json") for goal in goals])
            case AddGoal():
                if not request.text.strip():
                    raise AidError("empty_goal", "a goal needs text")
                session = self._supervised_session(request.supervisor, request.session)
                goals = await self._read_goals(session.name)
                goal = add_goal(goals, request.text)
                await self._write_goals(session.name, goals)
                await self._publish(
                    events.supervision_topic(session.name),
                    SupervisionEvent(
                        session=session.name,
                        supervisor=request.supervisor,
                        goals=SupervisionGoals(action="added", goal=goal),
                    )
                    .model_dump_json()
                    .encode(),
                )
                return Done(id=request.id, data=goal.model_dump(mode="json"))
            case SetGoal():
                session = self._supervised_session(request.supervisor, request.session)
                goals = await self._read_goals(session.name)
                goal = set_goal(goals, request.goal_id, request.state)
                await self._write_goals(session.name, goals)
                await self._publish(
                    events.supervision_topic(session.name),
                    SupervisionEvent(
                        session=session.name,
                        supervisor=request.supervisor,
                        goals=SupervisionGoals(action="done" if goal.state is GoalState.DONE else "dropped", goal=goal),
                    )
                    .model_dump_json()
                    .encode(),
                )
                return Done(id=request.id, data=goal.model_dump(mode="json"))
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
                self._disarm_session_backstops(session.name)
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
                await self._reconcile_plugins()
                return Done(id=request.id)
            case ListPlugins():
                return Done(id=request.id, data=[s.model_dump(mode="json") for s in self._keys.plugins.values()])
            case RemovePlugin():
                if self._keys.plugins.pop(request.name, None) is None:
                    raise AidError("not_found", f"no plugin named {request.name!r}")
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.plugin_dir(request.name), True)
                await self._reconcile_plugins()
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
        nudge = request.sender is not None and session.spec.supervisor == request.sender
        if nudge:
            self._check_nudge(session, request.text)
        await self._accept_message(session, MessageEntry(sender=request.sender, text=request.text), request.id)
        if nudge and request.sender is not None:
            session.nudges += 1
            session.last_nudge = request.text
            await self._publish(
                events.supervision_topic(session.name),
                SupervisionEvent(
                    session=session.name,
                    supervisor=request.sender,
                    nudge=SupervisionNudge(text=clip(request.text), nudges=session.nudges),
                )
                .model_dump_json()
                .encode(),
            )

    def _check_nudge(self, session: _Session, text: str) -> None:
        """A supervisor's nudge to its supervised session goes out: within budget, and nothing like the last."""
        budget = session.spec.supervisor_nudge_budget
        if session.nudges >= budget:
            raise AidError(
                "nudge_budget",
                f"{session.name!r} had its {budget} nudges this episode; further stuck, it needs a person",
            )
        if nudge_repeats(session.last_nudge, text):
            raise AidError(
                "repeated_nudge", "that nudge says what the last one said; say it differently or stay silent"
            )

    async def _notify(self, session: _Session, reply: Notify) -> None:
        """A worker's watch fired: record its report and wake the session, like a message but from no person
        or session. A worker whose session is gone notifies nothing."""
        await self._accept_message(
            session, MessageEntry(sender=f"{BACKGROUND_SENDER}{reply.task_id}", text=reply.text), uuid.uuid4().hex
        )

    async def _accept_message(self, session: _Session, message: MessageEntry, turn: str) -> None:
        """Record a message for the session and wake it: a turn of its own when idle, the next one when a
        turn runs."""
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        await session.inbox.accept(message, turn)
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
        session.tool_loops = ToolLoopDetector()
        session.loop_suspected = session.supervisor_interrupted = False
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
        self._disarm_session_backstops(name)
        await self._publish_turn(name, turn, "ended", reason)
        await self._fire_supervisor(name, reason, turn)
        if session.inbox and not session.uses_channel and self._tg is not None:
            self._tg.start_soon(self._deliver, session)

    def _supervised_session(self, supervisor: str, name: str) -> _Session:
        """The session `supervisor` watches. Supervisors read and write only their own: anyone else gets an
        error, not another session's history."""
        session = self._session(name)
        if session.spec.supervisor != supervisor:
            raise AidError("not_supervised", f"{supervisor!r} supervises no session {name!r}")
        return session

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
            and (not isinstance(session.spec, LiteLMSpec) or pending.tool_name != "ask_user")
        ):
            raise AidError("no_text", f"{session.name!r} takes an option, not free text; pick one or cancel")
        return pending

    async def _read_goals(self, name: str) -> list[Goal]:
        """The session's goals, or none: a missing file means none yet; an unreadable one is logged and
        skipped, so a corrupt file never breaks supervision."""
        path = anyio.Path(self._paths.session_dir(name)) / GOALS_FILE
        try:
            data = await path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError:
            log.exception("goals_unreadable", session=name)
            return []
        try:
            return GoalsAdapter.validate_json(data)
        except ValidationError:
            log.exception("goals_unreadable", session=name)
            return []

    async def _write_goals(self, name: str, goals: list[Goal]) -> None:
        path = anyio.Path(self._paths.session_dir(name)) / GOALS_FILE
        tmp = path.with_name(path.name + ".tmp")
        await tmp.write_bytes(GoalsAdapter.dump_json(goals))
        await tmp.rename(path)

    async def _search_history(self, session: _Session, query: str, limit: int) -> SearchResults:
        """Entries whose rendered line holds `query`, newest first. Page-sized reads newest-first across the
        whole history: the scan costs one pass, and only the newest `limit` matches travel."""
        needle = query.casefold()
        matches: list[HistoryEntry] = []
        before: int | None = None
        while True:
            page = await session.history.page(before=before, limit=SEARCH_PAGE)
            for entry in reversed(page.entries):
                if (line := history_line(entry.item)) is not None and needle in line.casefold():
                    matches.append(entry)
            if not page.has_older:
                break
            before = page.entries[0].seq
        return SearchResults(entries=matches[:limit], total=len(matches))

    async def _fire_supervisor(self, name: str, reason: str, turn: str) -> None:
        """A supervised turn ended: score its tail for stuckness, then hand it to the supervisor as a message,
        through the same accept→inbox→wake path a message takes — a wake turn when idle, the next one when
        a turn runs. A supervisor that is gone hears nothing, and nothing replays. The firing is also
        published on the supervised session's supervision topic, with its reason and score."""
        if not self._serving:
            return
        session = self._sessions.get(name)
        supervisor = session.spec.supervisor if session is not None else None
        if session is None or supervisor is None:
            return
        page = await session.history.page(limit=SUPERVISION_ITEMS)
        outputs = await self._recent_outputs(session)
        interrupted_loop = (
            session.loop_suspected
            and session.supervisor_interrupted
            and bool(outputs)
            and outputs[-1].stop_reason == "cancelled"
        )
        score = 1.0 if interrupted_loop else loop_score(outputs)
        stuck, nudges, attention = supervision_stuck(
            name, supervisor, session.stuck_turns, session.nudges, session.spec.supervisor_nudge_budget, score
        )
        session.stuck_turns, session.nudges, session.attention = stuck, nudges, attention
        if stuck == 0 and nudges == 0:
            session.last_nudge = None
        overseer = self._sessions.get(supervisor)
        if overseer is None:
            log.info("supervisor_gone", session=name, supervisor=supervisor)
            return
        goals = [goal for goal in await self._read_goals(name) if goal.state is GoalState.OPEN]
        text = supervision_text(name, reason, page, goals, outputs=outputs)
        if interrupted_loop:
            text += (
                "\nThe supervisor interrupted a suspected live tool loop. This remains the same stuck episode; "
                "interruption alone is not evidence of progress and does not renew the nudge budget."
            )
        await self._accept_message(
            overseer,
            MessageEntry(sender=name, text=text),
            uuid.uuid4().hex,
        )
        await self._publish(
            events.supervision_topic(name),
            SupervisionEvent(
                session=name,
                supervisor=supervisor,
                firing=SupervisionFiring(turn=turn, reason=reason, score=score, stuck_turns=stuck),
            )
            .model_dump_json()
            .encode(),
        )

    async def _recent_outputs(self, session: _Session) -> list[Output]:
        """Restore the small scoring window once. Tool-heavy turns may push the previous answer thousands
        of entries behind the review tail; subsequent turns update this cache through _Session.note."""
        if session.recent_outputs is None:
            newest: list[Output] = []
            before: int | None = None
            while len(newest) < SUPERVISION_SCORE_OUTPUTS:
                page = await session.history.page(before=before, limit=SEARCH_PAGE)
                newest.extend(entry.item for entry in reversed(page.entries) if isinstance(entry.item, Output))
                if not page.has_older or not page.entries:
                    break
                before = page.entries[0].seq
            session.recent_outputs = deque(
                reversed(newest[:SUPERVISION_SCORE_OUTPUTS]), maxlen=SUPERVISION_SCORE_OUTPUTS
            )
        return list(session.recent_outputs)

    async def _review_tool_loop(self, session: _Session, turn: str, call: ToolCall) -> None:
        """Live tool telemetry goes straight to the supervisor's reliable inbox, not through PUB. Detection
        never interrupts the agent; only the supervisor can decide whether this repetition is useful."""
        supervisor = session.spec.supervisor
        if supervisor is None or session.turn != turn or not self._serving:
            return
        evidence = session.tool_loops.observe(call)
        overseer = self._sessions.get(supervisor)
        if evidence is None or overseer is None:
            return
        session.loop_suspected = True
        await self._send_loop_review(session, overseer, turn, evidence)

    async def _send_loop_review(self, session: _Session, overseer: _Session, turn: str, evidence: LoopEvidence) -> None:
        text = (
            f"Suspected tool loop in supervised session {session.name!r}, live turn {turn!r}.\n"
            f"{evidence.describe()}\n"
            "Read supervised_history/supervised_search to judge whether work is progressing. "
            "If it is useful, leave it running. If it is stuck, interrupt(session, turn) stops only this turn; "
            "send_message can then give corrective instructions. A message alone waits until the turn ends."
        )
        await self._accept_message(overseer, MessageEntry(sender=session.name, text=text), uuid.uuid4().hex)
        await self._publish(
            events.supervision_topic(session.name),
            SupervisionEvent(
                session=session.name,
                supervisor=overseer.name,
                firing=SupervisionFiring(
                    turn=turn, reason="suspected_tool_loop", score=None, stuck_turns=session.stuck_turns
                ),
            )
            .model_dump_json()
            .encode(),
        )

    async def _person_recent(self, session: _Session, window: float) -> bool:
        """A person-authored entry in the last `window` seconds: their own message (sender None), or their
        answer to a permission request. Agent and daemon senders name themselves, so None is a person."""
        cutoff = time.time() - window
        before: int | None = None
        while True:
            page = await session.history.page(before=before, limit=SEARCH_PAGE)
            for entry in reversed(page.entries):
                if entry.at < cutoff:
                    return False
                match entry.item:
                    case MessageEntry(sender=None):
                        return True
                    case PermissionDecision(by=PermissionDecider.PERSON):
                        return True
                    case _:
                        pass
            if not page.has_older:
                return False
            before = page.entries[0].seq

    def _backstop_bound(self, session: _Session, request: PermissionRequest) -> float | None:
        """Seconds the worker waits on this request itself: the backstop arms only before it. An `ask_user`
        question with a recommended option ends at the autoselect instead, which keeps working beside the
        backstop. None when the worker's bound is aid-unknown (interactive Claude)."""
        match session.spec:
            case AcpSpec(permission_timeout=timeout):
                return timeout
            case LiteLMSpec(permission_timeout=timeout, ask_autoselect_after=autoselect):
                if request.tool_name == "ask_user" and any(o.recommended for o in request.options):
                    return min(timeout, autoselect)
                return timeout
            case _:
                return None

    async def _arm_backstop(self, name: str, request: PermissionRequest) -> None:
        """Wake the session's supervisor over an unanswered permission request, once the person had their
        window: the full timeout with a person around, the shortened one without. Skipped when the tool is
        not allow-listed, and when the worker's own timeout answers first."""
        if self._tg is None:
            return
        session = self._sessions.get(name)
        if session is None:
            return
        spec = session.spec
        if spec.supervisor is None or request.tool_name not in spec.supervisor_permissions:
            return
        self._disarm_backstop(request.request_id)
        wait = spec.supervisor_permission_timeout
        # Presence doubles as the window: a person-authored entry this recent keeps the full wait.
        if not await self._person_recent(session, wait):
            wait = spec.supervisor_permission_absent_timeout
        if (bound := self._backstop_bound(session, request)) is not None and wait >= bound:
            log.info("backstop_moot", session=name, request=request.request_id, wait=wait, bound=bound)
            return
        scope = anyio.CancelScope()
        self._backstops[request.request_id] = (name, scope)
        self._tg.start_soon(self._await_backstop, name, request.request_id, wait, scope)

    def _disarm_backstop(self, request_id: str) -> None:
        """The request settled: its backstop, if any, stands down."""
        if (armed := self._backstops.pop(request_id, None)) is not None:
            armed[1].cancel()

    def _disarm_session_backstops(self, name: str) -> None:
        """The session's turn ended, its worker went, or it was deleted: no request of it still waits."""
        for request_id, (session, scope) in list(self._backstops.items()):
            if session == name:
                del self._backstops[request_id]
                scope.cancel()

    async def _await_backstop(self, name: str, request_id: str, wait: float, scope: anyio.CancelScope) -> None:
        """Outwait the person, then hand the still-waiting request to the supervisor as a message, through
        the same accept→inbox→wake path a firing takes. A decision first cancels the scope, so the wake
        only ever carries a live request."""
        with scope:
            await anyio.sleep(wait)
        armed = self._backstops.get(request_id)
        if armed is None or armed[1] is not scope:
            return
        self._backstops.pop(request_id, None)
        session = self._sessions.get(name)
        supervisor = session.spec.supervisor if session is not None else None
        overseer = self._sessions.get(supervisor) if supervisor is not None else None
        if session is None or overseer is None:
            if supervisor is not None:
                log.info("supervisor_gone", session=name, supervisor=supervisor)
            return
        if (pending := session.permissions.get(request_id)) is None:
            return
        await self._accept_message(
            overseer, MessageEntry(sender=name, text=permission_backstop_text(name, pending)), uuid.uuid4().hex
        )

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
                await self._fire_supervisor(session.name, "worker_exited", session.turn)
                session.turn = None
            session.permissions.clear()
            self._disarm_session_backstops(session.name)
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
                    await client.send(encode(Failure(id="", code="invalid_request", message="invalid agent request")))
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
                case Notify():
                    if (session := self._sessions.get(name)) is not None:
                        await self._notify(session, reply)
                case Event():
                    if (route := self._routes.get(reply.id)) is not None:
                        if route.client is not None:
                            await route.client.send(payload)
                        if (session := self._sessions.get(name)) is not None:
                            session.note(reply.event)
                        await self._record(route, reply)
                        if isinstance(reply.event, ToolCall) and (session := self._sessions.get(name)) is not None:
                            await self._review_tool_loop(session, reply.id, reply.event)
                case Done() | Failure():
                    if (route := self._routes.pop(reply.id, None)) is not None:
                        if route.client is not None:
                            await route.client.send(payload)
                        await self._record(route, reply)
                        reason = "done" if isinstance(reply, Done) else reply.code
                        await self._end_turn(route.session, reply.id, reason)
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
            if isinstance(reply.event, PermissionRequest):
                await self._arm_backstop(route.session, reply.event)
            elif isinstance(reply.event, PermissionDecision):
                self._disarm_backstop(reply.event.request_id)


async def list_agents() -> AgentCatalog:
    """Discover agents in a fresh process: agent modules are user code, and edits since the last call count."""
    with anyio.fail_after(CATALOG_TIMEOUT):
        result = await anyio.run_process([sys.executable, "-m", "aid.catalog"], check=False)
    if result.returncode != 0:
        raise AidError("catalog_failed", result.stderr.decode(errors="replace").strip()[-2000:])
    return AgentCatalog.model_validate_json(result.stdout)


async def run(
    paths: Paths, launcher: Launcher, *, workers_listen: Sequence[str] = (), allow_worker_command: bool = False
) -> None:
    await Daemon(paths, launcher, workers_listen=workers_listen, allow_worker_command=allow_worker_command).serve()
