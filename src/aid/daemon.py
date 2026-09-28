"""The daemon: owns sessions, starts workers, and routes requests and replies between clients and workers."""

from __future__ import annotations

import logging
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import zmq
import zmq.asyncio
from pydantic import ValidationError

from aid.history import HISTORY_FILE, HistoryLog, Recorder
from aid.launcher import WorkerArgs
from aid.protocol import (
    AgentCatalog,
    AidError,
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
    Hello,
    ListAgents,
    ListSessions,
    MessageEntry,
    Prompt,
    ReceiveMessages,
    SendMessage,
    SessionInfo,
    SessionStatus,
    StartFailed,
    StopSession,
    decode_reply,
    decode_request,
    encode,
)
from aid.spec import BUILTIN_MCP_SERVER, AcpSpec, AgentKind, AgentSpecAdapter, ClaudeTtySpec, PydanticAISpec

if TYPE_CHECKING:
    from anyio.abc import TaskGroup, TaskStatus

    from aid.launcher import Launcher, WorkerHandle
    from aid.paths import Paths
    from aid.protocol import Request
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

    @property
    def running(self) -> bool:
        return self.handle is not None and self.ready.is_set() and not self.exited.is_set()

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


@dataclass(frozen=True)
class _Route:
    client: bytes | None
    """None for a turn the daemon started itself, which only history sees."""
    session: str
    recorder: Recorder | None = None
    """Set for prompts: writes the turn to the session's history as it passes."""


class Daemon:
    def __init__(self, paths: Paths, launcher: Launcher) -> None:
        self._paths = paths
        self._launcher = launcher
        self._sessions: dict[str, _Session] = {}
        self._routes: dict[str, _Route] = {}
        self._ctx = zmq.asyncio.Context()
        self._clients = self._ctx.socket(zmq.ROUTER)
        self._workers = self._ctx.socket(zmq.ROUTER)
        self._workers.setsockopt(zmq.ROUTER_HANDOVER, 1)
        self._workers.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._client_lock = anyio.Lock()
        self._worker_lock = anyio.Lock()
        self._tg: TaskGroup | None = None

    async def serve(self, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
        await anyio.Path(self._paths.runtime_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        await self._load_sessions()
        self._clients.bind(self._paths.control)
        self._workers.bind(self._paths.workers)
        log.info("listening on %s", self._paths.control)
        try:
            async with anyio.create_task_group() as tg:
                self._tg = tg
                tg.start_soon(self._client_loop)
                tg.start_soon(self._worker_loop)
                task_status.started()
                try:
                    await anyio.sleep_forever()
                finally:
                    # Inside the task group: its shielded watchers must see the workers exit before it can close.
                    with anyio.CancelScope(shield=True):
                        await self._stop_all()
        finally:
            self._clients.close(linger=0)
            self._workers.close(linger=0)
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

    async def _stop_all(self) -> None:
        async with anyio.create_task_group() as tg:
            for session in self._sessions.values():
                tg.start_soon(self._stop, session)

    async def _send_client(self, client: bytes, payload: bytes) -> None:
        async with self._client_lock:
            await self._clients.send_multipart([client, payload])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

    async def _send_worker(self, name: str, request: Request) -> None:
        async with self._worker_lock:
            await self._workers.send_multipart([name.encode(), encode(request)])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

    async def _client_loop(self) -> None:
        if self._tg is None:
            raise RuntimeError("daemon is not serving")
        while True:
            client, payload = await self._clients.recv_multipart()
            try:
                request = decode_request(payload)
            except ValidationError as error:
                await self._send_client(client, encode(Failure(id="", code="invalid_request", message=str(error))))
                continue
            self._tg.start_soon(self._handle, client, request)

    async def _handle(self, client: bytes, request: Request) -> None:
        try:
            reply = await self._dispatch(client, request)
        except AidError as error:
            reply = Failure(id=request.id, code=error.code, message=error.message)
        except Exception as error:
            log.exception("request %s failed", request.id)
            reply = Failure(id=request.id, code="internal", message=f"{type(error).__name__}: {error}")
        if reply is not None:
            await self._send_client(client, encode(reply))

    async def _dispatch(self, client: bytes, request: Request) -> Done | None:
        match request:
            case CreateSession():
                await self._create(request)
                return Done(id=request.id, data={"name": request.name})
            case ListSessions():
                infos = [SessionInfo(name=s.name, kind=s.spec.kind, running=s.running) for s in self._sessions.values()]
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
                    await self._send_worker(session.name, request)
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
                await self._send_worker(session.name, request)
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
                await self._send_worker(session.name, request)
                return None
            case StopSession():
                await self._stop(self._session(request.session))
                return Done(id=request.id)
            case DeleteSession():
                session = self._session(request.session)
                await self._stop(session)
                del self._sessions[session.name]
                await anyio.to_thread.run_sync(shutil.rmtree, self._paths.session_dir(session.name), True)
                return Done(id=request.id)
            case GetHistory():
                session = self._session(request.session)
                page = await session.history.page(before=request.before, after=request.after, limit=request.limit)
                return Done(id=request.id, data=page.model_dump(mode="json"))

    def _new_session(self, name: str, spec: AgentSpec) -> _Session:
        history = HistoryLog(anyio.Path(self._paths.session_dir(name)) / HISTORY_FILE)
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
            await self._send_worker(session.name, request)
        except (AidError, zmq.ZMQError) as error:
            log.exception("could not deliver %d messages to %s", len(messages), session.name)
            self._routes.pop(request.id, None)
            session.turn = None
            await self._record(
                _Route(None, session.name, recorder), Failure(id=request.id, code="undelivered", message=str(error))
            )

    def _end_turn(self, name: str, turn: str) -> None:
        session = self._sessions.get(name)
        if session is None or session.turn != turn:
            return
        session.turn = None
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
            args = WorkerArgs(
                endpoint=self._paths.workers,
                name=session.name,
                spec_json=AgentSpecAdapter.dump_json(session.spec).decode(),
                state_dir=str(self._paths.session_dir(session.name)),
                daemon_runtime_dir=str(self._paths.runtime_dir),
                daemon_state_dir=str(self._paths.state_dir),
            )
            handle = await self._launcher.launch(args)
            session.handle = handle
            self._tg.start_soon(self._watch, session, handle)
            with anyio.move_on_after(START_TIMEOUT) as scope:
                await session.ready.wait()
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
            for request_id, route in list(self._routes.items()):
                if route.session == session.name:
                    del self._routes[request_id]
                    failure = Failure(id=request_id, code="worker_exited", message=f"worker exited with {code}")
                    if route.client is not None:
                        await self._send_client(route.client, encode(failure))
                    await self._record(route, failure)
            # Messages still in the inbox wait for the next turn to end or the next message: a wake now could
            # start a crashing worker again and again.
            session.turn = None
            session.exited.set()
            session.ready.set()

    async def _stop(self, session: _Session) -> None:
        handle = session.handle
        if handle is None or session.exited.is_set():
            return
        try:
            await self._send_worker(session.name, StopSession(session=session.name))
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
            identity, payload = await self._workers.recv_multipart()
            name = identity.decode()
            try:
                reply = decode_reply(payload)
            except ValidationError:
                log.exception("invalid reply from worker %s", name)
                continue
            match reply:
                case Hello():
                    if (session := self._sessions.get(name)) is not None:
                        log.info("worker %s ready (pid %d)", name, reply.pid)
                        session.ready.set()
                case StartFailed():
                    if (session := self._sessions.get(name)) is not None:
                        log.warning("worker %s did not start: %s", name, reply.message)
                        session.start_error = reply.message
                        session.start_failed.set()
                        session.ready.set()
                case Event():
                    if (route := self._routes.get(reply.id)) is not None:
                        if route.client is not None:
                            await self._send_client(route.client, payload)
                        await self._record(route, reply)
                case Done() | Failure():
                    if (route := self._routes.pop(reply.id, None)) is not None:
                        if route.client is not None:
                            await self._send_client(route.client, payload)
                        await self._record(route, reply)
                        self._end_turn(route.session, reply.id)

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


async def run(paths: Paths, launcher: Launcher) -> None:
    await Daemon(paths, launcher).serve()
