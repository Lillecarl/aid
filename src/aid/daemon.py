"""The daemon: owns sessions, starts workers, and routes requests and replies between clients and workers."""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import zmq
import zmq.asyncio
from pydantic import ValidationError

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
    Hello,
    ListAgents,
    ListSessions,
    Prompt,
    SessionInfo,
    StartFailed,
    StopSession,
    decode_reply,
    decode_request,
    encode,
)
from aid.spec import AgentSpecAdapter

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


@dataclass
class _Session:
    name: str
    spec: AgentSpec
    handle: WorkerHandle | None = None
    ready: anyio.Event = field(default_factory=anyio.Event)
    exited: anyio.Event = field(default_factory=anyio.Event)
    exit_code: int | None = None
    start_error: str | None = None
    start_failed: anyio.Event = field(default_factory=anyio.Event)
    lock: anyio.Lock = field(default_factory=anyio.Lock)

    @property
    def running(self) -> bool:
        return self.handle is not None and self.ready.is_set() and not self.exited.is_set()


@dataclass(frozen=True)
class _Route:
    client: bytes
    session: str


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
            try:
                spec = AgentSpecAdapter.validate_json(await spec_file.read_bytes())
            except ValidationError:
                log.exception("skipping session %s with an invalid spec", session_dir.name)
                continue
            self._sessions[session_dir.name] = _Session(session_dir.name, spec)
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
                await self._ensure_running(session)
                self._routes[request.id] = _Route(client, session.name)
                await self._send_worker(session.name, request)
                return None
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

    def _session(self, name: str) -> _Session:
        if (session := self._sessions.get(name)) is None:
            raise AidError("not_found", f"no session named {name!r}")
        return session

    async def _create(self, request: CreateSession) -> None:
        if request.name in self._sessions:
            raise AidError("exists", f"session {request.name!r} already exists")
        session = _Session(request.name, request.spec)
        self._sessions[session.name] = session
        session_dir = anyio.Path(self._paths.session_dir(session.name))
        await session_dir.mkdir(parents=True, exist_ok=True)
        await (session_dir / SPEC_FILE).write_bytes(AgentSpecAdapter.dump_json(request.spec))
        try:
            await self._ensure_running(session)
        except BaseException:
            del self._sessions[session.name]
            await anyio.to_thread.run_sync(shutil.rmtree, session_dir, True)
            raise

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
                    await self._send_client(route.client, encode(failure))
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
                        await self._send_client(route.client, payload)
                case Done() | Failure():
                    if (route := self._routes.pop(reply.id, None)) is not None:
                        await self._send_client(route.client, payload)


async def list_agents() -> AgentCatalog:
    """Discover agents in a fresh process: agent modules are user code, and edits since the last call count."""
    with anyio.fail_after(CATALOG_TIMEOUT):
        result = await anyio.run_process([sys.executable, "-m", "aid.catalog"], check=False)
    if result.returncode != 0:
        raise AidError("catalog_failed", result.stderr.decode(errors="replace").strip()[-2000:])
    return AgentCatalog.model_validate_json(result.stdout)


async def run(paths: Paths, launcher: Launcher) -> None:
    await Daemon(paths, launcher).serve()
