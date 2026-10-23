"""Worker: hosts one agent session and serves the daemon over a zmq DEALER socket.

`main` takes only strings and touches no process-wide state (cwd, environ,
signals), so a subinterpreter can run it as well as a forked process. The
launcher owns process-wide setup.
"""

from __future__ import annotations

import json
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
import anyio.to_thread
import structlog
import zmq
import zmq.asyncio

from aid.backends import open_backend
from aid.backends.base import (
    CompactionBackend,
    FollowingBackend,
    HookBackend,
    PermissionBackend,
    ScreenBackend,
    WatchBackend,
)
from aid.protocol import (
    Activity,
    AnswerPermission,
    Cancel,
    CompactSession,
    Done,
    Event,
    Failure,
    GetPane,
    GetScreen,
    Hello,
    Hook,
    Lifecycle,
    Live,
    Notify,
    Observed,
    Prompt,
    StartFailed,
    StopSession,
    decode_request,
    encode,
)
from aid.spec import AgentSpecAdapter, NativeSpec
from aid.zap import Keypair, connect_curve

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

    from aid.backends.base import Backend
    from aid.protocol import HistoryItem, SessionEvent
    from aid.spec import AgentSpec

log = structlog.get_logger(__name__)


def main(
    endpoint: str,
    trust_pem: str,
    server_key: str,
    public_key: str,
    secret_key: str,
    spec_json: str,
    state_dir: str,
    hook_endpoint: str,
) -> None:
    spec = AgentSpecAdapter.validate_json(spec_json)
    if isinstance(spec, NativeSpec):
        sys.path[:0] = spec.python_path
    keys = Keypair(public=public_key, secret=secret_key)
    anyio.run(serve, endpoint, trust_pem, server_key, keys, spec, anyio.Path(state_dir), hook_endpoint)


async def serve(
    endpoint: str,
    trust_pem: str,
    server_key: str,
    keys: Keypair,
    spec: AgentSpec,
    state_dir: anyio.Path,
    hook_endpoint: str,
) -> None:
    await state_dir.mkdir(parents=True, exist_ok=True)
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 1000)
    connect_curve(sock, endpoint, server_key, keys, trust_pem)
    # The session's own hook socket, bound before the backend starts (Claude spawns inside it): hook.py's
    # fast path, past the daemon. No CURVE: same local-trust domain as control.sock.
    await anyio.to_thread.run_sync(Path(hook_endpoint.removeprefix("ipc://")).unlink, True)
    hook_sock = ctx.socket(zmq.ROUTER)
    hook_sock.setsockopt(zmq.LINGER, 1000)
    hook_sock.bind(hook_endpoint)
    try:
        async with AsyncExitStack() as stack:
            try:
                backend = await stack.enter_async_context(open_backend(spec, state_dir))
            except Exception as error:
                log.exception("backend_start_failed")
                await sock.send(encode(StartFailed(message=describe(error))))
                raise
            tg = await stack.enter_async_context(anyio.create_task_group())
            worker = _Worker(sock, backend, tg)
            await worker.send(Hello(started=backend.started()))
            tg.start_soon(worker.serve_hooks, hook_sock)
            if isinstance(backend, FollowingBackend):
                tg.start_soon(worker.follow, backend)
            if isinstance(backend, WatchBackend):
                tg.start_soon(worker.watch, backend)
            await worker.serve()
    finally:
        hook_sock.close()
        sock.close()
        ctx.term()


def usertee_text(data: bytes) -> str | None:
    """The text of a `usertee` frame (`aid/usertee.py`), or None for any other frame. Its op is local to this
    socket, never a daemon request."""
    message: object = json.loads(data)
    if not isinstance(message, dict):
        return None
    fields = cast(dict[str, object], message)
    text = fields.get("text")
    return text if fields.get("op") == "usertee" and isinstance(text, str) else None


def describe(error: BaseException) -> str:
    """One line per underlying error. A task group's own message says only how many there were."""
    if isinstance(error, BaseExceptionGroup):
        inner: tuple[BaseException, ...] = cast(BaseExceptionGroup[BaseException], error).exceptions
        return "; ".join(describe(e) for e in inner)
    return f"{type(error).__name__}: {error}"


class _Worker:
    def __init__(self, sock: zmq.asyncio.Socket, backend: Backend, tg: TaskGroup) -> None:
        self._sock = sock
        self._backend = backend
        self._tg = tg
        self._send_lock = anyio.Lock()
        self._hook_lock = anyio.Lock()
        self._busy = False
        self._turn: str | None = None

    async def serve_hooks(self, sock: zmq.asyncio.Socket) -> None:
        """Answer `aid/hook.py`'s fast path: Hook requests on the worker's own ROUTER, past the daemon. Pass
        `aid/usertee.py`'s output on to the daemon, answering nothing.
        A bad frame never kills the loop: hooks fail open downstream, and a dead loop would wedge them
        silently."""
        while True:
            try:
                *route, data = await sock.recv_multipart()
                if (text := usertee_text(data)) is not None:
                    await self.send(Live(text=text))
                    continue
                request = decode_request(data)
            except Exception:
                log.exception("hook_frame_dropped")
                continue
            if not isinstance(request, Hook):
                await self._hook_send(
                    sock,
                    route,
                    Failure(id=request.id, code="unsupported", message=f"worker cannot {request.op} over hooks"),
                )
            elif isinstance(self._backend, HookBackend):
                self._tg.start_soon(self._hook_direct, sock, route, self._backend, request)
            else:
                await self._hook_send(sock, route, Done(id=request.id))

    async def _hook_send(self, sock: zmq.asyncio.Socket, route: list[bytes], reply: Done | Failure) -> None:
        async with self._hook_lock:
            await sock.send_multipart([*route, encode(reply)])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

    async def _hook_direct(
        self, sock: zmq.asyncio.Socket, route: list[bytes], backend: HookBackend, request: Hook
    ) -> None:
        await self._hook_send(sock, route, await self._answer_hook(backend, request))

    async def send(self, reply: Hello | Event | Done | Failure | Observed | Activity | Notify | Live) -> None:
        async with self._send_lock:
            await self._sock.send(encode(reply))

    async def notify(self, task_id: str, text: str) -> None:
        await self.send(Notify(task_id=task_id, text=text))

    async def record(self, turn: str, item: HistoryItem) -> None:
        await self.send(Observed(turn=turn, item=item))

    async def follow(self, backend: FollowingBackend) -> None:
        await backend.follow(self.record, self.send)
        log.info("agent_gone")
        self._tg.cancel_scope.cancel()

    async def watch(self, backend: WatchBackend) -> None:
        """The session's watch loop, for the worker's life. A failure here must not take the worker: unlike
        `follow`, its return says nothing about the agent."""
        try:
            await backend.watch(self.notify)
        except Exception:
            log.exception("watch_failed")

    async def serve(self) -> None:
        while True:
            request = decode_request(await self._sock.recv())
            match request:
                case Prompt():
                    if self._busy:
                        await self.send(Failure(id=request.id, code="busy", message="a prompt is already running"))
                    else:
                        self._busy = True
                        self._turn = request.id
                        self._tg.start_soon(self._prompt, request)
                case Cancel():
                    if request.turn is not None and request.turn != self._turn:
                        await self.send(
                            Failure(id=request.id, code="stale_turn", message="the reviewed turn is no longer running")
                        )
                        continue
                    await self._backend.cancel()
                    await self.send(Done(id=request.id))
                case CompactSession():
                    if self._busy:
                        await self.send(Failure(id=request.id, code="busy", message="a prompt is running"))
                    else:
                        self._busy = True
                        self._tg.start_soon(self._compact, request)
                case GetScreen():
                    self._tg.start_soon(self._screen, request)
                case GetPane():
                    if isinstance(self._backend, ScreenBackend):
                        await self.send(Done(id=request.id, data=self._backend.pane_address().model_dump(mode="json")))
                    else:
                        await self.send(
                            Failure(id=request.id, code="no_screen", message="this session has no terminal")
                        )
                case Hook():
                    if isinstance(self._backend, HookBackend):
                        self._tg.start_soon(self._hook, self._backend, request)
                    else:
                        await self.send(Done(id=request.id))
                case AnswerPermission():
                    if isinstance(self._backend, PermissionBackend) and self._backend.answer_permission(
                        request.request_id, request.option_id, request.plugin, request.text, request.supervisor
                    ):
                        await self.send(Done(id=request.id))
                    else:
                        await self.send(
                            Failure(id=request.id, code="no_pending", message="no such permission request waits")
                        )
                case StopSession():
                    await self._backend.cancel()
                    await self.send(Done(id=request.id))
                    self._tg.cancel_scope.cancel()
                    return
                case _:
                    await self.send(Failure(id=request.id, code="unsupported", message=f"worker cannot {request.op}"))

    async def _hook(self, backend: HookBackend, request: Hook) -> None:
        await self.send(await self._answer_hook(backend, request))

    @staticmethod
    async def _answer_hook(backend: HookBackend, request: Hook) -> Done | Failure:
        try:
            answer = await backend.hook(request.event, request.payload)
        except Exception as error:
            log.exception("hook_failed", event=request.event)
            return Failure(id=request.id, code="hook_failed", message=f"{type(error).__name__}: {error}")
        return Done(id=request.id, data=answer)

    async def _screen(self, request: GetScreen) -> None:
        if not isinstance(self._backend, ScreenBackend):
            await self.send(Failure(id=request.id, code="no_screen", message="this session has no terminal"))
            return
        try:
            view = await self._backend.screen(stylesheet=request.stylesheet, since=request.since, wait=request.wait)
        except Exception as error:
            log.exception("screen_failed", id=request.id)
            await self.send(Failure(id=request.id, code="screen_failed", message=f"{type(error).__name__}: {error}"))
        else:
            await self.send(Done(id=request.id, data=view.model_dump(mode="json")))

    async def _compact(self, request: CompactSession) -> None:
        try:
            if not isinstance(self._backend, CompactionBackend):
                await self.send(
                    Failure(id=request.id, code="unsupported", message="this session cannot compact itself")
                )
                return
            digest = await self._backend.compact(request.instructions)
            await self.record(uuid.uuid4().hex, Lifecycle(event="compacted", detail="manual", summary=digest))
            await self.send(Done(id=request.id, data=digest))
        except Exception as error:
            log.exception("compact_failed", id=request.id)
            await self.send(Failure(id=request.id, code="compact_failed", message=f"{type(error).__name__}: {error}"))
        finally:
            self._busy = False

    async def _prompt(self, request: Prompt) -> None:
        async def emit(event: SessionEvent) -> None:
            await self.send(Event(id=request.id, event=event))

        try:
            output = await self._backend.prompt(request.text, emit)
        except Exception as error:
            log.exception("prompt_failed", id=request.id)
            await self.send(Failure(id=request.id, code="agent_error", message=f"{type(error).__name__}: {error}"))
        else:
            await emit(output)
            await self.send(Done(id=request.id))
        finally:
            self._turn = None
            self._busy = False
