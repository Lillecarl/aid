"""Worker: hosts one agent session and serves the daemon over a zmq DEALER socket.

`main` takes only strings and touches no process-wide state (cwd, environ,
signals), so a subinterpreter can run it as well as a forked process. The
launcher owns process-wide setup.
"""

from __future__ import annotations

import sys
import uuid
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, cast

import anyio
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
    Notify,
    Observed,
    Prompt,
    StartFailed,
    StopSession,
    decode_request,
    encode,
)
from aid.spec import AgentSpecAdapter, PydanticAISpec
from aid.zap import Keypair, connect_curve

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

    from aid.backends.base import Backend
    from aid.protocol import HistoryItem, SessionEvent
    from aid.spec import AgentSpec

log = structlog.get_logger(__name__)


def main(
    endpoint: str, trust_pem: str, server_key: str, public_key: str, secret_key: str, spec_json: str, state_dir: str
) -> None:
    spec = AgentSpecAdapter.validate_json(spec_json)
    if isinstance(spec, PydanticAISpec):
        sys.path[:0] = spec.python_path
    keys = Keypair(public=public_key, secret=secret_key)
    anyio.run(serve, endpoint, trust_pem, server_key, keys, spec, anyio.Path(state_dir))


async def serve(
    endpoint: str, trust_pem: str, server_key: str, keys: Keypair, spec: AgentSpec, state_dir: anyio.Path
) -> None:
    await state_dir.mkdir(parents=True, exist_ok=True)
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 1000)
    connect_curve(sock, endpoint, server_key, keys, trust_pem)
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
            if isinstance(backend, FollowingBackend):
                tg.start_soon(worker.follow, backend)
            if isinstance(backend, WatchBackend):
                tg.start_soon(worker.watch, backend)
            await worker.serve()
    finally:
        sock.close()
        ctx.term()


def describe(error: BaseException) -> str:
    """One line per underlying error. A task group's own message says only how many there were."""
    if isinstance(error, BaseExceptionGroup):
        inner: tuple[BaseException, ...] = cast("BaseExceptionGroup[BaseException]", error).exceptions
        return "; ".join(describe(e) for e in inner)
    return f"{type(error).__name__}: {error}"


class _Worker:
    def __init__(self, sock: zmq.asyncio.Socket, backend: Backend, tg: TaskGroup) -> None:
        self._sock = sock
        self._backend = backend
        self._tg = tg
        self._send_lock = anyio.Lock()
        self._busy = False

    async def send(self, reply: Hello | Event | Done | Failure | Observed | Activity | Notify) -> None:
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
                        self._tg.start_soon(self._prompt, request)
                case Cancel():
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
                        request.request_id, request.option_id, request.plugin, request.text
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
        try:
            answer = await backend.hook(request.event, request.payload)
        except Exception as error:
            log.exception("hook_failed", event=request.event)
            await self.send(Failure(id=request.id, code="hook_failed", message=f"{type(error).__name__}: {error}"))
        else:
            await self.send(Done(id=request.id, data=answer))

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
            self._busy = False
