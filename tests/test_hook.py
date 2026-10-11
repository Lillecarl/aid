from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq.asyncio

from aid import hook
from aid.protocol import Done, Failure, Hook, RequestAdapter, decode_request, encode

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio


def test_the_request_is_aids_hook_request() -> None:
    payload = {"hook_event_name": "Stop", "session_id": "s", "nested": {"a": [1, None]}}
    parsed = RequestAdapter.validate_python(hook.request("tty", "Stop", payload))
    assert isinstance(parsed, Hook)
    assert (parsed.session, parsed.event, parsed.payload) == ("tty", "Stop", payload)


async def run_hook(control: str, session: str, event: str, hook_endpoint: str | None = None) -> tuple[int, str, float]:
    started = time.monotonic()
    argv = [sys.executable, hook.__file__, control, session, event]
    if hook_endpoint is not None:
        argv.append(hook_endpoint)
    result = await anyio.run_process(
        argv,
        input=json.dumps({"hook_event_name": event}).encode(),
        check=False,
    )
    return result.returncode, result.stdout.decode(), time.monotonic() - started


async def test_no_daemon_means_nothing_to_say(tmp_path: Path) -> None:
    code, out, took = await run_hook(f"ipc://{tmp_path / 'none.sock'}", "tty", "UserPromptSubmit")
    assert (code, out) == (0, "")
    assert took < hook.QUICK_WAIT


async def test_an_unknown_session_means_nothing_to_say(daemon: Paths) -> None:
    code, out, _ = await run_hook(daemon.control, "nobody", "Stop")
    assert (code, out) == (0, "")


async def test_hook_socket_answers_without_the_daemon(tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    endpoint = f"ipc://{tmp_path / 'hook.sock'}"
    router.bind(endpoint)
    try:
        async with anyio.create_task_group() as tg:
            received: list[str] = []

            async def answer_once() -> None:
                frames = await router.recv_multipart()
                request = decode_request(frames[-1])
                assert isinstance(request, Hook)
                received.append(request.event)
                await router.send_multipart([*frames[:-1], encode(Done(id=request.id, data={"fast": True}))])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

            tg.start_soon(answer_once)
            code, out, took = await run_hook(f"ipc://{tmp_path / 'none.sock'}", "tty", "Stop", endpoint)
            assert code == 0
            assert json.loads(out) == {"fast": True}
            assert received == ["Stop"]
            assert took < hook.QUICK_WAIT
            tg.cancel_scope.cancel()
    finally:
        router.close()
        ctx.term()


async def test_dead_hook_socket_falls_back_to_control(tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    control = ctx.socket(zmq.ROUTER)
    control.setsockopt(zmq.LINGER, 0)
    control.bind(f"ipc://{tmp_path / 'control.sock'}")
    try:
        async with anyio.create_task_group() as tg:
            received: list[str] = []

            async def answer_once() -> None:
                frames = await control.recv_multipart()
                request = decode_request(frames[-1])
                assert isinstance(request, Hook)
                received.append(request.session)
                await control.send_multipart([*frames[:-1], encode(Done(id=request.id, data="via-control"))])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence

            tg.start_soon(answer_once)
            code, out, _ = await run_hook(
                f"ipc://{tmp_path / 'control.sock'}", "tty", "Stop", f"ipc://{tmp_path / 'gone.sock'}"
            )
            assert code == 0
            assert json.loads(out) == "via-control"
            assert received == ["tty"]
            tg.cancel_scope.cancel()
    finally:
        control.close()
        ctx.term()


async def test_hook_failure_does_not_fall_back(tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(f"ipc://{tmp_path / 'hook.sock'}")
    control = ctx.socket(zmq.ROUTER)
    control.setsockopt(zmq.LINGER, 0)
    control.bind(f"ipc://{tmp_path / 'control.sock'}")
    try:
        async with anyio.create_task_group() as tg:

            async def fail_once() -> None:
                frames = await router.recv_multipart()
                request = decode_request(frames[-1])
                await router.send_multipart(  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
                    [*frames[:-1], encode(Failure(id=request.id, code="hook_failed", message="boom"))]
                )

            tg.start_soon(fail_once)
            code, out, _ = await run_hook(
                f"ipc://{tmp_path / 'control.sock'}",
                "tty",
                "Stop",
                f"ipc://{tmp_path / 'hook.sock'}",
            )
            assert (code, out) == (0, "")
            asked = False
            with anyio.move_on_after(0.3):
                await control.recv_multipart()
                asked = True
            assert not asked
            tg.cancel_scope.cancel()
    finally:
        router.close()
        control.close()
        ctx.term()
