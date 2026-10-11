from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest
import zmq.asyncio

from aid.protocol import Done, Hook, decode_reply, encode
from aid.worker import _Worker  # pyright: ignore[reportPrivateUsage] -- the worker's serve loop is the unit under test

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from aid.backends.base import Emit
    from aid.protocol import Output, Started

pytestmark = pytest.mark.anyio


class _HookBackend:
    """A backend whose hooks answer from a fixed value, recording what they heard."""

    def __init__(self, answer: JsonValue) -> None:
        self.answer = answer
        self.seen: list[tuple[str, JsonValue]] = []

    def started(self) -> Started:
        raise AssertionError("unreachable")

    async def prompt(self, text: str, emit: Emit) -> Output:
        raise AssertionError("unreachable")

    async def cancel(self) -> None:
        raise AssertionError("unreachable")

    async def hook(self, event: str, payload: JsonValue) -> JsonValue:
        self.seen.append((event, payload))
        return self.answer


class _PlainBackend:
    """A backend without hooks: hook.py hears nothing from it."""

    def started(self) -> Started:
        raise AssertionError("unreachable")

    async def prompt(self, text: str, emit: Emit) -> Output:
        raise AssertionError("unreachable")

    async def cancel(self) -> None:
        raise AssertionError("unreachable")


async def ask(client: zmq.asyncio.Socket, request: Hook) -> Done:
    await client.send(encode(request))
    with anyio.fail_after(5):
        reply = decode_reply(await client.recv())
    assert isinstance(reply, Done)
    return reply


async def run_server(
    tmp_path: Path, backend: _HookBackend | _PlainBackend, request: Hook, extra: bytes | None = None
) -> Done:
    """A worker's serve_hooks on a bound ROUTER, asked once over a DEALER. Closed and cancelled after."""
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(f"ipc://{tmp_path / 'hook.sock'}")
    daemon_sock = ctx.socket(zmq.DEALER)
    daemon_sock.setsockopt(zmq.LINGER, 0)
    client = ctx.socket(zmq.DEALER)
    client.setsockopt(zmq.LINGER, 0)
    client.connect(f"ipc://{tmp_path / 'hook.sock'}")
    replies: list[Done] = []
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_Worker(daemon_sock, backend, tg).serve_hooks, router)
            await anyio.sleep(0.1)
            if extra is not None:
                await client.send(extra)
            replies.append(await ask(client, request))
            tg.cancel_scope.cancel()
    finally:
        client.close()
        daemon_sock.close()
        router.close()
        ctx.term()
    assert len(replies) == 1
    return replies[0]


async def test_serve_hooks_answers_from_the_backend(tmp_path: Path) -> None:
    backend = _HookBackend({"fast": True})
    reply = await run_server(tmp_path, backend, Hook(id="h1", session="s", event="Stop", payload={"a": 1}))
    assert reply == Done(id="h1", data={"fast": True})
    assert backend.seen == [("Stop", {"a": 1})]


async def test_serve_hooks_without_a_hook_backend_says_nothing(tmp_path: Path) -> None:
    reply = await run_server(tmp_path, _PlainBackend(), Hook(id="h1", session="s", event="Stop", payload=None))
    assert reply == Done(id="h1", data=None)


async def test_serve_hooks_survives_a_bad_frame(tmp_path: Path) -> None:
    backend = _HookBackend(None)
    reply = await run_server(
        tmp_path,
        backend,
        Hook(id="h1", session="s", event="Stop", payload=None),
        extra=b"not a hook request",
    )
    assert reply == Done(id="h1", data=None)
    assert backend.seen == [("Stop", None)]
