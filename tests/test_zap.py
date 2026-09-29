from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
import zmq.asyncio
import zmq.utils.z85

from aid import zap

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["ipc", "ws"])
def endpoint(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    return f"ipc://{tmp_path / 'workers.sock'}" if request.param == "ipc" else "ws://127.0.0.1:*/aid"


@asynccontextmanager
async def _daemon_side(endpoint: str, keys: zap.Keys) -> AsyncGenerator[tuple[zmq.asyncio.Context, zmq.asyncio.Socket]]:
    ctx = zmq.asyncio.Context()
    try:
        handler = ctx.socket(zmq.REP)
        handler.bind(zap.ZAP_ENDPOINT)
        router = ctx.socket(zmq.ROUTER)
        zap.serve_curve(router, keys)
        router.bind(endpoint)
        async with anyio.create_task_group() as tg:
            tg.start_soon(zap.handle, handler, keys)
            yield ctx, router
            tg.cancel_scope.cancel()
    finally:
        ctx.destroy(linger=0)


def _worker(ctx: zmq.asyncio.Context, router: zmq.asyncio.Socket, server: str, keys: zap.Keypair) -> zmq.asyncio.Socket:
    sock = ctx.socket(zmq.DEALER)
    sock.curve_serverkey = server.encode()
    sock.curve_publickey = keys.public.encode()
    sock.curve_secretkey = keys.secret.encode()
    sock.connect(router.get_string(zmq.LAST_ENDPOINT))
    return sock


async def _user_id(router: zmq.asyncio.Socket) -> str | None:
    with anyio.move_on_after(1):
        _peer, frame = await router.recv_multipart(copy=False)
        return frame.get("User-Id")  # pyright: ignore[reportArgumentType, reportReturnType] -- see daemon._worker_loop
    return None


async def test_an_issued_key_names_its_session(endpoint: str) -> None:
    keys = zap.Keys()
    async with _daemon_side(endpoint, keys) as (ctx, router):
        worker = _worker(ctx, router, keys.server.public, keys.issue("alice"))
        worker.routing_id = b"bob"  # a claim, which the daemon ignores
        await worker.send(b"hello")
        assert await _user_id(router) == "alice"


async def test_a_key_not_issued_is_refused(endpoint: str) -> None:
    keys = zap.Keys()
    keys.issue("alice")
    async with _daemon_side(endpoint, keys) as (ctx, router):
        stranger = _worker(ctx, router, keys.server.public, zap.Keypair.new())
        await stranger.send(b"hello")
        assert await _user_id(router) is None


async def test_a_new_launch_revokes_the_last_key(endpoint: str) -> None:
    keys = zap.Keys()
    old = keys.issue("alice")
    keys.issue("alice")
    async with _daemon_side(endpoint, keys) as (ctx, router):
        stale = _worker(ctx, router, keys.server.public, old)
        await stale.send(b"hello")
        assert await _user_id(router) is None


def test_revoke_leaves_a_newer_key() -> None:
    keys = zap.Keys()
    old = keys.issue("alice")
    new = keys.issue("alice")
    keys.revoke("alice", old.public)
    raw: bytes = zmq.utils.z85.decode(new.public.encode())  # pyright: ignore[reportUnknownMemberType] -- untyped in pyzmq
    request = [b"1.0", b"1", zap.DOMAIN, b"", b"", b"CURVE", raw]
    assert keys.reply(request)[2:5] == [b"200", b"", b"alice"]
    keys.revoke("alice", new.public)
    assert keys.reply(request)[2] == b"400"
