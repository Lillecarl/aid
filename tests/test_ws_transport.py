from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
import zmq.asyncio

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.anyio


def test_libzmq_has_ws() -> None:
    # libzmq 4.3.5 spells the capability in upper case; zmq.has("ws") is False on the same build.
    assert zmq.has("WS")
    assert zmq.has("curve")


@contextmanager
def _pair(server_key: bytes | None) -> Generator[tuple[zmq.asyncio.Socket, zmq.asyncio.Socket]]:
    ctx = zmq.asyncio.Context()
    router, dealer = ctx.socket(zmq.ROUTER), ctx.socket(zmq.DEALER)
    public, secret = zmq.curve_keypair()
    router.curve_server = True
    router.curve_secretkey = secret
    router.curve_publickey = public
    dealer.curve_serverkey = public if server_key is None else server_key
    dealer.curve_publickey, dealer.curve_secretkey = zmq.curve_keypair()
    router.bind("ws://127.0.0.1:*/aid")
    dealer.connect(router.get_string(zmq.LAST_ENDPOINT))
    try:
        yield router, dealer
    finally:
        ctx.destroy(linger=0)


async def test_curve_over_ws() -> None:
    with _pair(None) as (router, dealer), anyio.fail_after(5):
        await dealer.send(b"hello")
        ident, msg = await router.recv_multipart()
        await router.send_multipart([ident, b"back"])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
        assert (msg, await dealer.recv()) == (b"hello", b"back")


async def test_curve_over_ws_refuses_the_wrong_server_key() -> None:
    with _pair(zmq.curve_keypair()[0]) as (router, dealer):
        await dealer.send(b"hello")
        with anyio.move_on_after(1) as scope:
            await router.recv_multipart()
        assert scope.cancelled_caught
