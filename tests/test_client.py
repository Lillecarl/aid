from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
import zmq.asyncio

from aid.client import Client, Endpoints
from aid.protocol import AidError, ListSessions

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


async def test_calls_fail_closed_once_nothing_reads_replies(tmp_path: Path) -> None:
    """A shielded cleanup calling after its daemon went must fail, not wait for a reply nothing reads (#61)."""
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    nowhere = f"ipc://{tmp_path}/nobody.sock"
    sock.connect(nowhere)
    client = Client(sock, ctx, Endpoints(nowhere, nowhere))
    failures: list[str] = []

    async def waiting() -> None:
        with anyio.CancelScope(shield=True):
            try:
                await client.call(ListSessions())
            except AidError as error:
                failures.append(error.code)

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(client.read_replies)
                tg.start_soon(waiting)
                await anyio.sleep(0.2)
                tg.cancel_scope.cancel()
            with pytest.raises(AidError, match="closed"):
                await client.call(ListSessions())
    finally:
        sock.close(linger=0)
        ctx.term()
    assert failures == ["closed"]
