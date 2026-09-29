from __future__ import annotations

import socket
import sys
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import Daemon
from aid.launcher import CommandLauncher, ForkserverLauncher
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.usefixtures("empty_agents_path")
async def test_a_worker_command_over_ws(paths: Paths, tmp_path: Path) -> None:
    endpoint = f"ws://127.0.0.1:{_free_port()}/aid"
    launcher = CommandLauncher([sys.executable, "-m", "aid", "worker"], endpoint)
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, workers_listen=[endpoint]).serve)
            async with aid.connect(paths) as client:
                session = await client.create("far", py_spec(tmp_path, "agents:echo", AID_TEST_VAR="remote"))
                result = await session.run("env")
                assert result.output == f"remote {tmp_path}"
                assert (await session.status()).pid is not None
                await session.stop()
            tg.cancel_scope.cancel()


@pytest.mark.usefixtures("empty_agents_path")
async def test_per_session_worker_command(paths: Paths, tmp_path: Path) -> None:
    """One session runs its own worker command while the daemon forks the rest: new worker code without
    restarting the daemon or touching other sessions."""
    command = [sys.executable, "-m", "aid", "worker"]
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, ForkserverLauncher()).serve)
            async with aid.connect(paths) as client:
                spec = py_spec(tmp_path, "agents:echo").model_copy(update={"worker_command": command})
                session = await client.create("sub", spec)
                assert (await session.run("hi")).output == "turn 1: echo hi"
                assert (await session.status()).pid is not None
                await session.stop()
            tg.cancel_scope.cancel()


@pytest.mark.usefixtures("empty_agents_path")
async def test_shutdown_while_a_worker_starts(paths: Paths, tmp_path: Path) -> None:
    """The worker never reaches the daemon; stopping the daemon must still end it, well within `STOP_TIMEOUT`."""
    launcher = CommandLauncher([sys.executable, "-m", "aid", "worker"], f"ws://127.0.0.1:{_free_port()}/aid")
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher).serve)
            async with aid.connect(paths) as client:
                with anyio.move_on_after(1):
                    await client.create("stuck", py_spec(tmp_path, "agents:echo"))
            tg.cancel_scope.cancel()
