"""Launchers start workers. The daemon depends only on the `Launcher` protocol.

`ForkserverLauncher` forks each worker from a zygote that has already
imported the heavy modules, so workers share those pages copy-on-write.
A subinterpreter launcher can implement the same protocol by calling
`aid.worker.main` in a new interpreter, once pydantic-core and pyzmq load
in more than one interpreter per process.
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import anyio
import anyio.to_thread

from aid import worker
from aid.paths import ENV_RUNTIME_DIR, ENV_SESSION, ENV_STATE_DIR
from aid.spec import AgentSpecAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.context import ForkServerProcess

PRELOAD: Final = ("aid.launcher", "aid.worker", "pydantic_ai", "acp", "zmq.asyncio")


@dataclass(frozen=True)
class WorkerArgs:
    """Everything a worker needs. Plain strings only, so any launcher can pass it on."""

    endpoint: str
    server_key: str
    public_key: str
    secret_key: str
    """CURVE keys, Z85: the daemon's public key, and the keypair the daemon issued to this worker."""
    name: str
    spec_json: str
    state_dir: str
    daemon_runtime_dir: str
    daemon_state_dir: str
    """The daemon's own paths, so code in the worker that calls `aid.connect()` finds this daemon."""


class WorkerHandle(Protocol):
    @property
    def pid(self) -> int | None: ...

    async def wait(self) -> int | None:
        """Return the exit code once the worker ends. Only one task may wait."""
        ...

    def kill(self) -> None: ...


class Launcher(Protocol):
    async def launch(self, args: WorkerArgs) -> WorkerHandle: ...


def _process_main(args: WorkerArgs) -> None:
    spec = AgentSpecAdapter.validate_json(args.spec_json)
    os.chdir(spec.cwd)
    os.environ.update(spec.env)
    os.environ[ENV_SESSION] = args.name
    os.environ[ENV_RUNTIME_DIR] = args.daemon_runtime_dir
    os.environ[ENV_STATE_DIR] = args.daemon_state_dir
    worker.main(args.endpoint, args.server_key, args.public_key, args.secret_key, args.spec_json, args.state_dir)


class ProcessHandle:
    def __init__(self, process: ForkServerProcess) -> None:
        self._process = process

    @property
    def pid(self) -> int | None:
        return self._process.pid

    async def wait(self) -> int | None:
        await anyio.wait_readable(self._process.sentinel)
        self._process.join()
        return self._process.exitcode

    def kill(self) -> None:
        self._process.kill()


class ForkserverLauncher:
    def __init__(self, preload: Sequence[str] = PRELOAD) -> None:
        self._ctx = multiprocessing.get_context("forkserver")
        self._ctx.set_forkserver_preload(list(preload))

    async def launch(self, args: WorkerArgs) -> ProcessHandle:
        process = self._ctx.Process(target=_process_main, args=(args,), name=f"aid-worker-{args.name}")
        await anyio.to_thread.run_sync(process.start)
        return ProcessHandle(process)
