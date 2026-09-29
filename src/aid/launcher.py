"""Launchers start workers. The daemon depends only on the `Launcher` protocol.

`ForkserverLauncher` forks each worker from a zygote that has already
imported the configured modules, so workers share those pages copy-on-write.
A subinterpreter launcher can implement the same protocol by calling
`aid.worker.main` in a new interpreter, once pydantic-core and pyzmq load
in more than one interpreter per process.

`CommandLauncher` runs a command per worker (`aid worker`, locally, over
ssh, in a pod) that reaches the daemon on an endpoint of its own, such as
`ws://` through a reverse proxy.
"""

from __future__ import annotations

import dataclasses
import json
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

    from anyio.abc import Process

PRELOAD: Final = ("zmq.asyncio",)
"""What the forkserver imports before forking workers. Only the transport: it is the daemon's side of the
worker channel, so it must work whatever a session puts on `python_path`. Everything else — aid itself,
`pydantic_ai`, `acp` — imports fresh in each worker, so restarting a worker picks up new code. A worker cannot
bring its own version of a preloaded module; add one back only to share its pages with `--preload-module`."""


@dataclass(frozen=True)
class WorkerArgs:
    """Everything a worker needs. Plain strings only, so any launcher can pass it on."""

    endpoint: str
    trust_pem: str
    """For a `wss://` endpoint: the CAs to check its certificate against, PEM; empty means the system's."""
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

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> WorkerArgs:
        return cls(**json.loads(text))


class WorkerHandle(Protocol):
    @property
    def pid(self) -> int | None: ...

    async def wait(self) -> int | None:
        """Return the exit code once the worker ends. Only one task may wait."""
        ...

    def kill(self) -> None: ...


class Launcher(Protocol):
    async def launch(self, args: WorkerArgs) -> WorkerHandle: ...


def process_main(args: WorkerArgs) -> None:
    """A worker process's entry: process-wide setup, then `worker.main`."""
    spec = AgentSpecAdapter.validate_json(args.spec_json)
    os.chdir(spec.cwd)
    os.environ.update(spec.env)
    os.environ[ENV_SESSION] = args.name
    os.environ[ENV_RUNTIME_DIR] = args.daemon_runtime_dir
    os.environ[ENV_STATE_DIR] = args.daemon_state_dir
    worker.main(
        args.endpoint, args.trust_pem, args.server_key, args.public_key, args.secret_key, args.spec_json, args.state_dir
    )


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
        process = self._ctx.Process(target=process_main, args=(args,), name=f"aid-worker-{args.name}")
        await anyio.to_thread.run_sync(process.start)
        return ProcessHandle(process)


class CommandHandle:
    def __init__(self, process: Process) -> None:
        self._process = process

    @property
    def pid(self) -> int | None:
        return self._process.pid

    async def wait(self) -> int | None:
        code = await self._process.wait()
        await self._process.aclose()
        return code

    def kill(self) -> None:
        self._process.kill()


class CommandLauncher:
    """Runs `command` per worker and writes its `WorkerArgs` to the command's stdin as JSON; `aid worker` reads
    them. Stdin, because argv is visible to every user on the host and the args hold the worker's secret key.

    `endpoint` is the daemon's workers socket as the worker reaches it; the daemon must listen there too.
    `trust_pem`: see `WorkerArgs.trust_pem`."""

    def __init__(self, command: Sequence[str], endpoint: str, *, trust_pem: str = "") -> None:
        self._command = list(command)
        self._endpoint = endpoint
        self._trust_pem = trust_pem

    async def launch(self, args: WorkerArgs) -> CommandHandle:
        process = await anyio.open_process(self._command, stdout=None, stderr=None)
        if process.stdin is None:
            raise RuntimeError("the worker command has no stdin")
        async with process.stdin:
            await process.stdin.send(
                dataclasses.replace(args, endpoint=self._endpoint, trust_pem=self._trust_pem).to_json().encode()
            )
        return CommandHandle(process)
