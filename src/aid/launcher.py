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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import anyio
import anyio.to_thread
import structlog

from aid import usertee, worker
from aid.control import ENV_AGENT_CONNECTION, ENV_MODEL_SLOTS, AgentConnection
from aid.log import configure_logging
from aid.paths import ENV_RUNTIME_DIR, ENV_SESSION, ENV_STATE_DIR, hook_socket
from aid.spec import AgentSpecAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.context import ForkServerProcess

    from anyio.abc import Process

log = structlog.get_logger(__name__)

PRELOAD: Final = ("zmq.asyncio",)
"""What the forkserver imports before forking workers. Only the transport: it is the daemon's side of the
worker channel, so it must work whatever a session puts on `python_path`. Everything else — aid itself,
the model SDKs, `acp` — imports fresh in each worker, so restarting a worker picks up new code. A worker cannot
bring its own version of a preloaded module; add one back only to share its pages with `--preload-module`."""


@dataclass(frozen=True)
class WorkerArgs:
    """Everything a worker needs. Plain strings only, so any launcher can pass it on."""

    endpoint: str
    trust_pem: str
    """For a `wss://` endpoint: the CAs to check its certificate against, PEM; empty means the system's."""
    server_key: str
    public_key: str
    secret_key: str = dataclasses.field(repr=False)
    """CURVE keys, Z85: the daemon's public key, and the keypair the daemon issued to this worker."""
    agent_public_key: str
    agent_secret_key: str = dataclasses.field(repr=False)
    name: str
    spec_json: str
    state_dir: str
    daemon_runtime_dir: str
    daemon_state_dir: str
    """The daemon's own paths, so code in the worker that calls `aid.connect()` finds this daemon."""
    model_slots: str = "{}"
    """The daemon's model slots as JSON, host → capacity (`aid.slots`): the worker knows them without asking."""

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


def install_usertee(runtime_dir: str) -> str | None:
    """Write the `usertee` command (`aid/usertee.py`) into `runtime_dir/bin`, for agents' pipelines, and return
    that directory; None when it cannot be written, which costs agents the command, never the worker. A shim runs
    the file with this interpreter, which has zmq; written whole and renamed, so a worker starting beside another
    never runs half of one."""
    directory = Path(runtime_dir) / "bin"
    shim = directory / "usertee"
    script = f'#!/bin/sh\nexec "{sys.executable}" "{usertee.__file__}" "$@"\n'
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not shim.exists() or shim.read_text() != script:
            partial = directory / f".usertee.{os.getpid()}"
            partial.write_text(script)
            partial.chmod(0o755)
            partial.replace(shim)
    except OSError as error:
        log.warning("usertee_not_installed", directory=str(directory), error=str(error))
        return None
    return str(directory)


def process_main(args: WorkerArgs) -> None:
    """A worker process's entry: process-wide setup, then `worker.main`."""
    configure_logging()
    structlog.contextvars.bind_contextvars(session=args.name)
    spec = AgentSpecAdapter.validate_json(args.spec_json)
    os.chdir(spec.cwd)
    os.environ.update(spec.env)
    os.environ[ENV_SESSION] = args.name
    os.environ[ENV_RUNTIME_DIR] = args.daemon_runtime_dir
    os.environ[ENV_STATE_DIR] = args.daemon_state_dir
    os.environ[ENV_MODEL_SLOTS] = args.model_slots
    if (bin_dir := install_usertee(args.daemon_runtime_dir)) is not None:
        os.environ["PATH"] = os.pathsep.join([bin_dir, os.environ.get("PATH", "")])
    os.environ[ENV_AGENT_CONNECTION] = AgentConnection(
        endpoint=args.endpoint,
        server_key=args.server_key,
        public_key=args.agent_public_key,
        secret_key=args.agent_secret_key,
        trust_pem=args.trust_pem,
    ).model_dump_json()
    worker.main(
        args.endpoint,
        args.trust_pem,
        args.server_key,
        args.public_key,
        args.secret_key,
        args.spec_json,
        args.state_dir,
        hook_socket(args.daemon_runtime_dir, args.name),
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
