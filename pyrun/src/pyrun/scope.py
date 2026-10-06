"""Scopes: where processes start, what they are recorded as, and what kills them.

A scope marks every process it starts, and every descendant that keeps its environment, with PYRUN_SCOPE and
PYRUN_ID, each a chain of the enclosing ids. Killing finds them by that mark in /proc, so a descendant that left its process group (`setsid`, a
double fork) is found too; only one that clears its environment escapes. No process-wide state is touched: no
subreaper, no signal handlers.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread

from pyrun.result import Denied, Output, Running, TimedOut

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
    from types import TracebackType

    from anyio.abc import ByteReceiveStream, Process
    from anyio.streams.memory import MemoryObjectSendStream

    from pyrun.command import Command

    type Policy = Callable[[Command], Awaitable[bool]]

SCOPE_MARK: Final = "PYRUN_SCOPE"
ID_MARK: Final = "PYRUN_ID"
MEMORY_LIMIT: Final = 16 * 1024 * 1024
"""Bytes of each stream a Result keeps in memory; the rest is read back from the record."""
KILL_GRACE: Final = 2.0
"""Seconds between SIGTERM and SIGKILL."""
KILL_POLL: Final = 0.05
ENV_STORE: Final = "PYRUN_STORE"

_current: ContextVar[Scope | None] = ContextVar("pyrun_scope", default=None)


def default_store() -> Path:
    if store := os.environ.get(ENV_STORE):
        return Path(store)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "pyrun" / "runs"


def current() -> Scope | None:
    return _current.get()


@dataclass(frozen=True)
class Marked:
    pid: int
    argv: list[str]
    id: str | None


def _chain(environ: list[bytes], key: str) -> list[str]:
    prefix = f"{key}=".encode()
    return next((v.removeprefix(prefix).decode().split(":") for v in environ if v.startswith(prefix)), list[str]())


def marked(key: str, value: str) -> list[Marked]:
    """Live processes of this user whose `key` chain holds `value`, from /proc. Zombies have no environment.

    The marks are chains, `outer:inner`, because scopes nest: a script's scope runs inside its host's, and the host
    must find the script's processes too."""
    found: list[Marked] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            environ = Path(f"/proc/{entry.name}/environ").read_bytes().split(b"\0")
            if value not in _chain(environ, key):
                continue
            argv = Path(f"/proc/{entry.name}/cmdline").read_bytes().split(b"\0")[:-1]
        except OSError:  # Gone, or not ours.
            continue
        ids = _chain(environ, ID_MARK)
        found.append(Marked(int(entry.name), [a.decode(errors="replace") for a in argv], ids[-1] if ids else None))
    return found


def _extend(chain: str | None, link: str) -> str:
    return f"{chain}:{link}" if chain else link


def _signal(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):  # It ended since the scan.
        os.kill(pid, sig)


async def terminate(key: str, value: str) -> None:
    """SIGTERM every process marked `key=value`, SIGKILL whatever is left after KILL_GRACE."""
    deadline = time.monotonic() + KILL_GRACE
    sig = signal.SIGTERM
    while found := await anyio.to_thread.run_sync(marked, key, value):
        for proc in found:
            _signal(proc.pid, sig)
        if sig == signal.SIGKILL and time.monotonic() > deadline + KILL_GRACE:
            return  # Unkillable (uninterruptible sleep); nothing more to do from here.
        if time.monotonic() > deadline:
            sig = signal.SIGKILL
        await anyio.sleep(KILL_POLL)


class _Tee:
    """One output stream of a process: to its record file, to memory up to MEMORY_LIMIT, and to a reader."""

    def __init__(self, path: Path, forward: MemoryObjectSendStream[bytes] | None = None) -> None:
        self.path = path
        self.head = bytearray()
        self.size = 0
        self._forward = forward

    async def pump(self, stream: ByteReceiveStream) -> None:
        try:
            async with await anyio.open_file(self.path, "wb") as f:
                async for chunk in stream:
                    await f.write(chunk)
                    await f.flush()
                    self.size += len(chunk)
                    if len(self.head) < MEMORY_LIMIT:
                        self.head += chunk[: MEMORY_LIMIT - len(self.head)]
                    if self._forward is not None:
                        await self._forward.send(chunk)
        except anyio.BrokenResourceError:
            pass
        finally:
            if self._forward is not None:
                await self._forward.aclose()

    def output(self) -> Output:
        return Output(path=self.path, head=bytes(self.head), size=self.size)


@dataclass
class Spawned:
    """A process a scope started, without pumps: the holder drains every PIPE it asked for, and the handle
    itself holds no scope — no task group, no cancel scope — so entering it in one task and closing it in
    another is sound. Closing kills what is left running and finalizes the record."""

    id: str
    record: Path
    command: Command
    process: Process
    """anyio's handle: `process.stdout` and `process.stderr` are the pipes the holder drains."""
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None

    async def close(self) -> None:
        """Kill whatever is left running and wait for it. Shielded, and safe to call twice: killing finds
        nothing the second time, and waiting returns at once."""
        with anyio.CancelScope(shield=True):
            await terminate(ID_MARK, self.id)
            await self.process.wait()


@dataclass
class Launched:
    """A process a scope started, until its record is closed."""

    id: str
    record: Path
    command: Command
    process: Process
    stdout: _Tee | None
    stderr: _Tee
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None

    def running(self) -> list[Running]:
        tail = self.stdout.output().tail() if self.stdout else ""
        return [
            Running(pid=m.pid, argv=m.argv, id=self.id if m.pid == self.process.pid else None, last_lines=tail)
            for m in marked(ID_MARK, self.id)
        ]


class Scope:
    """Owns the processes started in it; see the module docstring for how it finds them."""

    def __init__(
        self,
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str | None] | None = None,
        store: str | os.PathLike[str] | None = None,
        policy: Policy | None = None,
        timeout: float | None = None,
        id: str | None = None,
    ) -> None:
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.env = dict(env or {})
        self.store = Path(store) if store is not None else default_store()
        self.policy = policy
        self.timeout = timeout
        self.id = id or secrets.token_hex(3)
        self.dir = self.store / self.id
        self._count = 0
        self._cancel: anyio.CancelScope | None = None
        self._token: object = None
        self._live: dict[str, Spawned | Launched] = {}
        self._at_timeout: list[Running] = []
        """What was running when the scope's timeout struck."""

    async def __aenter__(self) -> Scope:
        await anyio.Path(self.dir).mkdir(parents=True, exist_ok=True)
        deadline = anyio.current_time() + self.timeout if self.timeout is not None else float("inf")
        self._cancel = anyio.CancelScope(deadline=deadline)
        self._cancel.__enter__()
        self._token = _current.set(self)
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None
    ) -> None:
        _current.reset(self._token)  # pyright: ignore[reportArgumentType] -- the Token __aenter__ set
        cancel = self._cancel
        if cancel is None:
            raise RuntimeError("scope was not entered")
        with anyio.CancelScope(shield=True):
            await terminate(SCOPE_MARK, self.id)
        # Its deadline is the only thing that cancels this scope, and that is the timeout: the one swallowed case.
        cancel.__exit__(kind, error, tb)
        if cancel.cancelled_caught and self.timeout is not None:
            raise TimedOut(f"scope {self.id}", self.timeout, self._at_timeout)

    def environment(self, command: Command, id: str) -> dict[str, str]:
        env = dict(os.environ)
        for key, value in {**self.env, **command.env}.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        env[SCOPE_MARK] = _extend(os.environ.get(SCOPE_MARK), self.id)
        env[ID_MARK] = _extend(os.environ.get(ID_MARK), id)
        return env

    async def check(self, command: Command) -> None:
        if self.policy is not None and not await self.policy(command):
            raise Denied(command)

    def _allocate(self) -> tuple[str, Path]:
        self._count += 1
        id = f"{self.id}.{self._count}"
        return id, self.dir / str(self._count)

    @asynccontextmanager
    async def spawn(
        self,
        command: Command,
        *,
        stdin: int,
        stdout: int,
        stderr: int = subprocess.PIPE,
    ) -> AsyncGenerator[Spawned]:
        """Start `command` and record it, without pumps: the holder drains every PIPE it asked for, and the
        handle holds no scope, so it survives turns, tasks and timeout scopes. On exit, whatever it left
        running is killed and the record is closed. `launch` is this plus pumps; a held `launch` would park
        its pump group on one task's scope stack, which any later scope exit breaks.

        `stdin`/`stdout`/`stderr`: subprocess.PIPE, DEVNULL, or a file descriptor."""
        id, record = self._allocate()
        await anyio.Path(record).mkdir(parents=True)
        cwd = self.cwd / command.cwd if command.cwd is not None else self.cwd
        meta = {
            "id": id,
            "argv": list(command.argv),
            "cwd": str(cwd),
            "env": sorted({**self.env, **command.env}),
            "started": time.time(),
        }
        await anyio.Path(record / "command.json").write_text(json.dumps(meta))
        process = await anyio.open_process(
            list(command.argv),
            cwd=cwd,
            env=self.environment(command, id),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        spawned = Spawned(id=id, record=record, command=command, process=process)
        self._live[id] = spawned
        try:
            yield spawned
        finally:
            await spawned.close()
            spawned.ended = time.monotonic()
            del self._live[id]
            with anyio.CancelScope(shield=True):
                code = process.returncode
                meta |= {
                    "ended": time.time(),
                    "code": code if code is not None and code >= 0 else None,
                    "signal": -code if code is not None and code < 0 else None,
                }
                await anyio.Path(record / "command.json").write_text(json.dumps(meta))

    @asynccontextmanager
    async def launch(
        self,
        command: Command,
        *,
        stdin: int,
        stdout: int,
        forward: MemoryObjectSendStream[bytes] | None = None,
    ) -> AsyncGenerator[Launched]:
        """Start `command` and record it. On exit, whatever it left running is killed and the record is closed.

        `stdin`/`stdout`: subprocess.PIPE, DEVNULL, or a file descriptor (a pipeline's pipe). The pumps run
        in a task group that lives only inside this block: a held launch never outlives its task, so anything
        longer-lived holds `spawn` instead."""
        async with self.spawn(command, stdin=stdin, stdout=stdout) as spawned:
            process = spawned.process
            out = _Tee(spawned.record / "stdout", forward) if stdout == subprocess.PIPE else None
            err = _Tee(spawned.record / "stderr")
            launched = Launched(spawned.id, spawned.record, command, process, out, err, started=spawned.started)
            try:
                async with anyio.create_task_group() as tg:
                    if out is not None and process.stdout is not None:
                        tg.start_soon(out.pump, process.stdout)
                    if process.stderr is not None:
                        tg.start_soon(err.pump, process.stderr)
                    try:
                        yield launched
                    finally:
                        with anyio.CancelScope(shield=True):
                            if self._cancel is not None and self._cancel.cancel_called:
                                self._at_timeout += await anyio.to_thread.run_sync(launched.running)
                            # Whatever the command left running dies with it; its pipes close, and the pumps end.
                            await spawned.close()
            finally:
                launched.ended = time.monotonic()
