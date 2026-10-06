"""A shell held open: commands run in one bash, so `cd` and the environment persist between calls.

A `ShellSession` is the framed-subprocess backend of a future family: whatever later backends run (xonsh
over pipes, a subinterpreter) implements the same three calls — `run`, `restart`, `aclose` — and returns
`ShellResult`. The protocol is the contract; bash is the first speaker of it.

Framing: every command is wrapped so the shell reports its own exit code on a line no command output can
forge, `{ <command>; } < /dev/null 2>&1; printf '<mark>:$?\\n'`. The mark is random per command. Streams
merge, so what the agent reads is ordered as the shell wrote it, with no concurrent drain to deadlock;
a command's stdin is empty, so anything interactive meets EOF and ends. A syntactically broken command
leaves the shell mid-continuation, eating the mark: the timeout then kills the shell, and the next run
starts fresh. That is the fail-safe, not a special case.

Lifetime: the session holds a `spawn` handle — file descriptors, paths and ids, no scope. `launch` would
park its pump group on the entering task's scope stack, and any scope opened after the spawn and closed
before the close (a turn timeout, a test guard) would break exiting it. The record files are written by
the session itself: stdout as it is read, stderr drained once the shell is dead.
"""

from __future__ import annotations

import math
import secrets
import shutil
import subprocess
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio

from pyrun.command import cmd
from pyrun.scope import Scope

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from pyrun.scope import Spawned

SHELL_ARGV: tuple[str, ...] = ("bash", "--noprofile", "--norc")
"""No rc files: the shell starts the same way every time, whatever the machine carries."""


@dataclass(frozen=True)
class ShellResult:
    """One command's run: what it wrote, how it ended, and whether the shell survived it."""

    command: str
    output: str
    """stdout and stderr merged, in the order the shell wrote them, without the framing line."""
    code: int | None
    """The exit code; None when the shell died mid-command and there is none to report."""
    duration: float
    reset: bool = False
    """The shell respawned around this run: `cd` and the environment are fresh."""
    timed_out: bool = False


class ShellSession:
    """One bash, held open across calls. Commands run one at a time; concurrent `run` calls wait."""

    def __init__(
        self,
        *,
        cwd: str | os.PathLike[str],
        store: str | os.PathLike[str],
        argv: Sequence[str] = SHELL_ARGV,
        id: str | None = None,
    ) -> None:
        if shutil.which(argv[0]) is None:
            raise FileNotFoundError(f"{argv[0]} is not on PATH")
        # Unentered, on purpose: entry binds the scope to one task, and this shell spans turns. `spawn`
        # needs none of what entry provides.
        self._scope = Scope(cwd=cwd, store=store, id=id or f"shell-{secrets.token_hex(3)}")
        self._argv = tuple(argv)
        self._stack = AsyncExitStack()
        self._spawned: Spawned | None = None
        self._buffer = b""
        self._lock = anyio.Lock()

    @property
    def id(self) -> str:
        """The session's id: its records live under `<store>/<id>/`, one numbered directory per incarnation."""

        return self._scope.id

    @property
    def pid(self) -> int | None:
        """The shell's pid, while it lives."""
        return self._spawned.process.pid if self._spawned is not None else None

    async def run(self, command: str, time_limit: float | None = None) -> ShellResult:
        """Run `command` in the shell and report it. An empty command runs nothing and reports success.

        A timeout, or a cancelled turn, kills the shell: framing past that point is untrustworthy, so the
        next run starts fresh and says so."""
        async with self._lock:
            if not command.strip():
                return ShellResult(command=command, output="", code=0, duration=0.0)
            if self._spawned is None:
                await self._spawn()
                reset = True
            else:
                reset = False
            started = anyio.current_time()
            raw = bytearray()
            code: int | None = None
            output: str | None = None
            try:
                with anyio.move_on_after(time_limit if time_limit is not None else math.inf) as timer:
                    output, code = await self._exchange(command, raw)
            except anyio.BrokenResourceError, anyio.ClosedResourceError:
                pass  # The shell died under the write; the read below reports it as such.
            except anyio.get_cancelled_exc_class():
                # An outer cancel only: move_on_after swallows its own deadline, and reports it below.
                await self._kill()
                raise
            raw += self._buffer
            self._buffer = b""
            if output is None:
                output = raw.decode(errors="replace")
            await self._append_record(raw)
            elapsed = anyio.current_time() - started
            if timer.cancelled_caught or code is None:
                if timer.cancelled_caught:
                    result = ShellResult(
                        command=command, output=output, code=None, duration=elapsed, reset=True, timed_out=True
                    )
                else:
                    result = ShellResult(command=command, output=output, code=None, duration=elapsed, reset=True)
                return await self._killed(result)
            return ShellResult(command=command, output=output, code=code, duration=elapsed, reset=reset)

    async def restart(self) -> None:
        """Kill the shell and start a fresh one now: `cd` and the environment reset."""
        async with self._lock:
            await self._kill()
            await self._spawn()

    async def aclose(self) -> None:
        """Kill the shell if it lives, and close its record. The worker calls this at session end."""
        async with self._lock:
            await self._kill()

    async def _spawn(self) -> None:
        stack = AsyncExitStack()
        spawned = await stack.enter_async_context(
            self._scope.spawn(cmd(*self._argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        )
        self._stack = stack
        self._spawned = spawned
        self._buffer = b""

    async def _kill(self) -> bytes:
        """Close the held spawn — whatever the shell left running dies — and drain its stderr. Returns what
        the shell itself wrote there: syntax errors and the like, which never reach the merged stream."""
        spawned = self._spawned
        try:
            await self._stack.aclose()
        finally:
            self._spawned = None
            self._buffer = b""
        if spawned is None or spawned.process.stderr is None:
            return b""
        err = bytearray()
        try:
            async for chunk in spawned.process.stderr:
                err += chunk
        except anyio.EndOfStream, anyio.ClosedResourceError, anyio.BrokenResourceError:
            pass
        if err:
            async with await anyio.open_file(spawned.record / "stderr", "ab") as f:
                await f.write(err)
        return bytes(err)

    async def _killed(self, result: ShellResult) -> ShellResult:
        """The shell is gone: kill it, and attach what it wrote to stderr, if anything."""
        err = await self._kill()
        if not err:
            return result
        trailer = f"\n[the shell also wrote to stderr: {err.decode(errors='replace').strip()}]"
        return ShellResult(
            command=result.command,
            output=result.output + trailer,
            code=result.code,
            duration=result.duration,
            reset=result.reset,
            timed_out=result.timed_out,
        )

    async def _append_record(self, raw: bytearray) -> None:
        """What the shell wrote, framing included, onto its incarnation's stdout record."""
        if self._spawned is not None and raw:
            async with await anyio.open_file(self._spawned.record / "stdout", "ab") as f:
                await f.write(raw)

    async def _exchange(self, command: str, raw: bytearray) -> tuple[str, int | None]:
        """Write the wrapped command and read until its mark. Returns the output without the framing line,
        and the exit code — or None, with what came before it, when the shell died first."""
        spawned = self._spawned
        assert spawned is not None  # run spawns before exchanging
        assert spawned.process.stdin is not None and spawned.process.stdout is not None
        mark = f"__AID_{secrets.token_hex(8)}"
        wrapped = f"{{ {command}; }} < /dev/null 2>&1; printf '%s:%s\\n' {mark} \"$?\"\n"
        await spawned.process.stdin.send(wrapped.encode())
        while (line := await self._readline(spawned)) is not None:
            raw += line
            if mark.encode() in line:
                text, _, tail = raw.decode(errors="replace").rpartition(mark)
                try:
                    return text, int(tail.strip().lstrip(":"))
                except ValueError:
                    pass
        return raw.decode(errors="replace"), None

    async def _readline(self, spawned: Spawned) -> bytes | None:
        """The next stdout line with its newline; None once the shell has ended. Bytes: the record keeps
        what the shell wrote, decoding only for the agent."""
        stdout = spawned.process.stdout
        assert stdout is not None
        while b"\n" not in self._buffer:
            try:
                self._buffer += await stdout.receive()
            except anyio.EndOfStream:
                if not self._buffer:
                    return None
                line, self._buffer = self._buffer, b""
                return line
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line + b"\n"
