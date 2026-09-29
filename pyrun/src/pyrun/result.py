"""What a finished process leaves: its Result, and the errors that carry one."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from pyrun.command import Command

TAIL_LINES = 20


@dataclass(frozen=True)
class Output:
    """One stream of a process. `path` holds all of it; `head` is what memory kept, all of it unless `clipped`."""

    path: Path
    head: bytes
    size: int

    @property
    def clipped(self) -> bool:
        return self.size > len(self.head)

    @property
    def bytes(self) -> bytes:
        return self.path.read_bytes() if self.clipped else self.head

    @property
    def text(self) -> str:
        return self.bytes.decode(errors="replace")

    def tail(self, lines: int = TAIL_LINES) -> str:
        return "\n".join(self.text.splitlines()[-lines:])


@dataclass(frozen=True)
class Result:
    id: str
    """`<scope>.<n>`: the process's record in the scope's store."""
    command: Command
    code: int | None
    """The exit code; None when a signal ended the process."""
    signal: int | None
    stdout: Output
    stderr: Output
    duration: float
    record: Path

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return self.stdout.text

    @property
    def lines(self) -> list[str]:
        return self.stdout.text.splitlines()

    def json(self) -> Any:
        return json.loads(self.stdout.bytes)

    def __str__(self) -> str:
        return self.text


class PyrunError(Exception):
    pass


class Failed(PyrunError):
    """A process exited non-zero, or a signal ended it."""

    def __init__(self, result: Result) -> None:
        self.result = result
        how = f"exit {result.code}" if result.code is not None else f"signal {result.signal}"
        tail = result.stderr.tail()
        super().__init__(f"{result.command} failed ({how}, {result.id})" + (f"\n{tail}" if tail else ""))


@dataclass(frozen=True)
class Running:
    """A process still alive when a timeout struck."""

    pid: int
    argv: list[str]
    id: str | None
    """The record it belongs to, when it is one of the scope's own and not a descendant."""
    last_lines: str


class TimedOut(PyrunError):
    """A command's or a scope's timeout struck. Everything still running was killed; `running` says what it was."""

    def __init__(self, what: str, seconds: float, running: list[Running]) -> None:
        self.what, self.seconds, self.running = what, seconds, running
        lines = [f"{what} timed out after {seconds:g}s; still running then:"]
        for proc in running:
            lines.append(f"  pid {proc.pid} {' '.join(proc.argv)}" + (f" ({proc.id})" if proc.id else ""))
            lines.extend(f"    {line}" for line in proc.last_lines.splitlines())
        super().__init__("\n".join(lines))


class Denied(PyrunError):
    """The scope's policy refused to start the command."""

    def __init__(self, command: Command, reason: str | None = None) -> None:
        self.command, self.reason = command, reason
        super().__init__(f"not allowed to run {command}" + (f": {reason}" if reason else ""))
