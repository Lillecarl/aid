"""Run a script in a child process (`pyrun.script`), answer its policy questions, and report what it did."""

from __future__ import annotations

import json
import secrets
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio

from pyrun.command import Command, cmd
from pyrun.result import Failed, TimedOut
from pyrun.scope import Scope, default_store
from pyrun.script import ERROR_FILE, PRINTED_FILE, SCRIPT_FILE

if TYPE_CHECKING:
    import os

    from pyrun.command import Process
    from pyrun.scope import Policy

HOST_GRACE: Final = 10.0
"""Seconds past the script's own timeout before the host kills the script's process."""
HEAD_LINES: Final = 5
TAIL_LINES: Final = 15
PRINTED_LIMIT: Final = 200
"""Lines of the script's printed output a report shows, from the end."""


@dataclass(frozen=True)
class Ran:
    """One process the script started, from its record."""

    id: str
    argv: list[str]
    code: int | None
    signal: int | None
    duration: float | None
    stdout: Path
    stderr: Path


@dataclass(frozen=True)
class Report:
    scope: str
    directory: Path
    processes: list[Ran]
    printed: str
    error: str | None
    """The script's traceback, or why it did not finish."""

    @property
    def ok(self) -> bool:
        return self.error is None

    def render(self) -> str:
        """What an agent reads; the shape is in README.md."""
        lines: list[str] = []
        for ran in self.processes:
            outcome = (
                "running"
                if ran.code is None and ran.signal is None
                else f"exit {ran.code}"
                if ran.code is not None
                else f"signal {ran.signal}"
            )
            took = f"{ran.duration:.2f}s" if ran.duration is not None else ""
            lines.append(f"$ {shlex.join(ran.argv)}  ({outcome}{f', {took}' if took else ''}, {ran.id})")
            lines += _clipped(ran.stdout, ran.id, "stdout")
            lines += _clipped(ran.stderr, ran.id, "stderr", label="stderr: ")
        if self.printed.strip():
            printed = self.printed.rstrip("\n").splitlines()
            lines.append("printed:")
            if len(printed) > PRINTED_LIMIT:
                lines.append(f"  … {len(printed) - PRINTED_LIMIT} earlier lines in {self.directory / PRINTED_FILE}")
            lines += [f"  {line}" for line in printed[-PRINTED_LIMIT:]]
        if self.error:
            lines.append("error:")
            lines += [f"  {line}" for line in self.error.rstrip("\n").splitlines()]
        return "\n".join(lines) if lines else "(the script started nothing and printed nothing)"


def _clipped(path: Path, id: str, stream: str, *, label: str = "") -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return []
    lines = text.rstrip("\n").splitlines()
    if not lines:
        return []
    if len(lines) <= HEAD_LINES + TAIL_LINES:
        return [f"  {label}{line}" for line in lines]
    hidden = len(lines) - HEAD_LINES - TAIL_LINES
    marker = f"  … {hidden} more lines of {stream} ({len(text.encode())} bytes in all): read {id}/{stream}"
    return [
        *(f"  {label}{line}" for line in lines[:HEAD_LINES]),
        marker,
        *(f"  {label}{line}" for line in lines[-TAIL_LINES:]),
    ]


def read_report(directory: Path, error: str | None = None) -> Report:
    processes: list[Ran] = []
    numbered = sorted((d for d in directory.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    for record in numbered:
        meta: dict[str, Any] = json.loads((record / "command.json").read_text())
        ended, started = meta.get("ended"), meta.get("started")
        processes.append(
            Ran(
                id=meta["id"],
                argv=meta["argv"],
                code=meta.get("code"),
                signal=meta.get("signal"),
                duration=ended - started if ended is not None and started is not None else None,
                stdout=record / "stdout",
                stderr=record / "stderr",
            )
        )
    printed = (directory / PRINTED_FILE).read_text() if (directory / PRINTED_FILE).exists() else ""
    if error is None and (directory / ERROR_FILE).exists():
        error = (directory / ERROR_FILE).read_text()
    return Report(directory.name, directory, processes, printed, error)


async def run_script(
    source: str,
    *,
    policy: Policy | None = None,
    cwd: str | os.PathLike[str] = ".",
    store: str | os.PathLike[str] | None = None,
    time_limit: float | None = None,
    python: str = sys.executable,
) -> Report:
    """Run `source` with `python -m pyrun.script` and report on it. Everything it starts dies with it.

    `time_limit`: the script's own scope timeout; the report says what was running when it struck."""
    store = Path(store) if store is not None else default_store()
    scope = secrets.token_hex(3)
    directory = store / scope
    await anyio.Path(directory).mkdir(parents=True)
    await anyio.Path(directory / SCRIPT_FILE).write_text(source)
    cwd = await anyio.Path(cwd).resolve()
    argv = [python, "-m", "pyrun.script", "--store", str(store), "--scope", scope, "--cwd", str(cwd)]
    if time_limit is not None:
        argv += ["--timeout", str(time_limit)]
    if policy is not None:
        argv.append("--policy")
    runner = cmd(*argv, timeout=time_limit + HOST_GRACE if time_limit is not None else None, check=False)
    error: str | None = None
    try:
        # The host's own scope records the script's process under the script's directory, and finds whatever
        # the script started even if the script's process dies first: the marks chain.
        async with Scope(store=directory / "host", cwd=cwd), runner.start() as process:
            if policy is not None:
                await _answer(process, policy)
            finished = await process.wait()
        if not finished.ok and not (directory / ERROR_FILE).exists():
            error = f"the script runner exited {finished.code}:\n{finished.stderr.tail()}"
    except TimedOut as timed_out:
        error = str(timed_out)
    except Failed as failed:
        error = str(failed)
    return read_report(directory, error)


async def _answer(process: Process, policy: Policy) -> None:
    async with anyio.create_task_group() as tg:
        async for line in process:
            question = json.loads(line)
            if question.get("done"):
                break
            tg.start_soon(_decide, process, policy, question)
    await process.close_stdin()


async def _decide(process: Process, policy: Policy, question: dict[str, Any]) -> None:
    command = Command(argv=tuple(question["argv"]), cwd=question.get("cwd"), env=question.get("env") or {})
    reason: str | None = None
    try:
        allowed = await policy(command)
    except Exception as error:
        allowed, reason = False, f"{type(error).__name__}: {error}"
    answer = {"id": question["id"], "allow": allowed, "reason": reason}
    await process.write(json.dumps(answer) + "\n")
