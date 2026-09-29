"""`python -m pyrun.script`: run an async Python script in a scope of its own. `pyrun.host` starts it.

The script is `<store>/<scope>/script.py`, which the host writes. It runs with top-level `await`, and `run`,
`cmd`, `sh`, `pyrun` and `Path` in scope. What it prints goes to `printed` in the scope's directory, and a
traceback, trimmed to the script's own frames, to `error`.

With `--policy`, stdin and stdout are the policy channel, one JSON object per line: this process asks
`{"id", "argv", "cwd", "env"}` before each command starts, and the host answers `{"id", "allow", "reason"}`.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import itertools
import json
import linecache
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

import anyio
import anyio.to_thread

import pyrun
from pyrun.result import Denied
from pyrun.scope import Scope

if TYPE_CHECKING:
    from pyrun.command import Command

FILENAME: Final = "<script>"
SCRIPT_FILE: Final = "script.py"
PRINTED_FILE: Final = "printed"
ERROR_FILE: Final = "error"


class _PolicyChannel:
    """The host's answers, matched to questions by id: commands started together ask together."""

    def __init__(self, ask: TextIO, answer: TextIO) -> None:
        self._ask, self._answer = ask, answer
        self._ids = itertools.count(1)
        self._waiting: dict[int, tuple[anyio.Event, list[dict[str, Any]]]] = {}
        self._lock = anyio.Lock()
        self.ended = anyio.Event()

    async def read(self) -> None:
        try:
            while line := await anyio.to_thread.run_sync(self._answer.readline, abandon_on_cancel=True):
                reply = json.loads(line)
                if (waiting := self._waiting.get(reply["id"])) is not None:
                    waiting[1].append(reply)
                    waiting[0].set()
        finally:
            self.ended.set()

    async def __call__(self, command: Command) -> bool:
        id = next(self._ids)
        event, box = anyio.Event(), list[dict[str, Any]]()
        self._waiting[id] = (event, box)
        question = {"id": id, "argv": list(command.argv), "cwd": command.cwd, "env": dict(command.env)}
        try:
            async with self._lock:
                await anyio.to_thread.run_sync(self._write, json.dumps(question))
            await event.wait()
        finally:
            del self._waiting[id]
        if not box[0]["allow"]:
            raise Denied(command, box[0].get("reason"))
        return True

    async def done(self) -> None:
        """No more questions: the host closes our stdin, which ends `read`. Without it, exit would wait on the
        thread blocked reading stdin, and the host on our stdout."""
        async with self._lock:
            await anyio.to_thread.run_sync(self._write, json.dumps({"done": True}))

    def _write(self, line: str) -> None:
        self._ask.write(line + "\n")
        self._ask.flush()


def script_frames(error: BaseException) -> str:
    """The traceback with pyrun's own frames left out: the script's lines and the error are what a reader needs."""
    lines: list[str] = []
    for part in traceback.TracebackException.from_exception(error).format():
        if part.startswith("  File ") and f'"{FILENAME}"' not in part:
            continue
        lines.append(part)
    return "".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    directory = Path(args.store) / args.scope
    source = (directory / SCRIPT_FILE).read_text()
    linecache.cache[FILENAME] = (len(source), None, source.splitlines(keepends=True), FILENAME)
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "run": pyrun.run,
        "cmd": pyrun.cmd,
        "sh": pyrun.sh,
        "pyrun": pyrun,
        "Path": Path,
    }
    # The policy channel is this process's real stdin and stdout; what the script prints goes to a file.
    ask = os.fdopen(os.dup(1), "w")
    answer = os.fdopen(os.dup(0), "r")
    policy = _PolicyChannel(ask, answer) if args.policy else None
    failure: Exception | None = None
    with (directory / PRINTED_FILE).open("w", buffering=1) as printed:
        async with anyio.create_task_group() as tg:
            if policy is not None:
                tg.start_soon(policy.read)
            try:
                with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                    code = compile(source, FILENAME, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    async with Scope(
                        id=args.scope, store=args.store, cwd=args.cwd, timeout=args.timeout, policy=policy
                    ):
                        # With top-level await, eval gives a coroutine; without, it has already run.
                        outcome = eval(code, namespace)
                        if inspect.iscoroutine(outcome):
                            await outcome
            except Exception as error:
                failure = error
            finally:
                if policy is not None:
                    await policy.done()
                    await policy.ended.wait()
                tg.cancel_scope.cancel()
    if failure is not None:
        (directory / ERROR_FILE).write_text(script_frames(failure))
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pyrun.script")
    parser.add_argument("--store", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--policy", action="store_true", help="ask the host on stdin/stdout before each command")
    sys.exit(anyio.run(main_async, parser.parse_args()))


if __name__ == "__main__":
    main()
