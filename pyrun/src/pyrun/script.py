"""`python -m pyrun.script`: run an async Python script in a scope of its own. `pyrun.host` starts it.

The script is `<store>/<scope>/script.py`, which the host writes. It runs with top-level `await`, and `run`,
`cmd`, `sh`, `pyrun` and `Path` in scope. What it prints goes to `printed` in the scope's directory, and a
traceback, trimmed to the script's own frames, to `error`.

With `--policy`, stdin and stdout are the policy channel, one JSON object per line: this process asks
`{"id", "argv", "cwd", "env"}` before each command starts, and the host answers `{"id", "allow", "reason"}`.
With `--files`, the same channel carries `{"id", "file": {"access", "path", "event"}}` before each file access
outside `--cwd` and `--store` (`pyrun.files`); a refused one raises PermissionError in the script.
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
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

import anyio
import anyio.to_thread

import pyrun
from pyrun.files import FileGuard
from pyrun.result import Denied
from pyrun.scope import Scope

if TYPE_CHECKING:
    from pyrun.command import Command
    from pyrun.files import FileAccess

FILENAME: Final = "<script>"
SCRIPT_FILE: Final = "script.py"
PRINTED_FILE: Final = "printed"
ERROR_FILE: Final = "error"


class _PolicyChannel:
    """The host's answers, matched to questions by id: commands started together ask together. A thread of its own
    reads them, so a question from the file audit hook, which blocks the event loop while it waits, still gets its
    answer."""

    def __init__(self, ask: TextIO, answer: TextIO) -> None:
        self._ask, self._answer = ask, answer
        self._ids = itertools.count(1)
        self._waiting: dict[int, tuple[threading.Event, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self._ended = threading.Event()
        threading.Thread(target=self._read, name="pyrun-policy", daemon=True).start()

    def _read(self) -> None:
        try:
            while line := self._answer.readline():
                reply = json.loads(line)
                if (waiting := self._waiting.get(reply["id"])) is not None:
                    waiting[1].append(reply)
                    waiting[0].set()
        finally:
            self._ended.set()
            for event, _ in list(self._waiting.values()):
                event.set()  # The host is gone: no answer comes, and an unanswered question is refused.

    def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        """Blocks until the host answers; from any thread."""
        id = next(self._ids)
        event, box = threading.Event(), list[dict[str, Any]]()
        self._waiting[id] = (event, box)
        try:
            if not self._ended.is_set():
                self._write(json.dumps({"id": id, **question}))
                event.wait()
        finally:
            del self._waiting[id]
        return box[0] if box else {"allow": False, "reason": "the host stopped answering"}

    async def __call__(self, command: Command) -> bool:
        question = {"argv": list(command.argv), "cwd": command.cwd, "env": dict(command.env)}
        reply = await anyio.to_thread.run_sync(self.ask, question, abandon_on_cancel=True)
        if not reply["allow"]:
            raise Denied(command, reply.get("reason"))
        return True

    def ask_file(self, access: FileAccess) -> tuple[bool, str | None]:
        reply = self.ask({"file": {"access": access.access, "path": access.path, "event": access.event}})
        return bool(reply["allow"]), reply.get("reason")

    async def done(self) -> None:
        """No more questions: the host closes our stdin, which ends the reader. Without it, the host would wait on
        our stdout."""
        await anyio.to_thread.run_sync(self._write, json.dumps({"done": True}))
        await anyio.to_thread.run_sync(self._ended.wait)

    def _write(self, line: str) -> None:
        with self._lock:
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
    policy = _PolicyChannel(ask, answer) if args.policy or args.files else None
    failure: Exception | None = None
    with (directory / PRINTED_FILE).open("w", buffering=1) as printed:
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                code = compile(source, FILENAME, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                if args.files and policy is not None:
                    # Bytecode caches would be writes beside every module the script imports.
                    sys.dont_write_bytecode = True
                    sys.addaudithook(FileGuard([args.cwd, args.store], policy.ask_file))
                async with Scope(
                    id=args.scope,
                    store=args.store,
                    cwd=args.cwd,
                    timeout=args.timeout,
                    policy=policy if args.policy else None,
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
    parser.add_argument(
        "--files", action="store_true", help="ask the host before each file access outside --cwd and --store"
    )
    sys.exit(anyio.run(main_async, parser.parse_args()))


if __name__ == "__main__":
    main()
