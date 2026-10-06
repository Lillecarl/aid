"""Tools for pydantic-ai agents that work on code: read files, edit them through pyedit, run processes through pyrun.

    import aid
    from pydantic_ai import Agent

    class Coder(aid.PydanticAgent):
        \"\"\"Changes code in its session's directory.\"\"\"

        def build(self) -> Agent:
            return Agent("deepseek:deepseek-chat", toolsets=[aid.coding_tools])

Each aid session has one `Coding`: its directory, a pyedit session holding edits until `apply_edits`, a pyrun store
for what `python` runs, a persistent shell for what `shell` runs, and the session's permission mode. The worker
sets it for each turn (`CODING`). Edits run in aid's own process through pyedit's library, so no code the agent
writes runs there; `python` scripts run in a child process, and every command they start asks the permission mode
first. `shell` commands run in one bash held for the session, each asking permission as written.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import inspect
import io
import linecache
import re
import shlex
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import anyio
import anyio.to_thread
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pyedit.session import EditSession  # pyright: ignore[reportMissingTypeStubs] -- annotated, no py.typed
from shellous import ResultError, Runner, sh

import pyrun
from aid.confine import inside
from aid.protocol import AidError, PermissionChoice, PermissionDecider, PermissionDecision, PermissionRequest, clip
from aid.spec import PermissionMode
from pyrun.host import run_script
from pyrun.scope import Scope
from pyrun.script import FILENAME as SCRIPT_FILENAME
from pyrun.script import script_frames
from pyrun.shell import ShellResult, ShellSession

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pyedit.syntax.nodes import NodeInfo  # pyright: ignore[reportMissingTypeStubs]

    from aid.backends.base import Emit
    from aid.backends.permissions import PermissionWaits
    from pyrun import Command

READ_LIMIT: Final = 2000
SCRIPT_TIME_LIMIT: Final = 600.0
SHELL_TIME_LIMIT: Final = 600.0
OUTPUTS_DIR: Final = "outputs"
SPILL_LIMIT: Final = 32_000
"""Characters of a tool result sent to the model; the rest lives in a content-addressed file under the
session's outputs directory, named by the result. The head plus the path is stable across turns, so the
prefix cache holds, and nothing is lost: the model reads the file with `read`."""
YES, ALWAYS, NO = "yes", "always", "no"
BACKGROUND_DIR: Final = "background"
TASK_POLL: Final = 0.2
"""Seconds between polls while task_output waits."""
TASK_TAIL: Final = 30
"""Output lines a running task's report carries; finished tasks report all of it (spilled past the limit)."""
# A pyrun record, as a report names it: `<scope>.<n>/stdout`, or a scope's own file, `<scope>/printed`.
_RECORD = re.compile(r"^(?P<scope>[0-9a-f]{6})(?:\.(?P<n>\d+))?/(?P<file>[\w.]+)$")


@dataclass
class BackgroundTask:
    """A process the session started and did not wait for. Its output files live under the session's store,
    so `read` reaches them; the runner stays open until the task is reaped, and reaping records how it ended."""

    id: str
    argv: list[str]
    runner: Runner
    stdout_file: Path
    stderr_file: Path
    time_limit: float | None
    deadline: float | None
    """When the time limit runs out; shellous's own timeout stays off, since it cancels the task that
    started the runner, which long outlives a background start."""
    started: float
    finished: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    stopped: bool = False
    reaped: bool = False


@dataclass
class Coding:
    cwd: Path
    store: Path
    """Where `python` scripts are recorded: the session's `runs` directory."""
    outputs: Path
    """Where oversize tool results spill: content-addressed files the model reads back with `read`."""
    mode: PermissionMode
    timeout: float
    waits: PermissionWaits
    autoselect_after: float = 240.0
    """Seconds an `ask_user` question with a recommended option waits for a person before it picks it."""
    edits: EditSession = field(init=False)
    emit: Emit | None = None
    """The running turn's; the worker sets it before each turn."""
    always: set[str] = field(default_factory=set[str])
    """Programs a person allowed for the rest of the session."""
    tasks: dict[str, BackgroundTask] = field(default_factory=dict[str, BackgroundTask])
    """Background tasks started this session, by id. They outlive their turn, and die with the worker."""
    task_seq: int = 0
    shell: ShellSession | None = None
    """The session's shell, started by the first `shell` call. `cd` and the environment persist between
    calls; it dies with the worker, or sooner on a timeout, which the next call reports as a reset."""
    compact: Callable[[str], Awaitable[str]] | None = None
    """Summarize the conversation into a digest; the worker sets it per session, and `compact` calls it."""
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    def __post_init__(self) -> None:
        self.edits = EditSession(root=self.cwd)

    async def edit[T](self, work: Callable[[EditSession], T]) -> T:
        """pyedit is synchronous, and rope can be slow: one call at a time, off the event loop."""
        async with self._lock:
            return await anyio.to_thread.run_sync(work, self.edits)

    def diff(self, path: Path | None = None) -> str:
        """The staged diff, of one file (relative to the directory) or of all."""
        # pyedit names files by absolute path; the agent knows them by the session's directory.
        root = self.cwd.resolve()
        text = self.edits.diff_git().replace(f"a/{root}/", "a/").replace(f"b/{root}/", "b/")
        if path is None:
            return text
        sections = re.split(r"(?m)^(?=--- )", text)
        return "".join(s for s in sections if f"a/{path}\n" in s.split("\n+++", 1)[0] or f"+++ b/{path}\n" in s)

    async def permit(
        self,
        tool_call_id: str,
        *,
        tool_name: str,
        title: str,
        input: Any,
        remember: str | None = None,
    ) -> bool:
        """The session's answer. In ask mode, a person's: `remember` offers "always" for that key."""
        if self.mode is not PermissionMode.ASK:
            return self.mode is PermissionMode.ALLOW
        if remember is not None and remember in self.always:
            return True
        options = [PermissionChoice(option_id=YES, name="Yes", kind="allow_once")]
        if remember is not None:
            options.append(PermissionChoice(option_id=ALWAYS, name=f"Always allow {remember}", kind="allow_always"))
        options.append(PermissionChoice(option_id=NO, name="No", kind="reject_once"))
        request = PermissionRequest(
            request_id=uuid.uuid4().hex,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            title=title,
            input=input,
            options=options,
        )
        if self.emit is None:
            return False
        await self.emit(request)
        decision = PermissionDecision(request_id=request.request_id, option_id=NO, by=PermissionDecider.TIMEOUT)
        with anyio.move_on_after(self.timeout):
            decision = await self.waits.wait(request)
        await self.emit(decision)
        option_id = decision.option_id
        if option_id == ALWAYS and remember is not None:
            self.always.add(remember)
        return option_id in (YES, ALWAYS)

    async def close(self) -> None:
        """Stop every running background task and the shell, reaping each. The worker is going: shielded, so
        a shutdown cancelling the close still leaves no process behind."""
        with anyio.CancelScope(shield=True):
            async with anyio.create_task_group() as tg:
                for task in self.tasks.values():
                    if not task.reaped:
                        if task.runner.returncode is None:
                            task.runner.cancel()
                        tg.start_soon(_reap, task)
                if self.shell is not None:
                    tg.start_soon(self.shell.aclose)
                    self.shell = None

    async def shell_run(self, command: str, time_limit: float) -> str:
        """Run `command` in the session's shell, starting it on first use, and report what it did."""
        first = self.shell is None
        if first:
            self.shell = ShellSession(cwd=self.cwd, store=self.store)
        assert self.shell is not None
        return _render_shell(await self.shell.run(command, time_limit=time_limit), time_limit, first)

    async def shell_restart(self) -> str:
        """Kill the session's shell now; the next `shell` call starts fresh. `cd` and env reset."""
        if self.shell is None:
            return "the shell never started; nothing to restart"
        await self.shell.restart()
        return "shell restarted; cd and env reset"

    def resolve(self, path: str) -> Path:
        """A path the agent names: under the session's directory, a pyrun record in the store, or a spilled
        tool result in the outputs directory."""
        if (record := _RECORD.match(path)) is not None:
            scope = self.store / record["scope"]
            return inside(scope / record["n"] if record["n"] else scope, record["file"])
        if Path(path).is_absolute():
            target = Path(path).resolve()
            if target.is_relative_to(self.store.resolve()) or target.is_relative_to(self.outputs.resolve()):
                return target
        return inside(self.cwd, path)


CODING: ContextVar[Coding] = ContextVar("aid_coding")


def _coding() -> Coding:
    try:
        return CODING.get()
    except LookupError:
        raise AidError("no_coding", "aid's coding tools work only in an aid pydantic-ai session") from None


def _task(task_id: str) -> BackgroundTask:
    try:
        return _coding().tasks[task_id]
    except KeyError:
        raise AidError("unknown_task", f"no background task {task_id!r}; tasks lists them") from None


async def _read(path: Path) -> str:
    try:
        return await anyio.Path(path).read_text()
    except OSError:
        return ""


async def _tail(path: Path, lines: int = TASK_TAIL) -> str:
    return "\n".join((await _read(path)).splitlines()[-lines:])


async def _reap(task: BackgroundTask) -> None:
    """Reap a finished task and record how it ended. Shielded: a cancelled turn must not orphan the
    bookkeeping while the process itself is already gone."""
    if task.reaped:
        return
    task.reaped = True
    with anyio.CancelScope(shield=True):
        try:
            await task.runner.__aexit__(None, None, None)
        except TimeoutError:
            task.timed_out = True
    task.finished = anyio.current_time()
    task.exit_code = task.runner.returncode


async def _enforce(task: BackgroundTask) -> None:
    """Kill a task past its deadline and reap it as timed out. The check runs wherever the agent touches
    the task, so an over-limit task is already dead by the time its output is read."""
    if task.runner.returncode is None and task.deadline is not None and anyio.current_time() >= task.deadline:
        task.runner.cancel()
        await _reap(task)
        task.timed_out = True


async def _report(task: BackgroundTask) -> str:
    """What the agent reads: running tasks a tail of their output so far, finished ones how they ended and
    all of their output (spilled past the limit by the tool wrapper)."""
    name = shlex.join(task.argv)
    if task.runner.returncode is None:
        tail = await _tail(task.stdout_file)
        return f"{task.id} is running (pid {task.runner.pid}): {name}\n{tail or '(no output yet)'}"
    elapsed = (task.finished or task.started) - task.started
    out = await _read(task.stdout_file)
    err = await _read(task.stderr_file)
    if task.timed_out:
        head = f"{task.id} timed out after {task.time_limit}s and was killed: {name}"
    elif task.stopped:
        head = f"{task.id} stopped (exit {task.exit_code}): {name}"
    else:
        head = f"{task.id} finished: exit {task.exit_code} after {elapsed:.1f}s: {name}"
    body = out or "(no output)"
    if err:
        body += f"\nstderr:\n{err}"
    return f"{head}\n{body}"


def numbered(text: str, offset: int, limit: int) -> str:
    lines = text.splitlines()
    shown = lines[offset - 1 : offset - 1 + limit]
    body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(shown, start=offset))
    rest = len(lines) - (offset - 1 + len(shown))
    return body + (f"\n… {rest} more lines; read with offset={offset + len(shown)}" if rest > 0 else "")


async def spill(text: str) -> str:
    """An oversize tool result as a stable head plus the file holding all of it. Content-addressed, so the
    same result always names the same path and the history never shifts under the prefix cache."""
    if len(text) <= SPILL_LIMIT:
        return text
    coding = _coding()
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    path = coding.outputs / f"{digest}.txt"
    await anyio.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not await anyio.Path(path).exists():
        await anyio.Path(path).write_text(text)
    return f"{text[:SPILL_LIMIT]}\n… [full output in {path}; read it with read]"


async def ls(path: str = ".") -> str:
    """List a directory's entries, directories first with a trailing `/`: find your way before you read."""
    coding = _coding()
    target = coding.resolve(path)
    if not await anyio.Path(target).exists():
        raise AidError("no_such_file", f"{path} does not exist")
    if not await anyio.Path(target).is_dir():
        raise AidError("not_dir", f"{path} is a file; read it with read")
    entries = [(entry.name, await entry.is_dir()) async for entry in anyio.Path(target).iterdir()]
    entries.sort(key=lambda entry: (not entry[1], entry[0]))
    if not entries:
        return f"{path} is empty"
    return "\n".join(name + "/" if is_dir else name for name, is_dir in entries)


async def read(path: str, offset: int = 1, limit: int = READ_LIMIT) -> str:
    """Read a text file with line numbers, as it will be once your staged edits apply.

    `path` is relative to your working directory. A `python` report's records read the same way: `3fa9c1.2/stdout`,
    `3fa9c1.2/stderr`, `3fa9c1/printed`. `offset` is the first line (1-based), `limit` how many."""
    coding = _coding()
    target = coding.resolve(path)
    if target.is_relative_to(coding.cwd.resolve()):
        content = await coding.edit(lambda s: s.read(target))
    else:
        content = await anyio.Path(target).read_bytes()
    if isinstance(content, bytes):
        try:
            content = content.decode()
        except UnicodeDecodeError:
            return f"{path} is binary, {len(content)} bytes"
    return numbered(content, max(offset, 1), limit) or f"{path} is empty"


async def edit(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace `old` with `new` in a file, staged: nothing is written until apply_edits.

    `old` must match the file exactly, and occur once unless `replace_all`. Returns the file's staged diff."""
    coding = _coding()
    target = coding.resolve(path)

    def replace(session: EditSession) -> int:
        current = session.read(target)
        if isinstance(current, bytes):
            raise ValueError(f"{path} is binary")
        found = current.count(old)
        if found == 0:
            raise ValueError(f"{old!r} is not in {path}")
        if found > 1 and not replace_all:
            raise ValueError(f"{old!r} occurs {found} times in {path}: give more context, or replace_all=True")
        return session.edit(target, old, new)

    await coding.edit(replace)
    return coding.diff(target.relative_to(coding.cwd.resolve()))


async def write(path: str, content: str) -> str:
    """Create or replace a whole file, staged. Returns its staged diff."""
    coding = _coding()
    target = coding.resolve(path)
    await coding.edit(lambda s: s.write(target, content))
    return coding.diff(target.relative_to(coding.cwd.resolve()))


async def delete(path: str) -> str:
    """Delete a file, staged."""
    coding = _coding()
    target = coding.resolve(path)
    await coding.edit(lambda s: s.delete(target))
    return f"{path} will be deleted"


async def rename(old: str, new: str) -> str:
    """Move a file or a directory tree, staged."""
    coding = _coding()
    source, target = coding.resolve(old), coding.resolve(new)
    await coding.edit(lambda s: s.rename(source, target))
    return f"{old} will move to {new}"


async def rename_symbol(
    path: str, old_name: str, new_name: str, line: int | None = None, column: int | None = None
) -> str:
    """Rename a Python symbol everywhere it is used, imports included, staged. Without `line`/`column` the
    definition is found by `old_name`; an ambiguous name lists the candidates. Returns the staged diff."""
    coding = _coding()
    target = coding.resolve(path)
    await coding.edit(lambda s: s.rename_symbol(target, line=line, column=column, old_name=old_name, new_name=new_name))
    return coding.diff()


async def apply_patch(patch: str) -> str:
    """Stage a patch in OpenAI's apply_patch (V4A) format: `*** Begin Patch` … `*** End Patch`, with `*** Update
    File:`, `*** Add File:`, `*** Delete File:` and `*** Move to:` sections. Returns the staged diff."""
    coding = _coding()

    def stage(session: EditSession) -> None:
        session.apply_v4a(patch)  # pyright: ignore[reportUnknownMemberType] -- returns its operations, unused

    await coding.edit(stage)
    return coding.diff()


async def outline(path: str) -> str:
    """Every named definition in a file, with its lines: find your way in a file too big to read whole."""
    coding = _coding()
    target = coding.resolve(path)
    if await anyio.Path(target).is_dir():
        raise AidError("is_dir", f"{path} is a directory; outline reads a single file")
    nodes = cast("list[NodeInfo]", await coding.edit(lambda s: s.outline(target)))  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType] -- pyedit names the type as a string it does not import
    return "\n".join(f"{n.start_line}-{n.end_line}\t{n.kind}\t{n.name}" for n in nodes) or f"{path} defines nothing"


async def show_edits() -> str:
    """Everything staged so far, as one diff."""
    return _coding().diff() or "nothing is staged"


async def compact(instructions: str) -> str:
    """Summarize the conversation so far into a digest focused by `instructions`, and replace older history
    with it: the digest plus recent turns is what later turns are sent. Call it when the context grows heavy,
    naming what the upcoming work needs. Returns the digest."""
    coding = _coding()
    if coding.compact is None:
        raise AidError("no_compact", "compaction needs an aid pydantic-ai session")
    return await coding.compact(instructions)


async def discard_edits() -> str:
    """Drop everything staged."""
    coding = _coding()
    coding.edits = EditSession(root=coding.cwd)
    return "staged edits dropped"


async def apply_edits(ctx: RunContext[Any]) -> str:
    """Write everything staged to disk. The session's permission mode decides; a person may be asked."""
    coding = _coding()
    # pyedit caches every file read among the staged ones; only the changed ones are edits.
    await coding.edit(lambda s: s.prune_unchanged())
    diff = coding.diff()
    if not diff:
        return "nothing is staged"
    files = sorted({coding.edits.relpath(p) for p in coding.edits.staged()})
    allowed = await coding.permit(
        ctx.tool_call_id or "",
        tool_name="apply_edits",
        title=f"write {len(files)} file{'s' if len(files) != 1 else ''}: {', '.join(files)}",
        input={"diff": clip(diff)},
    )
    if not allowed:
        return "not allowed: the edits stay staged"
    await coding.edit(lambda s: s.apply())  # pyright: ignore[reportUnknownMemberType] -- `paths` is unannotated
    coding.edits = EditSession(root=coding.cwd)
    return f"wrote {', '.join(files)}"


async def ask_user(
    ctx: RunContext[Any], question: str, options: list[str] | None = None, recommended: str | None = None
) -> str:
    """Ask the person watching the session one question, and wait for their answer. `options` names the
    alternatives; empty asks open-ended. `recommended` names the option picked automatically when nobody
    answers in time; the answer then says it was picked for them. The answer names their pick, their own
    words, or both; when nobody answers in time with no recommendation, it says so, and you proceed with
    your best judgment."""
    coding = _coding()
    if coding.mode is PermissionMode.DENY:
        raise AidError("questions_declined", "this session declines questions; proceed with your best judgment")
    names = options or []
    if recommended is not None and recommended not in names:
        raise AidError("unknown_option", f"{recommended!r} is not one of the options; name one of them")
    if coding.mode is not PermissionMode.ASK or coding.emit is None:
        return "No one is watching; proceed with your best judgment."
    picked_id = str(names.index(recommended)) if recommended is not None else None
    request = PermissionRequest(
        request_id=uuid.uuid4().hex,
        tool_call_id=ctx.tool_call_id or "",
        tool_name="ask_user",
        title=question,
        options=[
            PermissionChoice(option_id=str(i), name=name, kind="ask_once", recommended=name == recommended)
            for i, name in enumerate(names)
        ],
    )
    await coding.emit(request)
    decision = PermissionDecision(request_id=request.request_id, option_id=None, by=PermissionDecider.TIMEOUT)
    # The recommendation fires first; a shorter session timeout refuses instead, with nobody to pick from.
    autoselect = picked_id is not None and coding.autoselect_after < coding.timeout
    with anyio.move_on_after(coding.autoselect_after if autoselect else coding.timeout):
        decision = await coding.waits.wait(request)
    if decision.by is PermissionDecider.TIMEOUT and autoselect:
        decision = PermissionDecision(request_id=request.request_id, option_id=picked_id, by=PermissionDecider.AUTO)
    await coding.emit(decision)
    picked = names[int(decision.option_id)] if decision.option_id is not None else None
    if decision.by is PermissionDecider.AUTO and picked is not None:
        return f'Nobody answered in time; proceeding with the recommended option "{picked}".'
    if picked is not None and decision.text:
        return f'The person picked "{picked}" and added: {decision.text}'
    if picked is not None:
        return f'The person picked "{picked}".'
    if decision.text:
        return f"The person answered: {decision.text}"
    return "Nobody answered; proceed with your best judgment."


async def python(
    ctx: RunContext[Any],
    script: str,
    time_limit: float = SCRIPT_TIME_LIMIT,
    mode: Literal["async", "sync", "xonsh", "in-loop"] = "async",
) -> str:
    """Run a Python script in your working directory, and report what it did.

    `async` runs an isolated async script through pyrun: top-level `await` works, and `run`, `cmd`, `sh`,
    `pyrun` and `Path` are in scope. There is no shell: argv only.

        r = await run("git", "status", "--short")      # r.text, r.lines, r.json(), r.code; non-zero raises Failed
        await run("grep", "x", "f", check=False)        # a non-zero exit is a result
        await run("make", cwd="sub", env={"CC": "clang"}, timeout=60, input="stdin text")
        r = await (cmd("rg", "-n", "TODO") | cmd("head", "-5"))    # pipefail
        async for line in cmd("pytest", "-x").lines(): ...          # stream
        a, b = await pyrun.all(cmd("make", "a"), cmd("make", "b"))  # together
        await sh("echo $HOME")                          # the one way to use a shell

    Each command asks the session's permission first; a refused one raises Denied. Everything a script starts
    dies when it ends or after `time_limit` seconds. Read a record's full output with read("<id>/stdout").

    `sync` runs an isolated plain script instead, for code with no `await` in it. `xonsh` runs the script
    as xonsh, the Python-powered shell: pipelines, globs and `!` subprocesses read shell-like, `@()` holds
    Python. `in-loop` runs the async script in the worker's own event loop: same scope as `async`, but no
    isolation. In-loop code shares the process, so it can inspect aid itself yet no gate can contain it; code
    that never awaits cannot be timed out, and then the worker needs a restart. The other modes ask per
    command; `sync`, `xonsh` and `in-loop` ask once for the whole script, which the person judges as written."""
    coding = _coding()
    tool_call_id = ctx.tool_call_id or ""
    if mode == "async":
        return await _python_async(coding, tool_call_id, script, time_limit)
    if await _permit_script(coding, tool_call_id, mode, script):
        if mode == "sync":
            return await _python_oneshot(coding, [sys.executable, "-c", script], mode, time_limit)
        if mode == "xonsh":
            return await _python_oneshot(coding, [sys.executable, "-m", "xonsh", "-c", script], mode, time_limit)
        return await _python_in_loop(coding, script, time_limit)
    return "not allowed: the script stays unrun"


async def _python_async(coding: Coding, tool_call_id: str, script: str, time_limit: float) -> str:
    async def policy(command: Command) -> bool:
        return await coding.permit(
            tool_call_id,
            tool_name="python",
            title=shlex.join(command.argv),
            input={"argv": list(command.argv), "cwd": command.cwd or "."},
            remember=Path(command.argv[0]).name,
        )

    report = await run_script(
        script, policy=policy, cwd=coding.cwd, store=coding.store, time_limit=time_limit, python=sys.executable
    )
    return report.render()


async def _permit_script(coding: Coding, tool_call_id: str, mode: str, script: str) -> bool:
    """One permit for the whole script: modes whose internals no gate can see are judged as written."""
    first = next((line for line in script.splitlines() if line.strip()), "(empty script)")
    return await coding.permit(
        tool_call_id,
        tool_name="python",
        title=f"{mode}: {first[:80]}",
        input={"mode": mode, "script": script},
    )


async def _python_oneshot(coding: Coding, argv: list[str], mode: str, time_limit: float) -> str:
    """Run argv once as a child and report exit, time and output. Awaited, so shellous's timeout is the
    same task that waits: it cancels no stranger."""
    started = anyio.current_time()
    try:
        out = await sh(argv[0], *argv[1:]).set(cwd=str(coding.cwd), timeout=time_limit)
        code: int | None = 0
    except ResultError as error:
        out, code = error.result.output, error.result.exit_code
    except TimeoutError:
        return f"{mode} timed out after {time_limit}s."
    body = str(out) or "(no output)"
    return f"{mode} finished: exit {code} after {anyio.current_time() - started:.1f}s:\n{body}"


async def _python_in_loop(coding: Coding, script: str, time_limit: float) -> str:
    """Eval the script in the worker's loop, in pyrun's namespace and a scope of its own: the scope kills
    what the script leaves running, and records it under the session's store. Prints are caught; failures
    carry the script's own frames."""
    linecache.cache[SCRIPT_FILENAME] = (len(script), None, script.splitlines(keepends=True), SCRIPT_FILENAME)
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "run": pyrun.run,
        "cmd": pyrun.cmd,
        "sh": pyrun.sh,
        "pyrun": pyrun,
        "Path": Path,
    }
    code = compile(script, SCRIPT_FILENAME, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    printed = io.StringIO()
    started = anyio.current_time()
    failure: Exception | None = None
    finished = False
    scope = uuid.uuid4().hex[:6]
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        async with Scope(id=f"loop-{scope}", store=coding.store, cwd=coding.cwd, policy=None):
            try:
                with anyio.move_on_after(time_limit):
                    outcome = eval(code, namespace)
                    if inspect.iscoroutine(outcome):
                        await outcome
                    finished = True
            except Exception as error:
                failure = error
    elapsed = anyio.current_time() - started
    body = printed.getvalue() or "(no output)"
    if failure is not None:
        return f"in-loop failed after {elapsed:.1f}s:\n{body}\nerror:\n{script_frames(failure)}"
    if not finished:
        return f"in-loop timed out after {time_limit}s:\n{body}"
    return f"in-loop finished after {elapsed:.1f}s:\n{body}"


def _render_shell(result: ShellResult, time_limit: float, first: bool) -> str:
    """A shell run as the agent reads it: how it ended, and whether the shell survived. A respawned shell
    says so up front, since the `cd` and env the agent built are gone."""
    body = result.output or "(no output)"
    reset = "the shell restarted fresh (cd and env reset), then: " if result.reset and not first else ""
    if result.timed_out:
        return f"{reset}shell timed out after {time_limit}s; the shell was killed, and the next command starts fresh:\n{body}"
    if result.code is None:
        return f"{reset}shell ended during the command; the next command starts fresh:\n{body}"
    return f"{reset}shell finished: exit {result.code} after {result.duration:.1f}s:\n{body}"


async def shell(ctx: RunContext[Any], command: str, time_limit: float = SHELL_TIME_LIMIT) -> str:
    """Run a command in the session's shell, and report what it did. The shell persists: `cd` and exported
    variables stay for later calls, across turns. It starts in your working directory, without rc files.

    Each call asks the session's permission first, judged as written; there is no remembering, since every
    command differs. The command's stdin is empty, so anything interactive meets EOF. A timeout kills the
    shell, and the next call starts fresh and says the state reset; restart it yourself with shell_restart."""
    coding = _coding()
    first_line = next((line for line in command.splitlines() if line.strip()), "(empty command)")
    allowed = await coding.permit(
        ctx.tool_call_id or "",
        tool_name="shell",
        title=first_line[:100],
        input={"command": command},
    )
    if not allowed:
        return "not allowed: the command stays unrun"
    return await coding.shell_run(command, time_limit)


async def shell_restart() -> str:
    """Restart the session's shell now: kill it and start fresh. `cd` and env reset. Runs nothing itself,
    so it asks no permission."""
    return await _coding().shell_restart()


async def background(
    ctx: RunContext[Any],
    argv: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    time_limit: float | None = None,
) -> str:
    """Start argv as a background task and return its id at once. The task outlives the turn: read its
    output with task_output, stop it with task_stop, list them with tasks. It dies with the worker, or
    sooner on its time limit, which kills it and reports the tail. argv only, no shell."""
    coding = _coding()
    if not argv:
        raise AidError("no_command", "background needs an argv, like python needs a script")
    directory = coding.cwd if cwd is None else inside(coding.cwd, cwd)
    allowed = await coding.permit(
        ctx.tool_call_id or "",
        tool_name="background",
        title=shlex.join(argv),
        input={"argv": argv, "cwd": cwd or "."},
        remember=Path(argv[0]).name,
    )
    if not allowed:
        return "not allowed: the task stays unstarted"
    coding.task_seq += 1
    task_id = f"bg{coding.task_seq}"
    task_dir = (coding.store / BACKGROUND_DIR / task_id).resolve()
    await anyio.Path(task_dir).mkdir(parents=True, exist_ok=True)
    cmd = sh(argv[0], *argv[1:]).set(cwd=str(directory))
    if env:
        cmd = cmd.env(**env)
    runner = Runner(cmd.stdout(task_dir / "stdout").stderr(task_dir / "stderr"))
    await runner.__aenter__()
    coding.tasks[task_id] = BackgroundTask(
        id=task_id,
        argv=list(argv),
        runner=runner,
        stdout_file=task_dir / "stdout",
        stderr_file=task_dir / "stderr",
        time_limit=time_limit,
        deadline=anyio.current_time() + time_limit if time_limit is not None else None,
        started=anyio.current_time(),
    )
    return f"started {task_id}: {shlex.join(argv)} (pid {runner.pid})"


async def task_output(task_id: str, wait: float | None = None) -> str:
    """A background task's report: running tasks a tail of their output so far, finished ones how they
    ended and all of their output. `wait` polls up to that many seconds for a running task to finish."""
    task = _task(task_id)
    await _enforce(task)
    if task.runner.returncode is None and wait:
        with anyio.move_on_after(wait):
            while task.runner.returncode is None:
                await _enforce(task)
                await anyio.sleep(TASK_POLL)
    if task.runner.returncode is not None and not task.reaped:
        await _reap(task)
    return await _report(task)


async def task_stop(task_id: str) -> str:
    """Stop a background task: SIGTERM, then kill. Reports how it ended. Stopping a finished task reports
    its finish instead."""
    task = _task(task_id)
    await _enforce(task)
    if task.runner.returncode is None:
        task.stopped = True
        task.runner.cancel()
        await _reap(task)
    return await _report(task)


async def tasks() -> str:
    """The session's background tasks, running and finished, with how each ended."""
    coding = _coding()
    if not coding.tasks:
        return "no background tasks"
    lines: list[str] = []
    for task in coding.tasks.values():
        if task.runner.returncode is None:
            state = f"running (pid {task.runner.pid})"
        elif task.timed_out:
            state = "timed out"
        elif task.stopped:
            state = f"stopped (exit {task.exit_code})"
        else:
            state = f"finished (exit {task.exit_code})"
        lines.append(f"{task.id}: {shlex.join(task.argv)} — {state}")
    return "\n".join(lines)


def _retrying[**P, R](tool: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """A tool's failure goes back to the model as a retry with the reason: an exception would end the run.
    Oversize text results spill to a file, so no turn carries more than a head plus a path."""

    @functools.wraps(tool)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            result = await tool(*args, **kwargs)
        except ModelRetry:
            raise
        except Exception as error:
            reason = error.message if isinstance(error, AidError) else str(error)
            raise ModelRetry(f"{type(error).__name__}: {reason}") from error
        if isinstance(result, str):
            return cast("R", await spill(result))
        return result

    return wrapper


coding_tools: FunctionToolset[Any] = FunctionToolset[Any](
    [
        _retrying(ls),
        _retrying(read),
        _retrying(edit),
        _retrying(write),
        _retrying(delete),
        _retrying(rename),
        _retrying(rename_symbol),
        _retrying(apply_patch),
        _retrying(outline),
        _retrying(show_edits),
        _retrying(discard_edits),
        _retrying(compact),
        _retrying(apply_edits),
        _retrying(ask_user),
        _retrying(python),
        _retrying(shell),
        _retrying(shell_restart),
        _retrying(background),
        _retrying(task_output),
        _retrying(task_stop),
        _retrying(tasks),
    ]
)
