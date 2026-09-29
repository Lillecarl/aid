"""Tools for pydantic-ai agents that work on code: read files, edit them through pyedit, run processes through pyrun.

    import aid
    from pydantic_ai import Agent

    class Coder(aid.PydanticAgent):
        \"\"\"Changes code in its session's directory.\"\"\"

        def build(self) -> Agent:
            return Agent("deepseek:deepseek-chat", toolsets=[aid.coding_tools])

Each aid session has one `Coding`: its directory, a pyedit session holding edits until `apply_edits`, a pyrun store
for what `python` runs, and the session's permission mode. The worker sets it for each turn (`CODING`). Edits run
in aid's own process through pyedit's library, so no code the agent writes runs there; `python` scripts run in a
child process, and every command they start asks the permission mode first.
"""

from __future__ import annotations

import functools
import re
import shlex
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import anyio
import anyio.to_thread
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pyedit.session import EditSession  # pyright: ignore[reportMissingTypeStubs] -- annotated, no py.typed

from aid.confine import inside
from aid.protocol import AidError, PermissionChoice, PermissionDecider, PermissionDecision, PermissionRequest, clip
from aid.spec import PermissionMode
from pyrun.host import run_script

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pyedit.syntax.nodes import NodeInfo  # pyright: ignore[reportMissingTypeStubs]

    from aid.backends.base import Emit
    from aid.backends.permissions import PermissionWaits
    from pyrun import Command

READ_LIMIT: Final = 2000
SCRIPT_TIME_LIMIT: Final = 600.0
YES, ALWAYS, NO = "yes", "always", "no"
# A pyrun record, as a report names it: `<scope>.<n>/stdout`, or a scope's own file, `<scope>/printed`.
_RECORD = re.compile(r"^(?P<scope>[0-9a-f]{6})(?:\.(?P<n>\d+))?/(?P<file>[\w.]+)$")


@dataclass
class Coding:
    cwd: Path
    store: Path
    """Where `python` scripts are recorded: the session's `runs` directory."""
    mode: PermissionMode
    timeout: float
    waits: PermissionWaits
    edits: EditSession = field(init=False)
    emit: Emit | None = None
    """The running turn's; the worker sets it before each turn."""
    always: set[str] = field(default_factory=set[str])
    """Programs a person allowed for the rest of the session."""
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

    def resolve(self, path: str) -> Path:
        """A path the agent names: under the session's directory, or a pyrun record in the store."""
        if (record := _RECORD.match(path)) is not None:
            scope = self.store / record["scope"]
            return inside(scope / record["n"] if record["n"] else scope, record["file"])
        if Path(path).is_absolute() and Path(path).resolve().is_relative_to(self.store.resolve()):
            return Path(path).resolve()
        return inside(self.cwd, path)


CODING: ContextVar[Coding] = ContextVar("aid_coding")


def _coding() -> Coding:
    try:
        return CODING.get()
    except LookupError:
        raise AidError("no_coding", "aid's coding tools work only in an aid pydantic-ai session") from None


def numbered(text: str, offset: int, limit: int) -> str:
    lines = text.splitlines()
    shown = lines[offset - 1 : offset - 1 + limit]
    body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(shown, start=offset))
    rest = len(lines) - (offset - 1 + len(shown))
    return body + (f"\n… {rest} more lines; read with offset={offset + len(shown)}" if rest > 0 else "")


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
    if target.is_dir():
        raise AidError("is_dir", f"{path} is a directory; outline reads a single file")
    nodes = cast("list[NodeInfo]", await coding.edit(lambda s: s.outline(target)))  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType] -- pyedit names the type as a string it does not import
    return "\n".join(f"{n.start_line}-{n.end_line}\t{n.kind}\t{n.name}" for n in nodes) or f"{path} defines nothing"


async def show_edits() -> str:
    """Everything staged so far, as one diff."""
    return _coding().diff() or "nothing is staged"


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


async def python(ctx: RunContext[Any], script: str, time_limit: float = SCRIPT_TIME_LIMIT) -> str:
    """Run an async Python script that starts processes through pyrun, in your working directory. Returns a report
    of every process it ran (argv, exit, time, output clipped to head and tail, a record id) and what it printed.

    Top-level `await` works; `run`, `cmd`, `sh`, `pyrun` and `Path` are in scope. There is no shell: argv only.

        r = await run("git", "status", "--short")      # r.text, r.lines, r.json(), r.code; non-zero raises Failed
        await run("grep", "x", "f", check=False)        # a non-zero exit is a result
        await run("make", cwd="sub", env={"CC": "clang"}, timeout=60, input="stdin text")
        r = await (cmd("rg", "-n", "TODO") | cmd("head", "-5"))    # pipefail
        async for line in cmd("pytest", "-x").lines(): ...          # stream
        a, b = await pyrun.all(cmd("make", "a"), cmd("make", "b"))  # together
        await sh("echo $HOME")                          # the one way to use a shell

    Each command asks the session's permission first; a refused one raises Denied. Everything a script starts
    dies when it ends or after `time_limit` seconds. Read a record's full output with read("<id>/stdout")."""
    coding = _coding()
    tool_call_id = ctx.tool_call_id or ""

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


def _retrying[**P, R](tool: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """A tool's failure goes back to the model as a retry with the reason: an exception would end the run."""

    @functools.wraps(tool)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await tool(*args, **kwargs)
        except ModelRetry:
            raise
        except Exception as error:
            reason = error.message if isinstance(error, AidError) else str(error)
            raise ModelRetry(f"{type(error).__name__}: {reason}") from error

    return wrapper


coding_tools: FunctionToolset[Any] = FunctionToolset[Any](
    [
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
        _retrying(apply_edits),
        _retrying(python),
    ]
)
