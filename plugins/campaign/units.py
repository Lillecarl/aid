"""A campaign's units, pure: which files of a repository to audit, in which groups, and whether a turn read them.

Coverage comes from the session's recorded `read` calls, never from what the model says it did: a file counts as
read when the line ranges of its completed reads cover every line.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Annotated, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field

from aid.protocol import AidError, ToolCall

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

READ_LINES: Final = 2000
"""`aid.coding.READ_LIMIT`, the lines a `read` without `limit` returns: not imported, so the daemon need not load
aid.coding."""
SNIFF_BYTES: Final = 8192


class FileInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    """Relative to the repository."""
    lines: int
    size: int


class Skipped(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    reason: str


class Limits(BaseModel):
    """What a unit may hold and which files a campaign leaves out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_file_bytes: Annotated[int, Field(gt=0)] = 256 << 10
    unit_files: Annotated[int, Field(gt=0)] = 12
    unit_bytes: Annotated[int, Field(gt=0)] = 96 << 10
    exclude: list[str] = Field(default_factory=list[str])
    """Globs over the path relative to the repository (`PurePath.full_match`), such as `**/vendor/**`."""


async def tracked(repo: Path) -> list[str]:
    """The files git tracks in `repo`, as git names them."""
    result = await anyio.run_process(["git", "-C", str(repo), "ls-files", "-z"], check=False)
    if result.returncode != 0:
        raise AidError("not_a_repository", f"{repo}: {result.stderr.decode(errors='replace').strip()}")
    return [name for name in result.stdout.decode().split("\0") if name]


def _utf8(content: bytes) -> str | None:
    try:
        return content.decode()
    except UnicodeDecodeError:
        return None


def examine(repo: Path, names: Iterable[str], limits: Limits) -> tuple[list[FileInfo], list[Skipped]]:
    """The files worth a model's reading, and why each other one is left out. Blocking: run it in a thread."""
    files: list[FileInfo] = []
    skipped: list[Skipped] = []
    root = repo.resolve()
    for name in sorted(names):
        path = repo / name
        if any(PurePath(name).full_match(glob) for glob in limits.exclude):
            skipped.append(Skipped(path=name, reason="excluded"))
        elif path.is_symlink() or not path.is_file():
            # Submodules and links: a link may point out of the repository, and `read` would follow it.
            skipped.append(Skipped(path=name, reason="not a file"))
        elif not path.resolve().is_relative_to(root):
            skipped.append(Skipped(path=name, reason="outside"))
        elif (size := path.stat().st_size) > limits.max_file_bytes:
            skipped.append(Skipped(path=name, reason=f"{size} bytes"))
        elif b"\0" in (content := path.read_bytes())[:SNIFF_BYTES]:
            skipped.append(Skipped(path=name, reason="binary"))
        elif (text := _utf8(content)) is None:
            # `read` decodes strictly and calls anything else binary: such a file could never be covered.
            skipped.append(Skipped(path=name, reason="not UTF-8"))
        else:
            # As `read` numbers them: `str.splitlines`, form feeds and lone carriage returns included.
            files.append(FileInfo(path=name, lines=len(text.splitlines()), size=size))
    return files, skipped


def chunk(files: Sequence[FileInfo], limits: Limits) -> list[list[FileInfo]]:
    """Units in path order, so a directory's files stay together: a unit closes when one more file would pass a cap.
    A file over `unit_bytes` makes a unit of its own."""
    units: list[list[FileInfo]] = []
    current: list[FileInfo] = []
    size = 0
    for file in sorted(files, key=lambda f: f.path):
        if current and (len(current) == limits.unit_files or size + file.size > limits.unit_bytes):
            units.append(current)
            current, size = [], 0
        current.append(file)
        size += file.size
    if current:
        units.append(current)
    return units


def _merged(calls: Iterable[tuple[str, ToolCall]]) -> list[ToolCall]:
    """One record per call: later events fill what earlier ones left empty, as `ToolCall` says. Keyed by turn too:
    a model may number its calls afresh each turn (`call_0`), and fresh turns make that likelier."""
    merged: dict[tuple[str, str], ToolCall] = {}
    for turn, call in calls:
        key = (turn, call.tool_call_id)
        earlier = merged.get(key)
        if earlier is None:
            merged[key] = call
        else:
            given = call.model_dump(exclude_defaults=True, exclude={"type"})
            merged[key] = earlier.model_copy(update=given)
    return list(merged.values())


def _relative(repo: Path, given: str) -> str | None:
    root = repo.resolve()
    target = (root / given).resolve() if not Path(given).is_absolute() else Path(given).resolve()
    return str(target.relative_to(root)) if target.is_relative_to(root) else None


def read_ranges(repo: Path, calls: Iterable[tuple[str, ToolCall]]) -> dict[str, list[tuple[int, int]]]:
    """Each file's line ranges `[first, past)` that completed `read` calls returned; `calls` are `(turn, event)`."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for call in _merged(calls):
        if call.tool != "read" or call.status != "completed" or not isinstance(call.input, dict):
            continue
        path, offset, limit = call.input.get("path"), call.input.get("offset", 1), call.input.get("limit", READ_LINES)
        if not isinstance(path, str) or not isinstance(offset, int) or not isinstance(limit, int):
            continue
        if (relative := _relative(repo, path)) is not None:
            first = max(offset, 1)
            ranges.setdefault(relative, []).append((first, first + max(limit, 0)))
    return ranges


def _covers(ranges: Sequence[tuple[int, int]], lines: int) -> bool:
    if not ranges:
        return False
    reached = 1
    for first, past in sorted(ranges):
        if first > reached:
            return False
        reached = max(reached, past)
    return reached > lines


def uncovered(repo: Path, files: Sequence[FileInfo], calls: Iterable[tuple[str, ToolCall]]) -> list[str]:
    """The unit's files some line of which no completed `read` returned."""
    ranges = read_ranges(repo, calls)
    return [file.path for file in files if not _covers(ranges.get(file.path, []), file.lines)]


BRIEF_PROMPT: Final = (
    "Look over this repository before the audit: list its top level, and read its README and build or package "
    "files if there are any. Answer with a brief of at most 200 words: what it is, its languages and frameworks, how "
    "it is built and deployed, and what it handles that an attacker would want. Report nothing yet."
)


def unit_prompt(brief: str, files: Sequence[FileInfo]) -> str:
    listed = "\n".join(f"- {file.path} ({file.lines} lines)" for file in files)
    return (
        f"Repository brief:\n{brief}\n\n"
        f"Audit these files. Read each one in full with read; for a file over {READ_LINES} lines, read it in parts "
        f"with offset and limit until you have seen every line.\n{listed}\n\n"
        "Record each real issue with report_finding as soon as you are sure of it. You may search and read other "
        "files for context. End with one line per file: its path and what you found, or that you found nothing."
    )
