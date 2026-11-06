"""A campaign's units, pure: which files of a repository to audit, in which groups, and the prompt that carries them.

A unit's prompt holds every line of its files, numbered as `read` numbers them, so coverage holds by construction:
a unit is audited once a turn over it ends on its own. Measured on the dynamist campaign (2026-11): with the model
reading each file through tools, a unit took 13.6 requests, and each request cost about 420 thinking tokens.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Annotated, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field

from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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


BRIEF_PROMPT: Final = (
    "Look over this repository before the audit: list its top level, and read its README and build or package "
    "files if there are any. Answer with a brief of at most 200 words: what it is, its languages and frameworks, how "
    "it is built and deployed, and what it handles that an attacker would want. Report nothing yet."
)


def numbered(text: str) -> str:
    """Every line as `aid.coding.numbered` gives it, which the model knows from `read`: not imported, so the daemon
    need not load aid.coding."""
    return "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(text.splitlines(), start=1))


def load(repo: Path, files: Sequence[FileInfo]) -> list[tuple[FileInfo, str]]:
    """Each file's text, read as `examine` read it. Blocking: run it in a thread."""
    return [(file, (repo / file.path).read_bytes().decode(errors="replace")) for file in files]


def unit_prompt(brief: str, contents: Sequence[tuple[FileInfo, str]]) -> str:
    shown = "\n\n".join(
        f'<file path="{file.path}" lines="{file.lines}">\n{numbered(text)}\n</file>' for file, text in contents
    )
    return (
        f"Repository brief:\n{brief}\n\n"
        "Audit the files below. Each one is here in full, numbered as read numbers it: you need not read them again. "
        "Record each real issue with report_finding as soon as you are sure of it, with the line numbers shown. You "
        "may search and read other files for context. End with one line per file: its path and what you found, or "
        f"that you found nothing.\n\n{shown}"
    )
