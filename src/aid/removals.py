"""Removal plans: what a `remove` call deletes, worked out before anyone is asked, and kept in SQL for whoever
answers.

A pattern removes what it matches. A matched directory goes whole: the executor removes it as a tree, with
whatever it holds by then. An entry matched inside a directory goes alone. A directory holding something a
filter keeps (an exclude, a tracked file under `untracked_only`, …) cannot go whole: it stays, and what it holds
goes entry by entry. Execution removes the plan's rows, never a new match.

Each session's plans live in its state directory (`PLANS_FILE`), written by its worker. The supervision plugin
attaches the file read-only for a supervisor's `query`; `removal_plans.request_id` joins its rulings.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import anyio
from pydantic import BaseModel

from aid.protocol import AidError
from aid.sql import Database

if TYPE_CHECKING:
    from collections.abc import Iterator

    import apsw

PLANS_FILE: Final = "removals.db"
SCHEMA_VERSION: Final = 1
LISTED: Final = 50
"""Entries a plan's preview names one by one; it counts the rest."""
PROTECTED: Final = ".git"
"""A pattern never matches a directory of this name or anything in it; only an exact path names one."""
MAGIC: Final = re.compile(r"[*?[]")

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS removal_plans (
    id INTEGER PRIMARY KEY,
    created REAL NOT NULL,
    tool_call_id TEXT,
    request_id TEXT,
    request TEXT NOT NULL,
    status TEXT NOT NULL,
    said TEXT,
    decided REAL,
    files INTEGER NOT NULL,
    directories INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    tracked INTEGER
);
CREATE TABLE IF NOT EXISTS removal_entries (
    plan INTEGER NOT NULL REFERENCES removal_plans (id),
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    top INTEGER NOT NULL,
    within TEXT,
    matched TEXT,
    tracked INTEGER,
    ignored INTEGER,
    outcome TEXT,
    PRIMARY KEY (plan, path)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS removal_kept (
    plan INTEGER NOT NULL REFERENCES removal_plans (id),
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (plan, path)
) WITHOUT ROWID;
"""


class EntryKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    LINK = "link"
    OTHER = "other"


class KeptReason(StrEnum):
    EXCLUDED = "excluded"
    TRACKED = "tracked"
    NOT_IGNORED = "not_ignored"
    NEWER = "newer"


class PlanStatus(StrEnum):
    PREVIEWED = "previewed"
    ASKED = "asked"
    REFUSED = "refused"
    REMOVED = "removed"


class Outcome(StrEnum):
    REMOVED = "removed"
    MISSING = "missing"
    CHANGED = "changed"
    FAILED = "failed"


class Removal(BaseModel):
    """What a `remove` call asked for: patterns and filters, relative to the session's directory."""

    paths: list[str] = []
    globs: list[str] = []
    regexes: list[str] = []
    excludes: list[str] = []
    kind: Literal["any", "file", "directory", "link"] = "any"
    older_than_days: float | None = None
    untracked_only: bool = False
    ignored_only: bool = False


@dataclass(frozen=True, slots=True)
class Entry:
    path: str
    """Relative to the session's directory, POSIX."""
    kind: EntryKind
    size: int
    mtime: float
    top: bool
    """The executor removes it itself: a directory as a tree."""
    within: str | None
    """The top directory that takes it along; None for a top entry."""
    matched: str | None
    """The pattern that matched it; None when it goes with its directory."""
    tracked: bool | None
    ignored: bool | None


@dataclass(frozen=True, slots=True)
class Kept:
    path: str
    kind: EntryKind
    reason: KeptReason


@dataclass(slots=True)
class Plan:
    entries: list[Entry] = field(default_factory=list[Entry])
    kept: list[Kept] = field(default_factory=list[Kept])
    unmatched: list[str] = field(default_factory=list[str])
    """Patterns that matched nothing."""

    @property
    def tops(self) -> list[Entry]:
        return [entry for entry in self.entries if entry.top]

    @property
    def files(self) -> int:
        return sum(entry.kind is not EntryKind.DIRECTORY for entry in self.entries)

    @property
    def directories(self) -> int:
        return sum(entry.kind is EntryKind.DIRECTORY for entry in self.entries)

    @property
    def bytes(self) -> int:
        return sum(entry.size for entry in self.entries if entry.kind is not EntryKind.DIRECTORY)

    @property
    def tracked(self) -> int | None:
        if any(entry.tracked is None for entry in self.entries):
            return None
        return sum(bool(entry.tracked) for entry in self.entries)


class RemovalPreview(BaseModel):
    """A plan, for whoever answers its permission request; the rows are in the session's plans database."""

    plan: int
    removal: Removal
    files: int
    """Files and symbolic links: a link goes, never what it points to."""
    directories: int
    bytes: int
    tracked: int | None
    """Entries git tracks; None outside a repository."""
    top: list[str]
    """The first `LISTED` entries the executor removes itself; a directory ends with `/` and goes whole."""
    more: int
    """Top entries past `top`."""
    kept: list[str]
    """The first `LISTED` entries a filter kept, each with its reason."""
    kept_more: int
    unmatched: list[str]


@dataclass(frozen=True, slots=True)
class GitState:
    """What git knows of the session's directory, from two `git ls-files` runs."""

    tracked_files: frozenset[str]
    tracked_dirs: frozenset[str]
    ignored: frozenset[str]
    """Ignored files, and ignored directories with a trailing `/`."""

    def tracked(self, rel: str, kind: EntryKind) -> bool:
        return rel in (self.tracked_dirs if kind is EntryKind.DIRECTORY else self.tracked_files)

    def is_ignored(self, rel: str) -> bool:
        if rel in self.ignored:
            return True
        parts = rel.split("/")
        return any("/".join(parts[:end]) + "/" in self.ignored for end in range(1, len(parts) + 1))


async def git_state(root: Path) -> GitState | None:
    """None outside a git work tree, or without git."""
    try:
        listed = await anyio.run_process(["git", "-C", str(root), "ls-files", "-z"], check=False)
        others = await anyio.run_process(
            ["git", "-C", str(root), "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"],
            check=False,
        )
    except OSError:
        return None
    if listed.returncode or others.returncode:
        return None
    files = frozenset(p for p in listed.stdout.decode(errors="surrogateescape").split("\0") if p)
    dirs = frozenset("/".join(p.split("/")[:end]) for p in files for end in range(1, p.count("/") + 1))
    ignored = frozenset(p for p in others.stdout.decode(errors="surrogateescape").split("\0") if p)
    return GitState(files, dirs, ignored)


def removable(root: Path, path: str) -> Path:
    """`path` itself under `root`: its parent resolved, the entry not, so a link is removed as a link."""
    candidate = Path(os.path.normpath(path if Path(path).is_absolute() else root / path))
    if candidate == root or not candidate.is_relative_to(root):
        raise AidError("outside", f"{path!r} is not inside the session's directory")
    target = candidate.parent.resolve() / candidate.name
    if not target.parent.is_relative_to(root):
        raise AidError("outside", f"{path!r} leads outside the session's directory through a link")
    if not target.exists(follow_symlinks=False):
        raise AidError("not_found", f"{path!r} does not exist")
    return target


def _kind(mode: int) -> EntryKind:
    if stat.S_ISLNK(mode):
        return EntryKind.LINK
    if stat.S_ISDIR(mode):
        return EntryKind.DIRECTORY
    if stat.S_ISREG(mode):
        return EntryKind.FILE
    return EntryKind.OTHER


def _base(root: Path, pattern: str) -> Path | None:
    """Where a glob's matches start: its leading segments without wildcards. None when they do not exist."""
    if Path(pattern).is_absolute() or ".." in pattern.split("/"):
        raise AidError("outside", f"{pattern!r}: a pattern is relative to the session's directory, without `..`")
    literal: list[str] = []
    for segment in pattern.split("/")[:-1]:
        if MAGIC.search(segment):
            break
        literal.append(segment)
    base = root.joinpath(*literal)
    # A link on the way would walk outside the plan's own paths.
    if not base.is_dir(follow_symlinks=False) or base.resolve() != base:
        return None
    return base


def _prefixes(rel: str, *, inclusive: bool = False) -> list[str]:
    """The ancestors of a relative path, outermost first; with `inclusive`, the path itself last."""
    parts = rel.split("/")
    return ["/".join(parts[:end]) for end in range(1, len(parts) + inclusive)]


def _walk(root: Path, base: Path) -> Iterator[str]:
    """Every entry under `base`, relative to `root`, links not followed and `PROTECTED` directories skipped."""
    for current, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != PROTECTED]
        here = Path(current)
        for name in (*dirnames, *filenames):
            yield (here / name).relative_to(root).as_posix()


class _Planner:
    def __init__(self, root: Path, removal: Removal, git: GitState | None, now: float) -> None:
        self.root = root
        self.removal = removal
        self.git = git
        self.excludes = [re.compile(glob.translate(p, recursive=True, include_hidden=True)) for p in removal.excludes]
        self.cutoff = None if removal.older_than_days is None else now - removal.older_than_days * 86400
        self.plan = Plan()
        self.seen: set[str] = set()
        self.whole: set[str] = set()
        self.held: set[str] = set()
        """Kept entries: what is under one stays too."""

    def protected(self, rel: str, kind: EntryKind, mtime: float) -> KeptReason | None:
        """Why a filter keeps this entry, if one does. Only an exclude keeps a directory: the rest judge what it
        holds, so a directory with a kept entry in it stays and the others go."""
        if any(p.fullmatch(prefix) for prefix in _prefixes(rel, inclusive=True) for p in self.excludes):
            return KeptReason.EXCLUDED
        if kind is EntryKind.DIRECTORY:
            return None
        if self.removal.untracked_only and self.git is not None and self.git.tracked(rel, kind):
            return KeptReason.TRACKED
        if self.removal.ignored_only and self.git is not None and not self.git.is_ignored(rel):
            return KeptReason.NOT_IGNORED
        if self.cutoff is not None and mtime > self.cutoff:
            return KeptReason.NEWER
        return None

    def entry(self, rel: str, st: os.stat_result, *, top: bool, within: str | None, matched: str | None) -> Entry:
        kind = _kind(st.st_mode)
        git = self.git
        return Entry(
            path=rel,
            kind=kind,
            size=st.st_size,
            mtime=st.st_mtime,
            top=top,
            within=within,
            matched=matched,
            tracked=None if git is None else git.tracked(rel, kind),
            ignored=None if git is None else git.is_ignored(rel),
        )

    def kinds_match(self, kind: EntryKind) -> bool:
        return self.removal.kind == "any" or self.removal.kind == kind.value

    def match(self, rel: str, label: str) -> None:
        # Under a whole directory it goes already; under a kept one it stays with it.
        if rel in self.seen or any(prefix in self.whole or prefix in self.held for prefix in _prefixes(rel)):
            return
        st = (self.root / rel).lstat()
        kind = _kind(st.st_mode)
        if not self.kinds_match(kind):
            return
        self.seen.add(rel)
        if (reason := self.protected(rel, kind, st.st_mtime)) is not None:
            self.plan.kept.append(Kept(rel, kind, reason))
            self.held.add(rel)
            return
        if kind is not EntryKind.DIRECTORY:
            self.plan.entries.append(self.entry(rel, st, top=True, within=None, matched=label))
            return
        self.directory(rel, st, label)

    def directory(self, rel: str, st: os.stat_result, label: str) -> None:
        """A matched directory: whole when nothing in it is kept, else each part that can go."""
        found: list[tuple[str, os.stat_result]] = []
        kept: list[Kept] = []
        for current, dirnames, filenames in os.walk(self.root / rel):
            here = Path(current)
            descend: list[str] = []
            for name in (*dirnames, *filenames):
                path = here / name
                child = path.relative_to(self.root).as_posix()
                child_st = path.lstat()
                kind = _kind(child_st.st_mode)
                if (reason := self.protected(child, kind, child_st.st_mtime)) is not None:
                    kept.append(Kept(child, kind, reason))
                    continue
                found.append((child, child_st))
                if name in dirnames:
                    descend.append(name)
            dirnames[:] = descend
        self.seen.update(child for child, _ in found)
        self.seen.update(k.path for k in kept)
        self.held.update(k.path for k in kept)
        self.plan.kept.extend(kept)
        blocked = {prefix for k in kept for prefix in _prefixes(k.path)}
        if not kept:
            self.whole.add(rel)
            self.plan.entries.append(self.entry(rel, st, top=True, within=None, matched=label))
        for child, child_st in found:
            if child in blocked:
                continue
            outer = next(
                (a for a in _prefixes(child) if (a == rel or a.startswith(f"{rel}/")) and a not in blocked), None
            )
            if outer is None and _kind(child_st.st_mode) is EntryKind.DIRECTORY:
                self.whole.add(child)
            self.plan.entries.append(self.entry(child, child_st, top=outer is None, within=outer, matched=None))

    def run(self) -> Plan:
        # One sorted pass: a directory comes before what it holds, whichever pattern named each.
        candidates = [
            (removable(self.root, path).relative_to(self.root).as_posix(), f"path {path}")
            for path in self.removal.paths
        ]
        for pattern in self.removal.globs:
            compiled = re.compile(glob.translate(pattern, recursive=True, include_hidden=True))
            base = _base(self.root, pattern)
            hits = [] if base is None else [rel for rel in _walk(self.root, base) if compiled.fullmatch(rel)]
            if not hits:
                self.plan.unmatched.append(f"glob {pattern}")
            candidates.extend((rel, f"glob {pattern}") for rel in hits)
        for pattern in self.removal.regexes:
            try:
                compiled = re.compile(pattern)
            except re.error as error:
                raise AidError("invalid_regex", f"{pattern!r}: {error}") from None
            hits = [rel for rel in _walk(self.root, self.root) if compiled.search(rel)]
            if not hits:
                self.plan.unmatched.append(f"regex {pattern}")
            candidates.extend((rel, f"regex {pattern}") for rel in hits)
        for rel, label in sorted(candidates):
            self.match(rel, label)
        self.plan.entries.sort(key=lambda e: e.path)
        self.plan.kept.sort(key=lambda k: k.path)
        return self.plan


def plan(root: Path, removal: Removal, git: GitState | None, now: float | None = None) -> Plan:
    """Every entry `removal` takes under `root` (resolved), and what its filters keep. Blocking: run off the loop."""
    if not (removal.paths or removal.globs or removal.regexes):
        raise AidError("nothing_named", "name paths, globs or regexes")
    if (removal.untracked_only or removal.ignored_only) and git is None:
        raise AidError("no_git", "untracked_only and ignored_only need a git work tree")
    return _Planner(root, removal, git, time.time() if now is None else now).run()


def execute(root: Path, tops: list[Entry]) -> dict[str, Outcome]:
    """Remove each top entry still as planned: same kind, reached through the same parents. Blocking."""
    outcomes: dict[str, Outcome] = {}
    for entry in tops:
        path = root / entry.path
        try:
            st = path.lstat()
        except FileNotFoundError:
            outcomes[entry.path] = Outcome.MISSING
            continue
        if _kind(st.st_mode) is not entry.kind or path.parent.resolve() != path.parent:
            outcomes[entry.path] = Outcome.CHANGED
            continue
        try:
            if entry.kind is EntryKind.DIRECTORY:
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError:
            outcomes[entry.path] = Outcome.FAILED
            continue
        outcomes[entry.path] = Outcome.REMOVED
    return outcomes


def preview(plan_id: int, removal: Removal, plan: Plan) -> RemovalPreview:
    tops = plan.tops
    return RemovalPreview(
        plan=plan_id,
        removal=removal,
        files=plan.files,
        directories=plan.directories,
        bytes=plan.bytes,
        tracked=plan.tracked,
        top=[f"{e.path}/" if e.kind is EntryKind.DIRECTORY else e.path for e in tops[:LISTED]],
        more=max(0, len(tops) - LISTED),
        kept=[f"{k.path} ({k.reason})" for k in plan.kept[:LISTED]],
        kept_more=max(0, len(plan.kept) - LISTED),
        unmatched=plan.unmatched,
    )


def _insert(db: apsw.Connection, removal: Removal, plan: Plan, tool_call_id: str, request_id: str | None) -> int:
    # BEGIN, not `with db`: apsw's context exit may swallow, so pyright sees no return.
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(
            "INSERT INTO removal_plans (created, tool_call_id, request_id, request, status, files, directories,"
            " bytes, tracked) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                tool_call_id,
                request_id,
                removal.model_dump_json(exclude_defaults=True),
                PlanStatus.ASKED if request_id else PlanStatus.PREVIEWED,
                plan.files,
                plan.directories,
                plan.bytes,
                plan.tracked,
            ),
        )
        plan_id = db.last_insert_rowid()
        db.executemany(
            "INSERT INTO removal_entries (plan, path, kind, size, mtime, top, within, matched, tracked, ignored)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (plan_id, e.path, e.kind, e.size, e.mtime, e.top, e.within, e.matched, e.tracked, e.ignored)
                for e in plan.entries
            ],
        )
        db.executemany(
            "INSERT INTO removal_kept (plan, path, kind, reason) VALUES (?, ?, ?, ?)",
            [(plan_id, k.path, k.kind, k.reason) for k in plan.kept],
        )
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    return plan_id


def _decide(db: apsw.Connection, plan_id: int, status: PlanStatus, said: str | None) -> None:
    db.execute(
        "UPDATE removal_plans SET status = ?, said = ?, decided = ? WHERE id = ?", (status, said, time.time(), plan_id)
    )


def _outcomes(db: apsw.Connection, plan_id: int, outcomes: dict[str, Outcome]) -> None:
    with db:
        db.executemany(
            "UPDATE removal_entries SET outcome = ? WHERE plan = ? AND path = ?",
            [(outcome, plan_id, path) for path, outcome in outcomes.items()],
        )


class PlanStore:
    """A session's plans database, written by its worker only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> PlanStore:
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION))

    async def record(self, removal: Removal, plan: Plan, tool_call_id: str, request_id: str | None) -> int:
        return await self._db.run(_insert, removal, plan, tool_call_id, request_id)

    async def decide(self, plan_id: int, status: PlanStatus, said: str | None = None) -> None:
        await self._db.run(_decide, plan_id, status, said)

    async def outcomes(self, plan_id: int, outcomes: dict[str, Outcome]) -> None:
        await self._db.run(_outcomes, plan_id, outcomes)

    async def aclose(self) -> None:
        await self._db.aclose()
