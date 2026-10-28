"""The findings database: what sessions report, and the verdicts later passes give (docs/audit-campaigns.md).

A finding names a place in the reporting session's directory, `repo`: the plugin takes it from the caller's spec,
never from the model. `evidence` is checked against the file, so a quote the model made up is refused at once.
Duplicates are kept: `dedup` groups them for the review pass, and a refusal would teach the model to vary titles.
"""

from __future__ import annotations

import hashlib
import re
import time
from enum import StrEnum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Annotated, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aid.confine import inside
from aid.protocol import AidError
from aid.sql import Database

if TYPE_CHECKING:
    import apsw

FINDINGS_FILE: Final = "findings.db"
SCHEMA_VERSION: Final = 2
MIGRATIONS: Final = {1: "ALTER TABLE verdicts ADD COLUMN severity TEXT"}
SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT NOT NULL,
    evidence TEXT,
    session TEXT NOT NULL,
    model TEXT,
    reported REAL NOT NULL,
    dedup TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    finding INTEGER NOT NULL REFERENCES findings (id),
    pass TEXT NOT NULL,
    status TEXT NOT NULL,
    duplicate_of INTEGER REFERENCES findings (id),
    note TEXT,
    by TEXT NOT NULL,
    decided REAL NOT NULL,
    severity TEXT
);
CREATE INDEX IF NOT EXISTS findings_dedup ON findings (dedup);
CREATE INDEX IF NOT EXISTS findings_repo ON findings (repo, path);
CREATE INDEX IF NOT EXISTS verdicts_finding ON verdicts (finding, decided);
"""
EVIDENCE_BYTES: Final = 4 << 20
"""Files past this size are not searched for a finding's evidence."""


class Category(StrEnum):
    SECURITY = "security"
    SECRET = "secret"
    BUG = "bug"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    QUALITY = "quality"
    OTHER = "other"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Report(BaseModel):
    """One finding as a session reports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(min_length=1)]
    category: Category
    severity: Severity
    title: Annotated[str, Field(min_length=1, max_length=200)]
    detail: Annotated[str, Field(min_length=1, max_length=8000)]
    line_start: Annotated[int, Field(ge=1)] | None = None
    line_end: Annotated[int, Field(ge=1)] | None = None
    evidence: Annotated[str, Field(min_length=1, max_length=4000)] | None = None

    @model_validator(mode="after")
    def _lines_in_order(self) -> Report:
        if self.line_end is not None and (self.line_start is None or self.line_end < self.line_start):
            raise ValueError("line_end needs a line_start at or before it")
        return self


def _squeezed(text: str) -> str:
    return " ".join(text.split())


def dedup_key(repo: str, report: Report) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", report.title.lower()).strip()
    key = "\0".join((repo, report.path, str(report.line_start or ""), report.category, title))
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def check(cwd: Path, report: Report) -> tuple[str, str]:
    """The resolved `cwd` and the report's path relative to it, or `AidError` naming what the model got wrong."""
    repo = cwd.resolve()
    return str(repo), _checked(repo, report)


def _checked(repo: Path, report: Report) -> str:
    if PurePath(report.path).is_absolute():
        # As `Coding.resolve` reads one: absolute is absolute, and must lie in the directory.
        target = inside(repo, str(Path(report.path).resolve().relative_to(repo, walk_up=True)))
    else:
        target = inside(repo, report.path)
    relative = str(target.relative_to(repo)) if target != repo else "."
    if not target.exists():
        raise AidError("no_such_file", f"{report.path} does not exist in {repo}")
    if target.is_dir():
        if report.line_start is not None or report.evidence is not None:
            raise AidError("bad_args", f"{report.path} is a directory: give no lines or evidence")
        return relative
    if target.stat().st_size > EVIDENCE_BYTES:
        return relative
    content = target.read_bytes()
    try:
        text = content.decode()
    except UnicodeDecodeError:
        return relative
    lines = text.count("\n") + (not text.endswith("\n"))
    if report.line_start is not None and report.line_start > lines:
        raise AidError("bad_args", f"{report.path} has {lines} lines, not {report.line_start}")
    if report.evidence is not None and _squeezed(report.evidence) not in _squeezed(text):
        raise AidError("bad_args", f"the evidence is not in {report.path}: quote the file exactly")
    return relative


def _insert(
    db: apsw.Connection, repo: str, path: str, report: Report, session: str, model: str | None
) -> tuple[int, int | None]:
    key = dedup_key(repo, report.model_copy(update={"path": path}))
    first = db.execute("SELECT min(id) FROM findings WHERE dedup = ?", (key,)).get
    db.execute(
        "INSERT INTO findings (repo, path, line_start, line_end, category, severity, title, detail, evidence,"
        " session, model, reported, dedup) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            repo,
            path,
            report.line_start,
            report.line_end,
            report.category,
            report.severity,
            report.title,
            report.detail,
            report.evidence,
            session,
            model,
            time.time(),
            key,
        ),
    )
    return db.last_insert_rowid(), first if isinstance(first, int) else None


class Status(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    UNSURE = "unsure"


class Verdict(BaseModel):
    """A review pass's judgement of one finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: int
    status: Status
    note: Annotated[str, Field(min_length=1, max_length=4000)]
    severity: Severity | None = None
    """The severity the reviewer gives, when it differs from the reporter's."""
    duplicate_of: int | None = None

    @model_validator(mode="after")
    def _duplicate_names_one(self) -> Verdict:
        if (self.status == Status.DUPLICATE) != (self.duplicate_of is not None):
            raise ValueError("a duplicate names duplicate_of, and only a duplicate does")
        return self


def _judge(db: apsw.Connection, repo: str, verdict: Verdict, session: str, review: str) -> None:
    """Record `verdict` if its findings lie in `repo` and `session` did not report the one judged."""
    row = db.execute("SELECT repo, session FROM findings WHERE id = ?", (verdict.finding,)).fetchone()
    if row is None or row[0] != repo:
        raise AidError("not_found", f"no finding {verdict.finding} in this directory")
    if row[1] == session:
        raise AidError("forbidden", "a session does not judge its own findings")
    if verdict.duplicate_of is not None:
        other = db.execute("SELECT repo FROM findings WHERE id = ?", (verdict.duplicate_of,)).fetchone()
        if other is None or other[0] != repo or verdict.duplicate_of == verdict.finding:
            raise AidError("not_found", f"no other finding {verdict.duplicate_of} in this directory")
    db.execute(
        "INSERT INTO verdicts (finding, pass, status, duplicate_of, note, by, decided, severity)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            verdict.finding,
            review,
            verdict.status,
            verdict.duplicate_of,
            verdict.note,
            session,
            time.time(),
            verdict.severity,
        ),
    )


class FindingStore:
    """The findings plugin's database, written by the plugin only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> FindingStore:
        """Findings may quote secrets: the file is the owner's alone, and SQLite gives its -wal and -shm the same."""
        await anyio.Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        await anyio.Path(path).touch(mode=0o600)
        await anyio.Path(path).chmod(0o600)
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION, MIGRATIONS))

    async def insert(
        self, repo: str, path: str, report: Report, *, session: str, model: str | None
    ) -> tuple[int, int | None]:
        """The new finding's id, and the first earlier one at the same place with the same title."""
        return await self._db.run(_insert, repo, path, report, session, model)

    async def judge(self, repo: str, verdict: Verdict, *, session: str, review: str) -> None:
        await self._db.run(_judge, repo, verdict, session, review)

    async def aclose(self) -> None:
        await self._db.aclose()
