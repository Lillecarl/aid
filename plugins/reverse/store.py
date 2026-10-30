"""The investigations database (docs/reverse-engineering.md): each investigation's config, and the evidence its
sessions record.

A row keeps apart what was seen (an observation), what was reasoned from rows it names (an inference), what is to be
tested (a hypothesis) and what is unknown (a question). An artifact is named by its path in the workspace and its
hash when recorded, so a later change to the file shows. The plugin takes the investigation and the workspace from
the caller's spec, never from the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
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

REVERSE_FILE: Final = "reverse.db"
SCHEMA_VERSION: Final = 1
SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS investigations (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    investigation TEXT NOT NULL REFERENCES investigations (name),
    kind TEXT NOT NULL,
    claim TEXT NOT NULL,
    detail TEXT,
    artifact TEXT,
    sha256 TEXT,
    location TEXT,
    source TEXT,
    supports TEXT NOT NULL,
    session TEXT NOT NULL,
    model TEXT,
    recorded REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settlements (
    evidence INTEGER NOT NULL REFERENCES evidence (id),
    status TEXT NOT NULL,
    note TEXT NOT NULL,
    answer INTEGER REFERENCES evidence (id),
    by TEXT NOT NULL,
    decided REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_investigation ON evidence (investigation, kind);
CREATE INDEX IF NOT EXISTS settlements_evidence ON settlements (evidence, decided);
"""
HASH_BYTES: Final = 1 << 30
"""Artifacts past this size are recorded without a hash: a firmware image is far smaller, a disk image is not."""


class Kind(StrEnum):
    OBSERVATION = "observation"
    """Seen directly: in a tool's output, a table, the bytes."""
    INFERENCE = "inference"
    """Reasoned from the rows it names in `supports`."""
    HYPOTHESIS = "hypothesis"
    """A claim to test; settled confirmed or refuted."""
    QUESTION = "question"
    """Unknown; settled answered."""


class Status(StrEnum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    ANSWERED = "answered"
    RETRACTED = "retracted"
    """The row was wrong, of any kind: an observation misread, an inference on a refuted hypothesis."""


SETTLES: Final = {
    Kind.HYPOTHESIS: {Status.CONFIRMED, Status.REFUTED, Status.RETRACTED},
    Kind.QUESTION: {Status.ANSWERED, Status.RETRACTED},
    Kind.OBSERVATION: {Status.RETRACTED},
    Kind.INFERENCE: {Status.RETRACTED},
}


class Evidence(BaseModel):
    """One row as a session records it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Kind
    claim: Annotated[str, Field(min_length=1, max_length=300)]
    detail: Annotated[str, Field(max_length=8000)] | None = None
    artifact: Annotated[str, Field(min_length=1)] | None = None
    """A file in the workspace the claim is about."""
    location: Annotated[str, Field(max_length=300)] | None = None
    """Where in it: an address, an offset, an ACPI path, a register."""
    source: Annotated[str, Field(max_length=2000)] | None = None
    """The command or tool call that showed it."""
    supports: list[int] = Field(default_factory=list[int])
    """The rows this one rests on."""

    @model_validator(mode="after")
    def _inference_names_its_ground(self) -> Evidence:
        if self.kind == Kind.INFERENCE and not self.supports:
            raise ValueError("an inference names the rows it rests on in supports")
        return self


class Settlement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: int
    status: Status
    note: Annotated[str, Field(min_length=1, max_length=4000)]
    answer: int | None = None
    """The row that settles it, such as the observation a test produced."""


def artifact(workspace: Path, path: str) -> tuple[str, str | None]:
    """`path` relative to the workspace and its sha256, or `AidError` naming what is wrong. Runs in a thread."""
    root = workspace.resolve()
    if PurePath(path).is_absolute():
        target = inside(root, str(Path(path).resolve().relative_to(root, walk_up=True)))
    else:
        target = inside(root, path)
    if not target.is_file():
        raise AidError("no_such_file", f"{path} is no file in the workspace {root}")
    if target.stat().st_size > HASH_BYTES:
        return str(target.relative_to(root)), None
    digest = hashlib.sha256()
    with target.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return str(target.relative_to(root)), digest.hexdigest()


@dataclass(frozen=True)
class Recorded:
    """Who records a row, and the artifact the plugin checked."""

    investigation: str
    session: str
    model: str | None
    artifact: str | None
    sha256: str | None


def _known(db: apsw.Connection, investigation: str, ids: list[int]) -> None:
    for row in ids:
        found = db.execute("SELECT 1 FROM evidence WHERE id = ? AND investigation = ?", (row, investigation)).get
        if found is None:
            raise AidError("not_found", f"no evidence {row} in this investigation")


def _record(db: apsw.Connection, evidence: Evidence, who: Recorded) -> int:
    _known(db, who.investigation, evidence.supports)
    db.execute(
        "INSERT INTO evidence (investigation, kind, claim, detail, artifact, sha256, location, source, supports,"
        " session, model, recorded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            who.investigation,
            evidence.kind,
            evidence.claim,
            evidence.detail,
            who.artifact,
            who.sha256,
            evidence.location,
            evidence.source,
            json.dumps(evidence.supports),
            who.session,
            who.model,
            time.time(),
        ),
    )
    return db.last_insert_rowid()


def _settle(db: apsw.Connection, investigation: str, settlement: Settlement, by: str) -> None:
    row = db.execute(
        "SELECT kind FROM evidence WHERE id = ? AND investigation = ?", (settlement.evidence, investigation)
    ).fetchone()
    if row is None:
        raise AidError("not_found", f"no evidence {settlement.evidence} in this investigation")
    kind = Kind(row[0])
    if settlement.status not in SETTLES[kind]:
        allowed = ", ".join(sorted(SETTLES[kind]))
        raise AidError("bad_args", f"evidence {settlement.evidence} is a {kind}: it settles as {allowed}")
    if settlement.answer is not None:
        _known(db, investigation, [settlement.answer])
    db.execute(
        "INSERT INTO settlements (evidence, status, note, answer, by, decided) VALUES (?, ?, ?, ?, ?, ?)",
        (settlement.evidence, settlement.status, settlement.note, settlement.answer, by, time.time()),
    )


LISTING: Final = """
SELECT e.id, e.kind, e.claim, e.artifact, e.location, e.supports,
       (SELECT s.status FROM settlements s WHERE s.evidence = e.id ORDER BY s.decided DESC LIMIT 1) AS status
FROM evidence e WHERE e.investigation = ? AND (? IS NULL OR e.kind = ?) ORDER BY e.id
"""


def _listing(db: apsw.Connection, investigation: str, kind: Kind | None) -> str:
    lines: list[str] = []
    for row_id, row_kind, claim, path, location, supports, status in db.execute(LISTING, (investigation, kind, kind)):
        place = " @ ".join(str(p) for p in (path, location) if p)
        rests = json.loads(str(supports))
        parts = [f"#{row_id} {row_kind}", f"[{status}]" if status else "", claim]
        tail = [f"({place})" if place else "", f"from {', '.join(f'#{n}' for n in rests)}" if rests else ""]
        lines.append(" ".join(p for p in parts + tail if p))
    return "\n".join(lines) or "No evidence recorded yet."


def _counts(db: apsw.Connection, investigation: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind, n in db.execute(
        "SELECT kind, count(*) FROM evidence WHERE investigation = ? GROUP BY kind", (investigation,)
    ):
        counts[str(kind)] = int(n)
    return counts


def _create(db: apsw.Connection, name: str, config: str) -> None:
    if db.execute("SELECT 1 FROM investigations WHERE name = ?", (name,)).get is not None:
        raise AidError("exists", f"an investigation {name!r} exists")
    db.execute("INSERT INTO investigations (name, config, created) VALUES (?, ?, ?)", (name, config, time.time()))


def _configs(db: apsw.Connection) -> list[tuple[str, str]]:
    return [(str(name), str(config)) for name, config in db.execute("SELECT name, config FROM investigations")]


class ReverseStore:
    """The plugin's database, written by the plugin only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> ReverseStore:
        """Evidence may quote firmware and machine details: the file is the owner's alone."""
        await anyio.Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        await anyio.Path(path).touch(mode=0o600)
        await anyio.Path(path).chmod(0o600)
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION, {}))

    async def create(self, name: str, config: str) -> None:
        await self._db.run(_create, name, config)

    async def configs(self) -> list[tuple[str, str]]:
        return await self._db.run(_configs)

    async def record(self, evidence: Evidence, who: Recorded) -> int:
        return await self._db.run(_record, evidence, who)

    async def settle(self, investigation: str, settlement: Settlement, by: str) -> None:
        await self._db.run(_settle, investigation, settlement, by)

    async def listing(self, investigation: str, kind: Kind | None) -> str:
        return await self._db.run(_listing, investigation, kind)

    async def counts(self, investigation: str) -> dict[str, int]:
        return await self._db.run(_counts, investigation)

    async def aclose(self) -> None:
        await self._db.aclose()
