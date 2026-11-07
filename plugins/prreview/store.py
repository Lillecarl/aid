"""The PR reviews, in the plugin's own database (`prreview.db`): every row is keyed by the PR's number and head, so a
head that moved is a new review.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import anyio
from pydantic import BaseModel

from aid.sql import Database

if TYPE_CHECKING:
    from pathlib import Path

    import apsw

    from aid.protocol import Usage

    from .review import Build, Pr, Review

PRREVIEW_FILE: Final = "prreview.db"
SCHEMA_VERSION: Final = 2
MIGRATIONS: Final = {1: "ALTER TABLE prs ADD COLUMN step TEXT"}
SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS runs (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    state TEXT NOT NULL,
    error TEXT,
    created REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS prs (
    run TEXT NOT NULL REFERENCES runs (name),
    number INTEGER NOT NULL,
    head TEXT NOT NULL,
    base TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT NOT NULL,
    url TEXT NOT NULL,
    diff TEXT,
    state TEXT NOT NULL,
    skipped TEXT,
    duplicate_of INTEGER,
    error TEXT,
    step TEXT,
    started REAL NOT NULL,
    ended REAL,
    PRIMARY KEY (run, number, head)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS builds (
    run TEXT NOT NULL,
    number INTEGER NOT NULL,
    head TEXT NOT NULL,
    attr TEXT NOT NULL,
    outcome TEXT NOT NULL,
    seconds REAL NOT NULL,
    log_tail TEXT NOT NULL,
    out_paths TEXT NOT NULL,
    closure_size INTEGER,
    PRIMARY KEY (run, number, head, attr),
    FOREIGN KEY (run, number, head) REFERENCES prs (run, number, head)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS angle_reviews (
    run TEXT NOT NULL,
    number INTEGER NOT NULL,
    head TEXT NOT NULL,
    angle TEXT NOT NULL,
    session TEXT NOT NULL,
    answer TEXT NOT NULL,
    usage TEXT,
    at REAL NOT NULL,
    PRIMARY KEY (run, number, head, angle),
    FOREIGN KEY (run, number, head) REFERENCES prs (run, number, head)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS duplicate_checks (
    run TEXT NOT NULL,
    number INTEGER NOT NULL,
    head TEXT NOT NULL,
    candidates TEXT NOT NULL,
    answer TEXT,
    duplicates TEXT,
    winner TEXT,
    beaten_by INTEGER,
    session TEXT,
    at REAL NOT NULL,
    PRIMARY KEY (run, number, head),
    FOREIGN KEY (run, number, head) REFERENCES prs (run, number, head)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS reviews (
    run TEXT NOT NULL,
    number INTEGER NOT NULL,
    head TEXT NOT NULL,
    event TEXT NOT NULL,
    body TEXT NOT NULL,
    comments TEXT NOT NULL,
    session TEXT NOT NULL,
    at REAL NOT NULL,
    PRIMARY KEY (run, number, head),
    FOREIGN KEY (run, number, head) REFERENCES prs (run, number, head)
) WITHOUT ROWID;
"""


class RunState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


class PrState(StrEnum):
    STARTED = "started"
    """Picked; whatever came after is done again on a restart, but angle answers at this head are kept."""
    BUILT = "built"
    REVIEWED = "reviewed"
    """The judge recorded its review."""
    DUPLICATE = "duplicate"
    """Another PR makes the same change and wins (`duplicate_of`): a merged one, or the better open one."""
    SKIPPED = "skipped"
    """Not reviewed: `skipped` says why (no CI artifact, too many rebuilds, the head moved)."""
    FAILED = "failed"


ENDED: Final = frozenset({PrState.REVIEWED, PrState.DUPLICATE, PrState.SKIPPED, PrState.FAILED})


@dataclass(frozen=True)
class DuplicateCheck:
    candidates: list[int]
    answer: str | None = None
    """The model's last answer; None when no candidate needed asking."""
    duplicates: list[int] | None = None
    winner: int | str | None = None
    beaten_by: int | None = None
    session: str | None = None


class Run(BaseModel):
    name: str
    config: str
    state: RunState
    error: str | None


class PrRow(BaseModel):
    number: int
    head: str
    state: PrState
    step: str | None = None
    """What a PR under review is doing now: a build, the angles, the judge."""


def _create(db: apsw.Connection, name: str, config: str) -> None:
    db.execute(
        "INSERT INTO runs (name, config, state, created) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (name) DO UPDATE SET config = excluded.config, state = excluded.state, error = NULL",
        (name, config, RunState.RUNNING, time.time()),
    )


def _runs(db: apsw.Connection) -> list[Run]:
    rows = db.execute("SELECT name, config, state, error FROM runs ORDER BY created")
    return [
        Run(name=str(n), config=str(c), state=RunState(str(s)), error=e if isinstance(e, str) else None)
        for n, c, s, e in rows
    ]


def _started(db: apsw.Connection, run: str, pr: Pr) -> None:
    with db:
        db.execute(
            "INSERT INTO prs (run, number, head, base, title, body, author, url, state, started)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (run, number, head) DO UPDATE SET state = excluded.state, error = NULL, skipped = NULL,"
            " duplicate_of = NULL",
            (run, pr.number, pr.head, pr.base, pr.title, pr.body, pr.author, pr.url, PrState.STARTED, time.time()),
        )
        db.execute("DELETE FROM builds WHERE run = ? AND number = ? AND head = ?", (run, pr.number, pr.head))


def _built(db: apsw.Connection, run: str, pr: Pr, diff: str, builds: list[Build], skipped: str | None) -> None:
    with db:
        for b in builds:
            db.execute(
                "INSERT OR REPLACE INTO builds"
                " (run, number, head, attr, outcome, seconds, log_tail, out_paths, closure_size)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run,
                    pr.number,
                    pr.head,
                    b.attr,
                    b.outcome,
                    b.seconds,
                    b.log_tail,
                    json.dumps(b.out_paths),
                    b.closure_size,
                ),
            )
        db.execute(
            "UPDATE prs SET state = ?, diff = ?, skipped = ? WHERE run = ? AND number = ? AND head = ?",
            (PrState.BUILT, diff, skipped, run, pr.number, pr.head),
        )


def _ended(db: apsw.Connection, run: str, number: int, head: str, state: PrState, note: str | None) -> None:
    column = "skipped" if state == PrState.SKIPPED else "error"
    db.execute(
        f"UPDATE prs SET state = ?, {column} = ?, ended = ? WHERE run = ? AND number = ? AND head = ?",
        (state, note, time.time(), run, number, head),
    )


def _angles(db: apsw.Connection, run: str, number: int, head: str) -> dict[str, str]:
    rows = db.execute(
        "SELECT angle, answer FROM angle_reviews WHERE run = ? AND number = ? AND head = ?", (run, number, head)
    )
    return {str(angle): str(answer) for angle, answer in rows}


def _review(db: apsw.Connection, run: str, number: int, head: str, review: Review, session: str) -> bool:
    db.execute(
        "INSERT OR IGNORE INTO reviews (run, number, head, event, body, comments, session, at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run,
            number,
            head,
            review.event,
            review.body,
            json.dumps([c.model_dump() for c in review.comments]),
            session,
            time.time(),
        ),
    )
    return db.changes() == 1


class PrStore:
    """The prreview plugin's database, written by the plugin only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> PrStore:
        await anyio.Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION, MIGRATIONS))

    async def aclose(self) -> None:
        await self._db.aclose()

    async def create(self, name: str, config: str) -> None:
        await self._db.run(_create, name, config)

    async def runs(self) -> list[Run]:
        return await self._db.run(_runs)

    async def set_state(self, name: str, state: RunState, error: str | None = None) -> None:
        await self._db.run(
            lambda db: db.execute("UPDATE runs SET state = ?, error = ? WHERE name = ?", (state, error, name))
        )

    async def prs(self, run: str) -> list[PrRow]:
        def rows(db: apsw.Connection) -> list[PrRow]:
            found = db.execute("SELECT number, head, state, step FROM prs WHERE run = ? ORDER BY started", (run,))
            return [
                PrRow(number=int(str(n)), head=str(h), state=PrState(str(s)), step=None if t is None else str(t))
                for n, h, s, t in found
            ]

        return await self._db.run(rows)

    async def step(self, run: str, pr: Pr, step: str) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE prs SET step = ? WHERE run = ? AND number = ? AND head = ?", (step, run, pr.number, pr.head)
            )
        )

    async def started(self, run: str, pr: Pr) -> None:
        await self._db.run(_started, run, pr)

    async def built(self, run: str, pr: Pr, diff: str, builds: list[Build], skipped: str | None) -> None:
        await self._db.run(_built, run, pr, diff, builds, skipped)

    async def ended(self, run: str, number: int, head: str, state: PrState, note: str | None = None) -> None:
        await self._db.run(_ended, run, number, head, state, note)

    async def duplicate_check(self, run: str, pr: Pr, check: DuplicateCheck) -> None:
        """The check's record; a PR beaten by another ends `DUPLICATE` with it."""

        def write(db: apsw.Connection) -> None:
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO duplicate_checks"
                    " (run, number, head, candidates, answer, duplicates, winner, beaten_by, session, at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run,
                        pr.number,
                        pr.head,
                        json.dumps(check.candidates),
                        check.answer,
                        None if check.duplicates is None else json.dumps(check.duplicates),
                        None if check.winner is None else str(check.winner),
                        check.beaten_by,
                        check.session,
                        time.time(),
                    ),
                )
                if check.beaten_by is not None:
                    db.execute(
                        "UPDATE prs SET state = ?, duplicate_of = ?, skipped = ?, ended = ?"
                        " WHERE run = ? AND number = ? AND head = ?",
                        (
                            PrState.DUPLICATE,
                            check.beaten_by,
                            f"a duplicate of #{check.beaten_by}",
                            time.time(),
                            run,
                            pr.number,
                            pr.head,
                        ),
                    )

        await self._db.run(write)

    async def angles(self, run: str, number: int, head: str) -> dict[str, str]:
        return await self._db.run(_angles, run, number, head)

    async def angle(
        self, run: str, number: int, head: str, angle: str, session: str, answer: str, usage: Usage | None
    ) -> None:
        dumped = None if usage is None else usage.model_dump_json(exclude_none=True)
        await self._db.run(
            lambda db: db.execute(
                "INSERT OR REPLACE INTO angle_reviews (run, number, head, angle, session, answer, usage, at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run, number, head, angle, session, answer, dumped, time.time()),
            )
        )

    async def review(self, run: str, number: int, head: str, review: Review, session: str) -> bool:
        """False when the PR at this head holds a review already."""
        return await self._db.run(_review, run, number, head, review, session)

    async def reviewed(self, run: str, number: int, head: str) -> bool:
        return await self.event(run, number, head) is not None

    async def event(self, run: str, number: int, head: str) -> str | None:
        """The recorded review's event, None before one is recorded."""

        def found(db: apsw.Connection) -> str | None:
            sql = "SELECT event FROM reviews WHERE run = ? AND number = ? AND head = ?"
            row = db.execute(sql, (run, number, head)).fetchone()
            return None if row is None else str(row[0])

        return await self._db.run(found)
