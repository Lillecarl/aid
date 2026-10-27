"""A campaign's state, in the plugin's own database: every step resumes from these rows alone.

A unit records the history seq before each attempt (`start_seq`) and, after it, the files read in full (`covered`):
a restart credits what the cut attempt recorded before prompting again.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import anyio
from pydantic import BaseModel, TypeAdapter

from aid.protocol import AidError
from aid.sql import Database

from .units import FileInfo, Skipped

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import apsw

CAMPAIGN_FILE: Final = "campaign.db"
SCHEMA_VERSION: Final = 2
MIGRATIONS: Final = {1: "ALTER TABLE units ADD COLUMN covered TEXT"}
SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS campaigns (
    name TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    state TEXT NOT NULL,
    error TEXT,
    created REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS repos (
    campaign TEXT NOT NULL REFERENCES campaigns (name),
    path TEXT NOT NULL,
    session TEXT NOT NULL,
    state TEXT NOT NULL,
    brief TEXT,
    skipped TEXT,
    error TEXT,
    PRIMARY KEY (campaign, path)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS units (
    campaign TEXT NOT NULL,
    repo TEXT NOT NULL,
    idx INTEGER NOT NULL,
    files TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    start_seq INTEGER,
    covered TEXT,
    missing TEXT,
    error TEXT,
    started REAL,
    ended REAL,
    PRIMARY KEY (campaign, repo, idx),
    FOREIGN KEY (campaign, repo) REFERENCES repos (campaign, path)
) WITHOUT ROWID;
"""
_FILES: Final = TypeAdapter(list[FileInfo])
_SKIPPED: Final = TypeAdapter(list[Skipped])
_PATHS: Final = TypeAdapter(list[str])


class CampaignState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


class RepoState(StrEnum):
    PENDING = "pending"
    LISTED = "listed"
    """Its units are planned."""
    BRIEFED = "briefed"
    DONE = "done"
    FAILED = "failed"


class UnitState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    INCOMPLETE = "incomplete"
    """Attempts ran out with files not read in full; `missing` names them."""
    FAILED = "failed"
    """Attempts ran out on turn errors; `error` holds the last."""


FINISHED: Final = frozenset({UnitState.DONE, UnitState.INCOMPLETE, UnitState.FAILED})


class Campaign(BaseModel):
    name: str
    config: str
    """`CampaignConfig` as JSON: the plugin validates it, the store keeps it."""
    state: CampaignState
    error: str | None


class Repo(BaseModel):
    campaign: str
    path: str
    session: str
    state: RepoState
    brief: str | None
    skipped: list[Skipped]
    error: str | None


class Unit(BaseModel):
    campaign: str
    repo: str
    idx: int
    files: list[FileInfo]
    state: UnitState
    attempts: int
    start_seq: int | None
    """Where the running attempt began in the session's history: a resumed unit credits what it recorded since."""
    covered: list[str]
    """Files earlier attempts read every line of, in turns that ended."""
    missing: list[str]
    error: str | None


def _repo(row: apsw.SQLiteValues) -> Repo:
    campaign, path, session, state, brief, skipped, error = row
    return Repo.model_validate(
        {
            "campaign": campaign,
            "path": path,
            "session": session,
            "state": state,
            "brief": brief,
            "skipped": _SKIPPED.validate_json(skipped) if isinstance(skipped, str) else [],
            "error": error,
        }
    )


def _unit(row: apsw.SQLiteValues) -> Unit:
    campaign, repo, idx, files, state, attempts, start_seq, covered, missing, error = row
    return Unit.model_validate(
        {
            "campaign": campaign,
            "repo": repo,
            "idx": idx,
            "files": _FILES.validate_json(files) if isinstance(files, str) else [],
            "state": state,
            "attempts": attempts,
            "start_seq": start_seq,
            "covered": _PATHS.validate_json(covered) if isinstance(covered, str) else [],
            "missing": _PATHS.validate_json(missing) if isinstance(missing, str) else [],
            "error": error,
        }
    )


def _create(db: apsw.Connection, name: str, config: str) -> None:
    if db.execute("SELECT 1 FROM campaigns WHERE name = ?", (name,)).get is not None:
        raise AidError("exists", f"a campaign {name!r} exists")
    db.execute(
        "INSERT INTO campaigns (name, config, state, created) VALUES (?, ?, ?, ?)",
        (name, config, CampaignState.RUNNING, time.time()),
    )


def _campaigns(db: apsw.Connection) -> list[Campaign]:
    rows = db.execute("SELECT name, config, state, error FROM campaigns ORDER BY created")
    return [
        Campaign(name=n, config=str(c), state=CampaignState(str(s)), error=e if isinstance(e, str) else None)
        for n, c, s, e in rows
        if isinstance(n, str)
    ]


def _add_repos(db: apsw.Connection, campaign: str, repos: Sequence[tuple[str, str]]) -> None:
    with db:
        for path, session in repos:
            db.execute(
                "INSERT OR IGNORE INTO repos (campaign, path, session, state) VALUES (?, ?, ?, ?)",
                (campaign, path, session, RepoState.PENDING),
            )


def _plan(
    db: apsw.Connection, campaign: str, repo: str, units: Sequence[Sequence[FileInfo]], skipped: Sequence[Skipped]
) -> None:
    with db:
        db.execute("DELETE FROM units WHERE campaign = ? AND repo = ?", (campaign, repo))
        for idx, files in enumerate(units):
            db.execute(
                "INSERT INTO units (campaign, repo, idx, files, state) VALUES (?, ?, ?, ?, ?)",
                (campaign, repo, idx, _FILES.dump_json(list(files)).decode(), UnitState.PENDING),
            )
        db.execute(
            "UPDATE repos SET state = ?, skipped = ? WHERE campaign = ? AND path = ?",
            (RepoState.LISTED, _SKIPPED.dump_json(list(skipped)).decode(), campaign, repo),
        )


class CampaignStore:
    """The campaign plugin's database, written by the plugin only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> CampaignStore:
        await anyio.Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION, MIGRATIONS))

    async def aclose(self) -> None:
        await self._db.aclose()

    async def create(self, name: str, config: str) -> None:
        await self._db.run(_create, name, config)

    async def campaigns(self) -> list[Campaign]:
        return await self._db.run(_campaigns)

    async def set_state(self, name: str, state: CampaignState, error: str | None = None) -> None:
        await self._db.run(
            lambda db: db.execute("UPDATE campaigns SET state = ?, error = ? WHERE name = ?", (state, error, name))
        )

    async def add_repos(self, campaign: str, repos: Sequence[tuple[str, str]]) -> None:
        """`(path, session)` pairs; a repository already known keeps its row."""
        await self._db.run(_add_repos, campaign, repos)

    async def repos(self, campaign: str) -> list[Repo]:
        return await self._db.run(
            lambda db: [
                _repo(row)
                for row in db.execute(
                    "SELECT campaign, path, session, state, brief, skipped, error FROM repos WHERE campaign = ?"
                    " ORDER BY path",
                    (campaign,),
                )
            ]
        )

    async def plan(
        self, campaign: str, repo: str, units: Sequence[Sequence[FileInfo]], skipped: Sequence[Skipped]
    ) -> None:
        await self._db.run(_plan, campaign, repo, units, skipped)

    async def briefed(self, campaign: str, repo: str, brief: str) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE repos SET state = ?, brief = ? WHERE campaign = ? AND path = ?",
                (RepoState.BRIEFED, brief, campaign, repo),
            )
        )

    async def repo_ended(self, campaign: str, repo: str, state: RepoState, error: str | None = None) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE repos SET state = ?, error = ? WHERE campaign = ? AND path = ?", (state, error, campaign, repo)
            )
        )

    async def units(self, campaign: str, repo: str) -> list[Unit]:
        return await self._db.run(
            lambda db: [
                _unit(row)
                for row in db.execute(
                    "SELECT campaign, repo, idx, files, state, attempts, start_seq, covered, missing, error FROM units"
                    " WHERE campaign = ? AND repo = ? ORDER BY idx",
                    (campaign, repo),
                )
            ]
        )

    async def attempt_started(self, unit: Unit, start_seq: int) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE units SET state = ?, start_seq = ?, started = ? WHERE campaign = ? AND repo = ? AND idx = ?",
                (UnitState.RUNNING, start_seq, time.time(), unit.campaign, unit.repo, unit.idx),
            )
        )

    async def attempt_ended(
        self, unit: Unit, attempts: int, covered: Sequence[str], missing: Sequence[str], error: str | None
    ) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE units SET attempts = ?, start_seq = NULL, covered = ?, missing = ?, error = ?"
                " WHERE campaign = ? AND repo = ? AND idx = ?",
                (
                    attempts,
                    json.dumps(list(covered)),
                    json.dumps(list(missing)),
                    error,
                    unit.campaign,
                    unit.repo,
                    unit.idx,
                ),
            )
        )

    async def unit_ended(self, unit: Unit, state: UnitState) -> None:
        await self._db.run(
            lambda db: db.execute(
                "UPDATE units SET state = ?, ended = ? WHERE campaign = ? AND repo = ? AND idx = ?",
                (state, time.time(), unit.campaign, unit.repo, unit.idx),
            )
        )
