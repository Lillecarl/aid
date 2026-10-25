"""Supervisor rulings: every permission request routed through the supervisor, its route and its decision, in
SQL for analysis (`aid supervision query`) and for supervisors (`query`).

Requests whose mode involves no supervisor never reach it, so routed requests are all its rulings. `plan` names
a `remove` request's plan in the session's `removals.db` (`aid.removals`); `request_id` joins
`removal_plans.request_id` too.

An `ask_user` question keeps none of its prose here (`QUESTION`): its text, the answer's words and the holders'
notes are free text, and a question once held a password the agent made for the person. `session` and `seq` find
the question in the session's history.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from aid.sql import Database

if TYPE_CHECKING:
    from pathlib import Path

    import apsw

    from aid.protocol import PermissionDecision, PermissionRequest

RULINGS_FILE: Final = "rulings.db"
SCHEMA_VERSION: Final = 1
SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS rulings (
    request_id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    turn TEXT NOT NULL,
    seq INTEGER NOT NULL,
    supervisor TEXT NOT NULL,
    mode TEXT NOT NULL,
    tool_name TEXT,
    title TEXT NOT NULL,
    input TEXT NOT NULL,
    plan INTEGER,
    started REAL NOT NULL,
    ended REAL,
    end TEXT,
    option_id TEXT,
    allowed INTEGER,
    decided_by TEXT,
    decider TEXT,
    said TEXT
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ruling_steps (
    request_id TEXT NOT NULL REFERENCES rulings (request_id),
    step INTEGER NOT NULL,
    holder TEXT NOT NULL,
    entered REAL NOT NULL,
    left REAL,
    outcome TEXT,
    note TEXT,
    PRIMARY KEY (request_id, step)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS rulings_session ON rulings (session, started);
UPDATE ruling_steps SET note = NULL
    WHERE note IS NOT NULL AND request_id IN (SELECT request_id FROM rulings WHERE tool_name = 'ask_user');
UPDATE rulings SET title = '', said = NULL WHERE tool_name = 'ask_user' AND (title != '' OR said IS NOT NULL);
"""
QUESTION: Final = "ask_user"


class RulingEnd(StrEnum):
    DECIDED = "decided"
    GONE = "gone"
    """The request stopped waiting without a decision this plugin saw: its turn ended, or the session went."""


class StepOutcome(StrEnum):
    DECIDED = "decided"
    FORWARDED = "forwarded"
    SILENT = "silent"
    """The holder said nothing within its window; the next one got the request."""
    GONE = "gone"


def _started(
    db: apsw.Connection,
    request: PermissionRequest,
    session: str,
    turn: str,
    seq: int,
    supervisor: str,
    mode: str,
) -> None:
    plan = request.input.get("plan") if request.tool_name == "remove" and isinstance(request.input, dict) else None
    db.execute(
        "INSERT OR IGNORE INTO rulings (request_id, session, turn, seq, supervisor, mode, tool_name, title, input,"
        " plan, started) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request.request_id,
            session,
            turn,
            seq,
            supervisor,
            mode,
            request.tool_name,
            "" if request.tool_name == QUESTION else request.title,
            json.dumps(request.input),
            plan if isinstance(plan, int) else None,
            time.time(),
        ),
    )


def _entered(db: apsw.Connection, request_id: str, step: int, holder: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO ruling_steps (request_id, step, holder, entered) VALUES (?, ?, ?, ?)",
        (request_id, step, holder, time.time()),
    )


def _left(db: apsw.Connection, request_id: str, step: int, outcome: StepOutcome, note: str | None) -> None:
    db.execute(
        "UPDATE ruling_steps SET left = ?, outcome = ?, note = CASE WHEN (SELECT tool_name FROM rulings WHERE"
        " request_id = ?) = ? THEN NULL ELSE ? END WHERE request_id = ? AND step = ? AND left IS NULL",
        (time.time(), outcome, request_id, QUESTION, note, request_id, step),
    )


def _allowed(kind: str | None) -> bool | None:
    """An approval or a refusal; None for an `ask_user` pick or a cancel, which are neither."""
    if kind is None:
        return None
    if kind.startswith("allow"):
        return True
    return False if kind.startswith("reject") else None


def _ended(db: apsw.Connection, request: PermissionRequest, step: int, decision: PermissionDecision | None) -> None:
    with db:
        _left(db, request.request_id, step, StepOutcome.GONE if decision is None else StepOutcome.DECIDED, None)
        if decision is None:
            db.execute(
                "UPDATE rulings SET ended = ?, end = ? WHERE request_id = ? AND ended IS NULL",
                (time.time(), RulingEnd.GONE, request.request_id),
            )
            return
        kind = next((o.kind for o in request.options if o.option_id == decision.option_id), None)
        db.execute(
            "UPDATE rulings SET ended = ?, end = ?, option_id = ?, allowed = ?, decided_by = ?, decider = ?,"
            " said = ? WHERE request_id = ? AND ended IS NULL",
            (
                time.time(),
                RulingEnd.DECIDED,
                decision.option_id,
                _allowed(kind),
                decision.by,
                decision.supervisor or decision.plugin,
                None if request.tool_name == QUESTION else decision.text,
                request.request_id,
            ),
        )


class RulingStore:
    """The supervision plugin's rulings, written by the plugin only."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> RulingStore:
        return cls(await Database.open(path, SCHEMA, SCHEMA_VERSION))

    async def started(
        self, request: PermissionRequest, *, session: str, turn: str, seq: int, supervisor: str, mode: str
    ) -> None:
        await self._db.run(_started, request, session, turn, seq, supervisor, mode)

    async def entered(self, request_id: str, step: int, holder: str) -> None:
        await self._db.run(_entered, request_id, step, holder)

    async def left(self, request_id: str, step: int, outcome: StepOutcome, note: str | None = None) -> None:
        await self._db.run(_left, request_id, step, outcome, note)

    async def ended(self, request: PermissionRequest, step: int, decision: PermissionDecision | None) -> None:
        await self._db.run(_ended, request, step, decision)

    async def aclose(self) -> None:
        await self._db.aclose()
