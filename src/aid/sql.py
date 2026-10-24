"""SQLite through apsw: databases one owner writes off the event loop, and read-only questions over them.

apsw carries its own SQLite (`nix/apsw.nix`). A writer keeps one connection and runs each call in a worker
thread, one at a time. A question opens a private in-memory connection, attaches each database read-only, and
runs under an authorizer that admits reading alone: `mode=ro` stops writes to a file, not ATTACH or PRAGMA.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import apsw
import apsw.ext

from aid.protocol import AidError, QueryResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Concatenate

    from pydantic import JsonValue

ROWS: Final = 200
"""Rows a question answers; it says when more were left."""
CELL_CHARS: Final = 2000
"""Characters of one text value a question answers; longer ones clip with their excess counted."""
SECONDS: Final = 5.0
"""How long a question may run before SQLite interrupts it."""
PROGRESS_STEPS: Final = 1000
"""Virtual machine steps between deadline checks."""
READING: Final = frozenset({apsw.SQLITE_SELECT, apsw.SQLITE_READ, apsw.SQLITE_FUNCTION, apsw.SQLITE_RECURSIVE})


def _create(db: apsw.Connection, schema: str, version: int) -> None:
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA busy_timeout = 5000")
    with db:
        db.execute(schema)
        db.execute(f"PRAGMA user_version = {version}")


class Database:
    """A database one owner writes: one connection, each call in a worker thread, one at a time."""

    def __init__(self, db: apsw.Connection) -> None:
        self._db = db
        self._limiter = anyio.CapacityLimiter(1)

    @classmethod
    async def open(cls, path: Path, schema: str, version: int) -> Database:
        """Open or create `path` in WAL mode, so readers on other connections never wait on the writer."""
        await anyio.Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        database = cls(await anyio.to_thread.run_sync(apsw.Connection, str(path)))
        try:
            await database.run(_create, schema, version)
        except BaseException:
            await database.aclose()
            raise
        return database

    async def run[**P, T](
        self, call: Callable[Concatenate[apsw.Connection, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return await anyio.to_thread.run_sync(lambda: call(self._db, *args, **kwargs), limiter=self._limiter)

    async def aclose(self) -> None:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(self._db.close, limiter=self._limiter)


def _authorize(op: int, _p1: str | None, _p2: str | None, _db: str | None, _view: str | None) -> int:
    return apsw.SQLITE_OK if op in READING else apsw.SQLITE_DENY


def _value(value: apsw.SQLiteValue) -> JsonValue:
    match value:
        case str() if len(value) > CELL_CHARS:
            return f"{value[:CELL_CHARS]}… {len(value) - CELL_CHARS} more"
        case None | int() | float() | str():
            return value
        case _:
            return f"<{len(memoryview(value))} bytes>"


def _ask(attach: dict[str, Path], sql: str, limit: int, seconds: float) -> QueryResult:
    db = apsw.Connection(":memory:", flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_URI)
    try:
        for name, path in attach.items():
            if path.exists():
                db.execute("ATTACH ? AS ?", (f"{path.as_uri()}?mode=ro", name))
        db.authorizer = _authorize
        deadline = time.monotonic() + seconds
        db.set_progress_handler(lambda: time.monotonic() > deadline, PROGRESS_STEPS)
        info = apsw.ext.query_info(db, sql)
        if info.query_remaining and info.query_remaining.strip():
            raise AidError("one_statement", "ask one statement at a time")
        rows: list[list[JsonValue]] = []
        for row in db.execute(sql):
            if len(rows) == limit:
                return QueryResult(columns=[c for c, _ in info.description or ()], rows=rows, truncated=True)
            rows.append([_value(v) for v in row])
        return QueryResult(columns=[c for c, _ in info.description or ()], rows=rows, truncated=False)
    except apsw.InterruptError:
        raise AidError("query_timeout", f"the question ran past {seconds:g} seconds") from None
    except apsw.Error as error:
        raise AidError("query_failed", str(error)) from None
    finally:
        db.close()


async def ask(attach: dict[str, Path], sql: str, *, limit: int = ROWS, seconds: float = SECONDS) -> QueryResult:
    """Answer one read-only statement over the databases in `attach` (schema name → file; a missing file is
    left out), up to `limit` rows, interrupted after `seconds`."""
    return await anyio.to_thread.run_sync(_ask, attach, sql, limit, seconds)
