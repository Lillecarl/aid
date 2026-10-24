from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aid.protocol import AidError
from aid.sql import CELL_CHARS, Database, ask

if TYPE_CHECKING:
    from pathlib import Path

    import apsw

pytestmark = pytest.mark.anyio

SCHEMA = "CREATE TABLE IF NOT EXISTS things (name TEXT, size INTEGER, data BLOB);"


def _fill(db: apsw.Connection) -> None:
    with db:
        db.executemany(
            "INSERT INTO things VALUES (?, ?, ?)",
            [("a", 1, None), ("b", 2, b"xyz"), ("long", 3, None), ("c", 4, None)],
        )
        db.execute("UPDATE things SET name = ? WHERE name = 'long'", ("x" * (CELL_CHARS + 5),))


async def things(tmp_path: Path) -> Path:
    path = tmp_path / "things.db"
    db = await Database.open(path, SCHEMA, 1)
    try:
        await db.run(_fill)
    finally:
        await db.aclose()
    return path


async def test_a_question_reads_attached_databases(tmp_path: Path) -> None:
    path = await things(tmp_path)
    result = await ask({"one": path, "gone": tmp_path / "missing.db"}, "SELECT name, size, data FROM things LIMIT 2")
    assert result.columns == ["name", "size", "data"]
    assert result.rows == [["a", 1, None], ["b", 2, "<3 bytes>"]]
    assert not result.truncated
    clipped = await ask({"one": path}, "SELECT name FROM one.things WHERE size = 3")
    assert clipped.rows == [[f"{'x' * CELL_CHARS}… 5 more"]]
    capped = await ask({"one": path}, "SELECT size FROM things ORDER BY size", limit=2)
    assert (capped.rows, capped.truncated) == ([[1], [2]], True)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO things VALUES ('z', 9, NULL)",
        "DELETE FROM things",
        "PRAGMA one.journal_mode = DELETE",
        "ATTACH 'other.db' AS other",
        "DETACH one",
        "CREATE TEMP TABLE t (x)",
        "SELECT 1; DELETE FROM things",
    ],
)
async def test_a_question_only_reads(tmp_path: Path, sql: str) -> None:
    path = await things(tmp_path)
    with pytest.raises(AidError):
        await ask({"one": path}, sql)
    assert (await ask({"one": path}, "SELECT count(*) FROM things")).rows == [[4]]


async def test_a_long_question_is_interrupted(tmp_path: Path) -> None:
    with pytest.raises(AidError) as caught:
        await ask(
            {}, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT max(i) FROM n", seconds=0.2
        )
    assert caught.value.code == "query_timeout"
