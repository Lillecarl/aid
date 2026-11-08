from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import apsw
import pytest

from aid.protocol import AidError
from aid.sql import Database, ask
from aid.vectors import ENV_SQLITE_VEC, K_LIMIT, Metric, Vectors, VectorTable, drop, load

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from aid.vectors import ColumnType

pytestmark = pytest.mark.skipif(not os.environ.get(ENV_SQLITE_VEC), reason=f"{ENV_SQLITE_VEC} is not set")

A = {"kind": "a", "turn": 1, "kept": True}
NOTES = VectorTable("notes", 3, "test/embed", columns={"kind": "text", "turn": "integer", "kept": "boolean"})


@asynccontextmanager
async def opened(path: Path) -> AsyncGenerator[Database]:
    database = await Database.open(
        path, "CREATE TABLE IF NOT EXISTS notes_text (id INTEGER PRIMARY KEY)", 1, setup=load
    )
    try:
        yield database
    finally:
        await database.aclose()


async def test_nearest_rows_come_closest_first_and_filtered(tmp_path: Path) -> None:
    async with opened(tmp_path / "v.db") as database:
        vectors = await Vectors.open(database, NOTES)
        await vectors.add(1, [1, 0, 0], {"kind": "a", "turn": 1, "kept": True})
        await vectors.add(2, [0, 1, 0], {"kind": "b", "turn": 2, "kept": False})
        await vectors.add(3, [0.9, 0.1, 0], {"kind": "a", "turn": 3, "kept": False})

        found = await vectors.nearest([1, 0, 0], k=3)
        assert [n.rowid for n in found] == [1, 3, 2]
        assert found[0].distance == pytest.approx(0.0, abs=1e-6)
        assert found[0].metadata == {"kind": "a", "turn": 1, "kept": True}

        assert [n.rowid for n in await vectors.nearest([0, 1, 0], k=5, where={"kind": "a"})] == [3, 1]
        assert [n.rowid for n in await vectors.nearest([1, 0, 0], where={"kept": False})] == [3, 2]


async def test_add_replaces_a_row_and_remove_drops_it(tmp_path: Path) -> None:
    async with opened(tmp_path / "v.db") as database:
        vectors = await Vectors.open(database, NOTES)
        await vectors.add(1, [1, 0, 0], A)
        await vectors.add(1, [0, 1, 0], A | {"kind": "b"})
        [only] = await vectors.nearest([0, 1, 0])
        assert (only.rowid, only.metadata["kind"]) == (1, "b")
        await vectors.remove(1)
        assert await vectors.nearest([0, 1, 0]) == []


async def test_rows_survive_a_reopen(tmp_path: Path) -> None:
    async with opened(tmp_path / "v.db") as database:
        await (await Vectors.open(database, NOTES)).add(7, [0, 0, 1], A)
    async with opened(tmp_path / "v.db") as database:
        [found] = await (await Vectors.open(database, NOTES)).nearest([0, 0, 1])
        assert found.rowid == 7


@pytest.mark.parametrize(
    "changed",
    [
        VectorTable("notes", 3, "test/other", columns=NOTES.columns),
        VectorTable("notes", 4, "test/embed", columns=NOTES.columns),
        VectorTable("notes", 3, "test/embed", Metric.L2, columns=NOTES.columns),
        VectorTable("notes", 3, "test/embed"),
    ],
)
async def test_a_table_from_another_model_is_refused_until_dropped(tmp_path: Path, changed: VectorTable) -> None:
    async with opened(tmp_path / "v.db") as database:
        await (await Vectors.open(database, NOTES)).add(1, [1, 0, 0], A)
        with pytest.raises(AidError) as refused:
            await Vectors.open(database, changed)
        assert refused.value.code == "vector_table_changed"
        await drop(database, "notes")
        vectors = await Vectors.open(database, changed)
        assert await vectors.nearest([1.0] * changed.dimensions) == []


async def test_bad_input_is_refused_by_name(tmp_path: Path) -> None:
    async with opened(tmp_path / "v.db") as database:
        vectors = await Vectors.open(database, NOTES)
        for call, code in [
            (vectors.add(1, [1, 0], A), "bad_vector"),
            (vectors.add(1, [1, 0, float("nan")], A), "bad_vector"),
            (vectors.add(1, [1, 0, 0], A | {"size": 1}), "bad_column"),
            (vectors.add(1, [1, 0, 0], {"kind": "a"}), "bad_column"),
            (vectors.nearest([1, 0, 0], k=K_LIMIT + 1), "bad_k"),
            (vectors.nearest([1, 0, 0], where={"nope": 1}), "bad_column"),
        ]:
            with pytest.raises(AidError) as refused:
                await call
            assert refused.value.code == code
    named: list[tuple[str, dict[str, ColumnType]]] = [
        ("drop table", {}),
        ("notes", {"embedding": "text"}),
        ("notes", {"k": "text"}),
    ]
    for name, columns in named:
        with pytest.raises(AidError) as refused:
            VectorTable(name, 3, "test/embed", columns=columns)
        assert refused.value.code == "bad_name"


async def test_without_the_extension_nothing_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_SQLITE_VEC)
    with pytest.raises(AidError) as refused:
        await Database.open(tmp_path / "v.db", "", 1, setup=load)
    assert refused.value.code == "no_vectors"


async def test_a_question_cannot_load_an_extension(tmp_path: Path) -> None:
    async with opened(tmp_path / "v.db") as database:
        await Vectors.open(database, NOTES)
        with pytest.raises(apsw.Error, match="not authorized"):
            await database.run(lambda db: db.execute("SELECT load_extension(?)", (os.environ[ENV_SQLITE_VEC],)))
    with pytest.raises(AidError) as refused:
        await ask({"v": tmp_path / "v.db"}, f"SELECT load_extension('{os.environ[ENV_SQLITE_VEC]}')")
    assert refused.value.code == "query_failed"
