"""Vectors beside the data they index: sqlite-vec's `vec0` tables in an `aid.sql.Database` (#60).

sqlite-vec (`AID_SQLITE_VEC`, the loadable extension's path without its suffix) loads into apsw's own SQLite.
0.1.9 has no approximate index: `nearest` compares the query with every row the filter leaves, which suits
thousands to about a million rows a table. Each table records its embedding model and dimensions, so vectors
from another model never mix with its rows.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

import apsw

from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aid.sql import Database

ENV_SQLITE_VEC: Final = "AID_SQLITE_VEC"
K_LIMIT: Final = 4096
"""The most neighbours one vec0 query answers (sqlite-vec 0.1.9, measured)."""
VECTOR: Final = "embedding"
NAME: Final = re.compile(r"[a-z_][a-z0-9_]{0,62}")
RESERVED: Final = frozenset({VECTOR, "rowid", "distance", "k"})
REGISTRY: Final = """CREATE TABLE IF NOT EXISTS vector_tables (
    name TEXT PRIMARY KEY,
    dimensions INTEGER NOT NULL,
    metric TEXT NOT NULL,
    model TEXT NOT NULL,
    columns TEXT NOT NULL
)"""

type Value = str | int | float | bool
type ColumnType = Literal["text", "integer", "float", "boolean"]


class Metric(StrEnum):
    COSINE = "cosine"
    L2 = "l2"
    L1 = "l1"


@dataclass(frozen=True)
class VectorTable:
    """A vec0 table: `dimensions` floats a row from `model`, compared by `metric`; `columns` are metadata a
    search filters on by equality."""

    name: str
    dimensions: int
    model: str
    metric: Metric = Metric.COSINE
    columns: Mapping[str, ColumnType] = field(default_factory=dict[str, "ColumnType"])

    def __post_init__(self) -> None:
        for name in (self.name, *self.columns):
            if not NAME.fullmatch(name) or name in RESERVED:
                raise AidError("bad_name", f"{name!r} is not a vector table or column name")
        if not 0 < self.dimensions <= 8192:
            raise AidError("bad_dimensions", f"{self.dimensions} dimensions")
        if not self.model:
            raise AidError("bad_model", "a vector table names its embedding model")

    def record(self) -> tuple[int, str, str, str]:
        return self.dimensions, self.metric.value, self.model, json.dumps(dict(self.columns), sort_keys=True)

    def ddl(self) -> str:
        columns = "".join(f", {name} {kind}" for name, kind in self.columns.items())
        return (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.name} USING vec0("
            f"{VECTOR} float[{self.dimensions}] distance_metric={self.metric.value}{columns})"
        )


@dataclass(frozen=True)
class Neighbor:
    rowid: int
    distance: float
    metadata: dict[str, Value]


def load(db: apsw.Connection) -> None:
    """Load sqlite-vec into `db`: `aid.sql.Database.open(..., setup=load)`. Only the C API may load: the SQL
    `load_extension()` stays off, as model-written questions reach SQLite."""
    if not (path := os.environ.get(ENV_SQLITE_VEC)):
        raise AidError("no_vectors", f"{ENV_SQLITE_VEC} names no sqlite-vec extension")
    db.config(apsw.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1)
    try:
        db.load_extension(path)
    finally:
        db.config(apsw.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 0)


def _pack(table: VectorTable, vector: Sequence[float]) -> bytes:
    if len(vector) != table.dimensions:
        raise AidError("bad_vector", f"{len(vector)} dimensions, {table.name} holds {table.dimensions}")
    if not all(math.isfinite(x) for x in vector):
        raise AidError("bad_vector", "a vector holds only finite numbers")
    return struct.pack(f"{table.dimensions}f", *vector)


def _columns(table: VectorTable, values: Mapping[str, Value]) -> list[str]:
    if unknown := sorted(set(values) - set(table.columns)):
        raise AidError("bad_column", f"{table.name} has no column {', '.join(unknown)}")
    return list(values)


def _declare(db: apsw.Connection, table: VectorTable) -> None:
    with db:
        db.execute(REGISTRY)
        found = db.execute(
            "SELECT dimensions, metric, model, columns FROM vector_tables WHERE name = ?", (table.name,)
        ).fetchone()
        if found is None:
            db.execute("INSERT INTO vector_tables VALUES (?, ?, ?, ?, ?)", (table.name, *table.record()))
        elif tuple(found) != table.record():
            raise AidError(
                "vector_table_changed",
                f"{table.name} holds {found[0]} dimensions of {found[2]} by {found[1]}, columns {found[3]}: "
                "drop it to rebuild",
            )
        db.execute(table.ddl())


def _add(db: apsw.Connection, table: VectorTable, rowid: int, vector: bytes, metadata: Mapping[str, Value]) -> None:
    names = _columns(table, metadata)
    # vec0 refuses NULL in a metadata column (0.1.9, measured).
    if missing := sorted(set(table.columns) - set(names)):
        raise AidError("bad_column", f"{table.name} needs {', '.join(missing)}")
    columns = "".join(f", {name}" for name in names)
    marks = ", ?" * len(names)
    with db:
        # vec0 refuses a known rowid, OR REPLACE too (0.1.9, measured).
        db.execute(f"DELETE FROM {table.name} WHERE rowid = ?", (rowid,))
        db.execute(
            f"INSERT INTO {table.name}(rowid, {VECTOR}{columns}) VALUES (?, ?{marks})",
            (rowid, vector, *metadata.values()),
        )


def _nearest(
    db: apsw.Connection, table: VectorTable, vector: bytes, k: int, where: Mapping[str, Value]
) -> list[Neighbor]:
    names = _columns(table, where)
    filters = "".join(f" AND {name} = ?" for name in names)
    shown = list(table.columns)
    columns = "".join(f", {name}" for name in shown)
    rows = db.execute(
        f"SELECT rowid, distance{columns} FROM {table.name} WHERE {VECTOR} MATCH ? AND k = ?{filters} "
        "ORDER BY distance",
        (vector, k, *where.values()),
    )
    found: list[Neighbor] = []
    for rowid, distance, *values in rows:
        assert isinstance(rowid, int) and isinstance(distance, float)
        metadata: dict[str, Value] = {}
        for name, value in zip(shown, values, strict=True):
            assert isinstance(value, str | int | float)
            metadata[name] = bool(value) if table.columns[name] == "boolean" else value
        found.append(Neighbor(rowid, distance, metadata))
    return found


def _drop(db: apsw.Connection, name: str) -> None:
    with db:
        db.execute(REGISTRY)
        db.execute(f"DROP TABLE IF EXISTS {name}")
        db.execute("DELETE FROM vector_tables WHERE name = ?", (name,))


class Vectors:
    """One vec0 table in a database its owner writes, opened with `setup=load`."""

    def __init__(self, database: Database, table: VectorTable) -> None:
        self._database = database
        self.table = table

    @classmethod
    async def open(cls, database: Database, table: VectorTable) -> Vectors:
        """Create `table`, or open it as it is: one recorded with another model, dimensions, metric or columns is
        refused (`vector_table_changed`); `drop` it and add the rows again."""
        await database.run(_declare, table)
        return cls(database, table)

    async def add(self, rowid: int, vector: Sequence[float], metadata: Mapping[str, Value] | None = None) -> None:
        """Store `vector` as `rowid` with a value for every column, replacing what that rowid held."""
        await self._database.run(_add, self.table, rowid, _pack(self.table, vector), metadata or {})

    async def remove(self, rowid: int) -> None:
        await self._database.run(lambda db: db.execute(f"DELETE FROM {self.table.name} WHERE rowid = ?", (rowid,)))

    async def nearest(
        self, vector: Sequence[float], k: int = 10, where: Mapping[str, Value] | None = None
    ) -> list[Neighbor]:
        """The `k` rows closest to `vector` among those whose columns equal `where`, closest first."""
        if not 0 < k <= K_LIMIT:
            raise AidError("bad_k", f"k is 1 to {K_LIMIT}, not {k}")
        return await self._database.run(_nearest, self.table, _pack(self.table, vector), k, where or {})


async def drop(database: Database, name: str) -> None:
    """Remove the vector table `name` and its record, for a rebuild with another model."""
    if not NAME.fullmatch(name) or name in RESERVED:
        raise AidError("bad_name", f"{name!r} is not a vector table name")
    await database.run(_drop, name)
