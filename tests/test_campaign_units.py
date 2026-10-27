from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio
import pytest

from aid.coding import READ_LIMIT
from aid.protocol import AidError, ToolCall
from plugins.campaign.units import READ_LINES, FileInfo, Limits, chunk, examine, tracked, uncovered

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


async def git_repo(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    await anyio.run_process(["git", "init", "-q"], cwd=root, env=GIT_ENV)
    await anyio.run_process(["git", "add", "."], cwd=root, env=GIT_ENV)
    return root


def test_read_lines_is_what_read_returns() -> None:
    assert READ_LINES == READ_LIMIT


@pytest.mark.anyio
async def test_tracked_files_are_what_git_lists(tmp_path: Path) -> None:
    repo = await git_repo(tmp_path / "repo", {"a.py": b"x\n", "sub/b.py": b"y\n", ".gitignore": b"build/\n"})
    (repo / "build").mkdir()
    (repo / "build" / "out.py").write_text("z\n")
    assert sorted(await tracked(repo)) == [".gitignore", "a.py", "sub/b.py"]
    with pytest.raises(AidError) as refused:
        await tracked(tmp_path)
    assert refused.value.code == "not_a_repository"


def test_examine_keeps_text_files_within_limits(tmp_path: Path) -> None:
    (tmp_path / "vendor").mkdir()
    files = {
        "a.py": b"one\ntwo\n",
        "big.txt": b"x" * 200,
        "blob.bin": b"\0\1\2",
        "vendor/lib.js": b"y\n",
        "empty": b"",
        "latin.txt": "caf\u00e9\n".encode("latin-1"),
        "paged.c": b"a\n\x0cb\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "link.py").symlink_to(tmp_path / "a.py")
    limits = Limits(max_file_bytes=100, exclude=["vendor/**"])
    kept, skipped = examine(tmp_path, [*files, "link.py", "gone.py"], limits)
    assert kept == [
        FileInfo(path="a.py", lines=2, size=8),
        FileInfo(path="empty", lines=0, size=0),
        FileInfo(path="paged.c", lines=3, size=5),
    ]
    assert {s.path: s.reason for s in skipped} == {
        "big.txt": "200 bytes",
        "blob.bin": "binary",
        "vendor/lib.js": "excluded",
        "link.py": "not a file",
        "latin.txt": "not UTF-8",
        "gone.py": "not a file",
    }


def test_units_close_at_either_cap_in_path_order() -> None:
    files = [FileInfo(path=f"d/{n}", lines=1, size=size) for n, size in (("c", 10), ("a", 10), ("b", 10), ("e", 90))]
    groups = chunk(files, Limits(unit_files=2, unit_bytes=50))
    assert [[f.path for f in group] for group in groups] == [["d/a", "d/b"], ["d/c"], ["d/e"]]


def call(call_id: str, turn: str = "t1", **fields: JsonValue) -> tuple[str, ToolCall]:
    return turn, ToolCall.model_validate({"tool_call_id": call_id, **fields})


def test_a_file_is_covered_only_when_completed_reads_return_every_line(tmp_path: Path) -> None:
    long = FileInfo(path="long.py", lines=4500, size=1)
    short = FileInfo(path="src/short.py", lines=10, size=1)
    failed = FileInfo(path="failed.py", lines=1, size=1)
    empty = FileInfo(path="empty", lines=0, size=0)
    calls = [
        call("1", tool="read", status="in_progress", input={"path": "./src/../src/short.py"}),
        call("1", status="completed"),
        call("2", tool="read", status="in_progress", input={"path": "long.py"}),
        call("2", status="completed"),
        call("3", tool="read", status="in_progress", input={"path": "long.py", "offset": 2001, "limit": 2000}),
        call("3", status="completed"),
        call("4", tool="read", status="in_progress", input={"path": "failed.py"}),
        call("4", status="failed"),
        call("5", tool="read", status="in_progress", input={"path": str(tmp_path / "empty")}),
        call("5", status="completed"),
        call("6", tool="search", status="completed", input={"path": "long.py"}),
    ]
    assert uncovered(tmp_path, [long, short, failed, empty], calls) == ["long.py", "failed.py"]
    calls += [call("7", tool="read", status="completed", input={"path": "long.py", "offset": 4001, "limit": 600})]
    assert uncovered(tmp_path, [long, short, empty], calls) == []
    outside = [call("8", tool="read", status="completed", input={"path": "../repo2/short.py"})]
    assert uncovered(tmp_path / "x", [short], outside) == ["src/short.py"]


def test_calls_numbered_afresh_each_turn_stay_apart(tmp_path: Path) -> None:
    files = [FileInfo(path="a.py", lines=1, size=1), FileInfo(path="b.py", lines=1, size=1)]
    calls = [
        call("call_0", "t1", tool="read", status="in_progress", input={"path": "b.py"}),
        call("call_0", "t1", status="completed"),
        call("call_0", "t2", tool="read", status="in_progress", input={"path": "a.py"}),
        call("call_0", "t2", status="completed"),
    ]
    assert uncovered(tmp_path, files, calls) == []
