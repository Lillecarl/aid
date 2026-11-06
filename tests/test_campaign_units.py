from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio
import pytest

from aid.coding import numbered as aid_numbered
from aid.protocol import AidError
from plugins.campaign.units import FileInfo, Limits, chunk, examine, load, numbered, tracked, unit_prompt

if TYPE_CHECKING:
    from pathlib import Path


GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


async def git_repo(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    await anyio.run_process(["git", "init", "-q"], cwd=root, env=GIT_ENV)
    await anyio.run_process(["git", "add", "."], cwd=root, env=GIT_ENV)
    return root


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


def test_numbered_is_what_read_returns() -> None:
    text = "one\n\ttwo\r\nthree\fx\n"
    assert numbered(text) == aid_numbered(text, 1, 100)


def test_a_unit_prompt_carries_every_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    (tmp_path / "b.py").write_bytes(b"caf\xe9\n")
    files = [FileInfo(path="a.py", lines=2, size=12), FileInfo(path="b.py", lines=1, size=5)]
    prompt = unit_prompt("A brief.", load(tmp_path, files))
    assert prompt.startswith("Repository brief:\nA brief.\n\n")
    assert '<file path="a.py" lines="2">\n     1\tx = 1\n     2\ty = 2\n</file>' in prompt
    assert '<file path="b.py" lines="1">\n     1\tcaf\ufffd\n</file>' in prompt
