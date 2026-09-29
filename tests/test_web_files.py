from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aid.protocol import AidError
from aid.web import files

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


async def test_directories_come_first(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("bb")
    (tmp_path / "A.txt").write_text("a")
    (tmp_path / "src").mkdir()
    listed = await files.list_dir(str(tmp_path), "")
    assert [(e.name, e.dir, e.size) for e in listed] == [("src", True, None), ("A.txt", False, 1), ("b.txt", False, 2)]


@pytest.mark.parametrize("rel", ["..", "../x", "src/../../x", "/etc", "out"])
async def test_nothing_outside_the_directory(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "out").symlink_to(tmp_path)
    with pytest.raises(AidError) as error:
        await files.list_dir(str(root), rel)
    # "/etc" is taken as relative to the root, where it does not exist.
    assert error.value.code == ("not_found" if rel == "/etc" else "outside")


async def test_a_file_reads_as_text(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 'é'\n")
    view = await files.read_file(str(tmp_path), "a.py")
    assert (view.text, view.size, view.truncated) == ("x = 'é'\n", 9, False)


async def test_a_binary_file_has_no_text(tmp_path: Path) -> None:
    (tmp_path / "b.bin").write_bytes(b"\x89PNG\0\0")
    view = await files.read_file(str(tmp_path), "b.bin")
    assert (view.text, view.size) == (None, 6)


async def test_a_cut_through_a_character_still_reads(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_bytes(b"a" * (files.MAX_FILE - 1) + "é".encode())
    view = await files.read_file(str(tmp_path), "big.txt")
    assert view.truncated
    assert view.text == "a" * (files.MAX_FILE - 1)


async def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(AidError) as error:
        await files.read_file(str(tmp_path), "")
    assert error.value.code == "not_a_file"
