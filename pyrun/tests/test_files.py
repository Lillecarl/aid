from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pyrun.files import FileAccess, FileGuard

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_an_import_through_a_linked_venv_reads_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Nix venv's package files link on into other store paths: anyio 4.15 imports `_fileio` lazily, inside a
    script, through such a link."""
    store = tmp_path / "store"
    store.mkdir()
    (store / "module.py").write_text("")
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "module.py").symlink_to(store / "module.py")
    monkeypatch.setattr(sys, "path", [str(venv)])
    asked: list[FileAccess] = []

    def ask(access: FileAccess) -> tuple[bool, str | None]:
        asked.append(access)
        return True, None

    guard = FileGuard([], ask)
    guard("open", (str(venv / "module.py"), "r", 0))
    assert asked == []
    guard("open", (str(store / "module.py"), "r", 0))
    assert [a.path for a in asked] == [str(store / "module.py")]
