"""aid_models and aid_agent stand alone: pyedit, pymux and others install them without aid, so nothing in them may
import aid."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [node.module]
    return []


@pytest.mark.parametrize("library", ["models/src/aid_models", "agent/src/aid_agent"])
def test_library_never_imports_aid(library: str) -> None:
    found = [
        f"{path.name}: {name}"
        for path in sorted((ROOT / library).rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        for name in _imported(node)
        if name == "aid" or name.startswith("aid.")
    ]
    assert found == []
