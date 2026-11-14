"""aid_models stands alone: pyedit and others install it without aid, so nothing in it may import aid."""

from __future__ import annotations

import ast
from pathlib import Path

LIBRARY = Path(__file__).parent.parent / "models" / "src" / "aid_models"


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [node.module]
    return []


def test_aid_models_never_imports_aid() -> None:
    found = [
        f"{path.name}: {name}"
        for path in sorted(LIBRARY.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        for name in _imported(node)
        if name == "aid" or name.startswith("aid.")
    ]
    assert found == []
