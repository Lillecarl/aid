"""The future-annotations policy (Lillecarl/aid#26): every aid file carries
`from __future__ import annotations`, and no type is quoted.

Ruff enforces the annotation half in the dev shell (UP037); these tests enforce the whole ban in the
sandbox suite, which runs pytest but not ruff. UP037 never visits `cast("T", ...)` first arguments, so
the cast check below covers those. Files are read, never imported: fixtures like
`agent_dir/_private.py` raise on import.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def aid_files() -> list[Path]:
    """Every Python file aid ships or tests, including the vivarium scripts mkTest type-checks."""
    return sorted(
        (*((ROOT / "src").rglob("*.py")), *((ROOT / "tests").rglob("*.py")), *((ROOT / "plugins").rglob("*.py")))
    )


def has_future_annotations(path: Path) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in ast.parse(path.read_text()).body
    )


def test_every_file_has_future_annotations() -> None:
    missing = [str(path.relative_to(ROOT)) for path in aid_files() if not has_future_annotations(path)]
    assert not missing, f"files without `from __future__ import annotations`: {missing}"


def has_quoted(node: ast.expr, in_literal: bool = False) -> bool:
    """Whether a type position holds a quoted annotation. String call arguments are values, not types:
    `Field(pattern=r"...")` and `Field(discriminator="op")` are runtime data, so calls are opaque except
    for `Annotated`, whose first argument is a type. `Literal[...]` contents are values too."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and not in_literal
    if isinstance(node, ast.Subscript):
        value = node.value
        literal = (isinstance(value, ast.Name) and value.id == "Literal") or (
            isinstance(value, ast.Attribute) and value.attr == "Literal"
        )
        return has_quoted(node.value, in_literal) or has_quoted(node.slice, in_literal or literal)
    if isinstance(node, ast.Call):
        func = node.func
        annotated = (isinstance(func, ast.Name) and func.id == "Annotated") or (
            isinstance(func, ast.Attribute) and func.attr == "Annotated"
        )
        return bool(node.args) and annotated and has_quoted(node.args[0], in_literal)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(has_quoted(element, in_literal) for element in node.elts)
    if isinstance(node, ast.BinOp):
        return has_quoted(node.left, in_literal) or has_quoted(node.right, in_literal)
    return False


def annotation_nodes(tree: ast.Module) -> list[ast.expr]:
    """Every annotation position: variable and argument annotations, returns, and type aliases."""
    nodes: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            nodes.append(node.annotation)
        elif isinstance(node, ast.arg):
            if node.annotation is not None:
                nodes.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                nodes.append(node.returns)
        elif isinstance(node, ast.TypeAlias):
            nodes.append(node.value)
    return nodes


def test_no_quoted_annotations() -> None:
    quoted: list[str] = []
    for path in aid_files():
        tree = ast.parse(path.read_text())
        for annotation in annotation_nodes(tree):
            if has_quoted(annotation):
                quoted.append(f"{path.relative_to(ROOT)}:{annotation.lineno}:{annotation.col_offset}")
    assert not quoted, f"quoted annotations (unquote them; UP037): {quoted}"


def cast_strings(tree: ast.Module) -> list[ast.Constant]:
    """`cast("T", ...)` first arguments: a runtime string, not an annotation, so UP037 never visits it."""
    found: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        is_cast = (isinstance(func, ast.Name) and func.id == "cast") or (
            isinstance(func, ast.Attribute) and func.attr == "cast"
        )
        if not is_cast:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append(first)
    return found


def test_no_quoted_cast_types() -> None:
    quoted: list[str] = []
    for path in aid_files():
        tree = ast.parse(path.read_text())
        for string in cast_strings(tree):
            quoted.append(f"{path.relative_to(ROOT)}:{string.lineno}:{string.col_offset}")
    assert not quoted, f"quoted cast types (unquote them): {quoted}"
