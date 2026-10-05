"""`@aid.mcptool`: a function in a module on the agents path, offered to every session as a tool.

    import aid

    @aid.mcptool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

ACP and interactive Claude sessions reach the tools through `python -m aid.mcp_server`, an MCP server named
`aid`; pydantic-ai sessions call them in-process. The input schema comes from the type hints, so a parameter
type imported only under `TYPE_CHECKING` cannot resolve. The function may be sync or async.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast, overload

if TYPE_CHECKING:
    from collections.abc import Callable

_MARKER: Final = "__aid_mcptool__"


@dataclass(frozen=True)
class McpTool:
    fn: Callable[..., Any]
    name: str
    description: str | None
    """None uses the docstring."""

    def summary(self) -> str:
        doc = self.description or inspect.getdoc(self.fn) or ""
        return doc.split("\n\n")[0].replace("\n", " ")


@overload
def mcptool[F: Callable[..., Any]](fn: F, /) -> F: ...
@overload
def mcptool[F: Callable[..., Any]](*, name: str | None = None, description: str | None = None) -> Callable[[F], F]: ...
def mcptool[F: Callable[..., Any]](
    fn: F | None = None, /, *, name: str | None = None, description: str | None = None
) -> F | Callable[[F], F]:
    """Mark `fn` as a tool, named after the function unless `name` is given."""

    def mark(f: F) -> F:
        setattr(f, _MARKER, McpTool(f, name or f.__name__, description))
        return f

    return mark if fn is None else mark(fn)


def tool_of(obj: object) -> McpTool | None:
    tool = getattr(obj, _MARKER, None)
    return tool if isinstance(tool, McpTool) else None


#: Argument names that carry a file a call touches, across Claude Code's tools, common MCP servers and
#: aid.coding: the web UI's gist() in web/src/lib/tools.ts reads the same names.
PATH_KEYS: Final = ("file_path", "path", "notebook_path")


def tool_paths(input: object) -> list[str]:
    """Files a call touches, from its arguments."""
    if not isinstance(input, dict):
        return []
    arguments = cast("dict[str, Any]", input)
    return [p for key in PATH_KEYS if isinstance(p := arguments.get(key), str)]
