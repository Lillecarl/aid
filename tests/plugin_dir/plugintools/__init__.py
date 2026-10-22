"""Tools a plugin module offers to agent sessions."""

from __future__ import annotations

import aid


@aid.mcptool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@aid.mcptool
def whoami() -> str:
    """The plugin tool that names itself."""
    return "plugin tool"
