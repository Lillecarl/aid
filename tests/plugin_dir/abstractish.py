"""Neither abstract subclasses nor re-exports are plugins."""

from __future__ import annotations

from alpha import (  # pyrefly: ignore[missing-import] -- importable once the dir is on the path
    Alpha,  # pyright: ignore[reportMissingImports] -- importable once the dir is on the path
)

import aid

Reexported = Alpha


class Base(aid.Plugin):
    """Abstract: no run, so not a plugin."""
