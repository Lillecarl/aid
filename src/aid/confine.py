"""Paths a session may reach: under its directory, however they are spelled. Free of web imports, for workers."""

from __future__ import annotations

from pathlib import Path

from aid.protocol import AidError


def inside(root: str | Path, rel: str) -> Path:
    """`rel` under `root`, resolved. AidError `outside` if it resolves anywhere else."""
    base = Path(root).resolve()
    target = (base / rel.lstrip("/")).resolve()
    if target != base and not target.is_relative_to(base):
        raise AidError("outside", f"{rel!r} is outside the session's directory")
    return target
