"""`@aid.mcptool` functions for the tool tests."""

from __future__ import annotations

import os

import aid
from aid.paths import ENV_SESSION


@aid.mcptool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Only the first paragraph describes the tool in the catalog.
    """
    return a + b


@aid.mcptool(name="whoami", description="The aid session that calls this tool.")
async def session_name() -> str:
    return os.environ.get(ENV_SESSION, "")
