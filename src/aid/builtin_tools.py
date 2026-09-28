"""aid's own tools, offered to every session beside the ones on the agents path."""

from __future__ import annotations

import os
from typing import Any

from aid.client import connect
from aid.paths import ENV_SESSION
from aid.tools import mcptool


@mcptool
async def send_message(to: str, text: str) -> str:
    """Send a message to another aid session, by its name from list_sessions. It wakes that session.

    Returns once the message is delivered. An answer, if the session sends one, arrives later as a message to
    you: do not wait for it.
    """
    async with connect() as client:
        await client.send_message(to, text, sender=os.environ.get(ENV_SESSION))
    return f"delivered to {to}"


@mcptool
async def list_sessions() -> list[dict[str, Any]]:
    """The aid sessions you can message: name, kind, whether its worker is running, and which one is you."""
    me = os.environ.get(ENV_SESSION)
    async with connect() as client:
        sessions = await client.sessions()
    return [{**s.model_dump(mode="json"), "you": s.name == me} for s in sessions]
