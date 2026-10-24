"""aid's own tools, offered to every session beside the ones on the plugins path."""

from __future__ import annotations

import os
from typing import Any, Literal

from aid.client import connect
from aid.paths import ENV_SESSION
from aid.protocol import Delivery, RequestDirectory
from aid.tools import mcptool


@mcptool
async def send_message(to: str, text: str, delivery: Literal["interrupt", "wait_for_turn"] = "interrupt") -> str:
    """Send a message to another aid session, by its name from list_sessions. It wakes that session.

    delivery: "interrupt" reaches a busy session between the steps of its running turn, where its kind allows
    (else after the turn); "wait_for_turn" waits until that turn ends. Interrupt for what changes the work in
    progress, wait_for_turn for what can wait.

    Returns once the message is delivered. An answer, if the session sends one, arrives later as a message to
    you: do not wait for it.
    """
    async with connect() as client:
        await client.send_message(to, text, sender=os.environ.get(ENV_SESSION), delivery=Delivery(delivery))
    return f"delivered to {to}"


@mcptool
async def change_directory(path: str, reason: str) -> str:
    """Ask to move this session to another working directory: absolute, or relative to the current one. A person
    approves or declines it, whatever your permissions allow. An approved move happens when this turn ends and
    restarts you there with your conversation kept; either way the answer arrives as a message from aid. Say in
    `reason` why the work needs the other directory."""
    session = os.environ.get(ENV_SESSION)
    if not session:
        return "not in an aid session: there is no directory to change"
    async with connect() as client:
        result = await client.call(RequestDirectory(session=session, cwd=path, reason=reason))
    target = result.get("cwd") if isinstance(result, dict) else path
    return f"asked to move to {target}; a person decides, and the answer arrives as a message"


@mcptool
async def list_sessions() -> list[dict[str, Any]]:
    """The aid sessions you can message: name, kind, whether its worker is running, and which one is you."""
    me = os.environ.get(ENV_SESSION)
    async with connect() as client:
        sessions = await client.sessions()
    return [{**s.model_dump(mode="json"), "you": s.name == me} for s in sessions]
