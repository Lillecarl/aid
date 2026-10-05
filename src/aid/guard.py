"""`aid guard`: a plugin that answers permission requests that plainly only read, and leaves the rest for a
person. A seed for Lillecarl/aid#3: rules, not a model.

    aid plugin add guard --grant read --grant permissions
    aid guard

It follows the session list; a session with requests waiting has its status read, and each request judged once. A
request it allows shows in the history as answered by plugin guard.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Final, cast

import structlog

import aid
from aid.protocol import AidError

if TYPE_CHECKING:
    from pydantic import JsonValue

    from aid.client import Client
    from aid.protocol import PermissionRequest

log = structlog.get_logger(__name__)

READERS: Final = frozenset({"ls", "cat", "head", "tail", "wc", "pwd", "rg", "grep", "stat", "file", "du", "df", "tree"})
"""Programs that read and print, whatever their arguments. Not `find` (-delete, -exec), `sed` (-i) or `git`."""
SHELL_SYNTAX: Final = frozenset(";|&$`<>(){}\n\\")
"""A command string holding any of these is more than one program with its arguments."""
ALLOW_KIND: Final = "allow_once"


def argv_of(tool_input: JsonValue) -> list[str] | None:
    """The command a tool call runs: pyrun's `argv`, or a shell `command` with no shell syntax in it."""
    if not isinstance(tool_input, dict):
        return None
    fields = cast("dict[str, JsonValue]", tool_input)
    argv = fields.get("argv")
    if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv):
        return cast("list[str]", argv)
    command = fields.get("command")
    if isinstance(command, str) and command.strip() and not SHELL_SYNTAX & set(command):
        try:
            return shlex.split(command)
        except ValueError:
            return None
    return None


def verdict(request: PermissionRequest) -> str | None:
    """The option to answer `request` with, or None to leave it for a person."""
    argv = argv_of(request.input)
    if argv is None or argv[0] not in READERS:
        return None
    return next((o.option_id for o in request.options if o.kind == ALLOW_KIND), None)


async def guard(client: Client) -> None:
    judged: set[str] = set()
    async for sessions in client.follow_sessions():
        for info in sessions:
            if not info.permissions:
                continue
            try:
                status = await client.session(info.name).status()
            except AidError:
                continue  # Gone since the list was published.
            for request in status.permissions:
                if request.request_id in judged:
                    continue
                judged.add(request.request_id)
                if (option := verdict(request)) is None:
                    log.info("permission_deferred", plugin=info.name, title=request.title)
                    continue
                try:
                    await client.session(info.name).answer(request.request_id, option)
                    log.info("permission_allowed", plugin=info.name, title=request.title)
                except AidError as error:
                    log.info("permission_raced", plugin=info.name, title=request.title, error=error.message)


async def main(plugin: str) -> None:
    async with aid.connect(plugin=plugin) as client:
        await guard(client)
