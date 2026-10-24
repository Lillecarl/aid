"""`aid guard`: a plugin that answers permission requests that plainly only read, and leaves the rest for a
person, by `aid.readonly`'s rules. A seed for Lillecarl/aid#3: rules, not a model.

    aid plugin add guard --grant read --grant permissions
    aid guard

It follows the session list; a session with requests waiting has its status read, and each request judged once. A
request it allows shows in the history as answered by plugin guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import structlog

import aid
from aid.protocol import AidError
from aid.readonly import reads

if TYPE_CHECKING:
    from aid.client import Client
    from aid.protocol import PermissionRequest

log = structlog.get_logger(__name__)

ALLOW_KIND: Final = "allow_once"


def verdict(request: PermissionRequest) -> str | None:
    """The option to answer `request` with, or None to leave it for a person."""
    if not reads(request.tool_name, request.input):
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
