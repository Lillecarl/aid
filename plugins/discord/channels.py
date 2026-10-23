"""Session channels: one guild text channel per aid session, under one category.

The map (session to channel id, plus the category id) persists in the plugin dir: channel names are
slugs, and slugs can collide after lowering, so the id map — never the name — is the source of truth.
Reconcile runs at startup and whenever the session list changes: it creates channels for new sessions
and deletes mapped channels whose sessions are gone. Mapped channels only — a stranger's channel is
never touched. A deleted session channel heals the same way: its map entry drops on CHANNEL_DELETE and
the next reconcile recreates it.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final

import anyio
import structlog

from aid.client import daemon_call
from aid.protocol import AidError

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from aid.client import Client
    from aid.protocol import SessionInfo

    from .rest import Rest

log = structlog.get_logger(__name__)

CHANNELS_FILE: Final = "channels.json"
"""The session-to-channel map beside the plugin's registration."""


def slug(name: str) -> str:
    """A session name as a Discord channel name: lowercase, dots and the rest to dashes."""
    return re.sub(r"[^a-z0-9_-]", "-", name.lower()) or "session"


def unique_slug(base: str, taken: set[str]) -> str:
    """`base` free, else suffixed until free: lowering collides (`A.B` and `a-b`), suffixes do not."""
    if base not in taken:
        return base
    number = 2
    while f"{base}-{number}" in taken:
        number += 1
    return f"{base}-{number}"


def state_word(info: SessionInfo) -> str:
    """A session's one-word state, for channel topics and lists: stopped beats idle, busy beats both."""
    if not info.running:
        return "stopped"
    return "busy" if info.working else "idle"


class ChannelMap:
    """Session names to Discord channel ids, plus the category id. Kept on disk so a restart neither
    recreates channels nor orphans them."""

    def __init__(self, path: Path, category_id: str | None = None, channels: dict[str, str] | None = None) -> None:
        self._path = path
        self.category_id = category_id
        self.channels: dict[str, str] = channels if channels is not None else {}

    @classmethod
    async def load(cls, path: Path) -> ChannelMap:
        try:
            raw: JsonValue = json.loads(await anyio.Path(path).read_text())
        except FileNotFoundError:
            return cls(path)
        except ValueError:
            raise AidError("bad_channels", f"{path} holds no channel map") from None
        if not isinstance(raw, dict):
            raise AidError("bad_channels", f"{path} holds no channel map")
        category = raw.get("category")
        channels = raw.get("channels", {})
        if (category is not None and not isinstance(category, str)) or not isinstance(channels, dict):
            raise AidError("bad_channels", f"{path} holds no channel map")
        clean: dict[str, str] = {}
        for name, channel in channels.items():
            if not isinstance(channel, str):
                raise AidError("bad_channels", f"{path} holds no channel map")
            clean[name] = channel
        return cls(path, category, clean)

    def channel_of(self, channel_id: str) -> str | None:
        """The session a channel id belongs to, or None when it is none of ours."""
        return next((name for name, mapped in self.channels.items() if mapped == channel_id), None)

    def forget_channel(self, channel_id: str) -> str | None:
        """Drop a deleted channel's entry, naming its session so the caller recreates it."""
        session = self.channel_of(channel_id)
        if session is not None:
            del self.channels[session]
        return session

    async def save(self) -> None:
        path = anyio.Path(self._path)
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_text(json.dumps({"category": self.category_id, "channels": self.channels}))


async def reconcile(client: Client, rest: Rest, guild_id: str, category_name: str, channel_map: ChannelMap) -> None:
    """The category and one channel per session exist; mapped channels of gone sessions are deleted. A session
    with a parent gets none: its results reach the parent's channel."""
    infos = [info for info in await daemon_call(client.sessions()) if info.parent is None]
    living = await rest.guild_channels(guild_id)
    by_id = {channel.id: channel for channel in living}
    category = channel_map.category_id
    if category is None or category not in by_id or by_id[category].type != 4:
        found = next((c.id for c in living if c.type == 4 and c.name == category_name), None)
        category = found if found is not None else await rest.create_channel(guild_id, category_name, channel_type=4)
        channel_map.category_id = category
        log.info("discord_category", category=category)
    taken = {c.name for c in living if c.type == 0 and c.parent_id == category}
    for info in infos:
        if info.name in channel_map.channels and channel_map.channels[info.name] in by_id:
            continue
        name = unique_slug(slug(info.name), taken)
        taken.add(name)
        channel_id = await rest.create_channel(
            guild_id, name, parent_id=category, topic=f"aid session {info.name}: {state_word(info)}"
        )
        channel_map.channels[info.name] = channel_id
        log.info("discord_channel_created", session=info.name, channel=channel_id)
    for session in [name for name in channel_map.channels if name not in {info.name for info in infos}]:
        channel_id = channel_map.channels.pop(session)
        try:
            await rest.delete_channel(channel_id)
        except Exception as error:
            log.warning("discord_channel_delete_failed", session=session, error=f"{type(error).__name__}: {error}")
            continue
        log.info("discord_channel_deleted", session=session, channel=channel_id)
    await channel_map.save()
