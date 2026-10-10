"""Session channels: slugs, the persisted map, and reconcile create/delete/adopt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.protocol import AidError
from plugins.discord.channels import CHANNELS_FILE, ChannelMap, reconcile, slug, unique_slug
from plugins.discord.rest import GuildChannel, Rest
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.client import Client
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def test_slug_lowers_and_dashes() -> None:
    assert slug("Echo") == "echo"
    assert slug("a.b") == "a-b"
    assert slug("A B_C-9") == "a-b_c-9"
    assert unique_slug("echo", set()) == "echo"
    assert unique_slug("echo", {"echo", "echo-2"}) == "echo-3"


class FakeRest(Rest):
    """A scripted guild listing; created channels join it, deleted ones leave it."""

    def __init__(self, listing: list[GuildChannel]) -> None:
        self.listing = listing
        self.created: list[tuple[str, str, int, str | None, str | None]] = []
        self.deleted: list[str] = []

    async def guild_channels(self, guild_id: str) -> list[GuildChannel]:
        assert guild_id == "g1"  # The plugin always scopes to its configured guild.
        return self.listing

    async def create_channel(
        self,
        guild_id: str,
        name: str,
        *,
        channel_type: int = 0,
        parent_id: str | None = None,
        topic: str | None = None,
    ) -> str:
        channel_id = f"new-{len(self.created)}"
        self.created.append((guild_id, name, channel_type, parent_id, topic))
        self.listing.append(GuildChannel(id=channel_id, name=name, type=channel_type, parent_id=parent_id))
        return channel_id

    async def delete_channel(self, channel_id: str) -> None:
        self.deleted.append(channel_id)
        self.listing = [channel for channel in self.listing if channel.id != channel_id]


async def test_reconcile_creates_category_and_channels(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest([])
    channel_map = await ChannelMap.load(tmp_path / CHANNELS_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await control.create("otu", py_spec(tmp_path, "agents:echo"))
            await reconcile(control, rest, "g1", "aid sessions", channel_map)
            assert channel_map.category_id == "new-0"
            assert set(channel_map.channels) == {"echo", "otu"}
            echo_id = channel_map.channels["echo"]
            assert rest.created[0] == ("g1", "aid sessions", 4, None, None)
            assert (rest.created[1][1], rest.created[1][3]) == ("echo", "new-0")
            assert rest.created[1][4] == f"aid session echo: {await _state_word(control, 'echo')}"
            reloaded = await ChannelMap.load(tmp_path / CHANNELS_FILE)
            assert (reloaded.category_id, reloaded.channels) == ("new-0", channel_map.channels)
            created = len(rest.created)
            await reconcile(control, rest, "g1", "aid sessions", channel_map)
            assert len(rest.created) == created  # Nothing new: the map names the living.
            assert rest.deleted == []
    assert echo_id in {channel.id for channel in rest.listing}


async def _state_word(control: Client, session: str) -> str:
    status = await control.session(session).status()
    if not status.running:
        return "stopped"
    return "busy" if status.working else "idle"


async def test_reconcile_deletes_gone_sessions(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest(
        [
            GuildChannel(id="cat", name="aid sessions", type=4),
            GuildChannel(id="c1", name="echo", type=0, parent_id="cat"),
        ]
    )
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, "cat", {"echo": "c1", "ghost": "c9"})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await reconcile(control, rest, "g1", "aid sessions", channel_map)
            assert channel_map.channels == {"echo": "c1"}
            assert rest.deleted == ["c9"]  # Gone from aid, and already gone from Discord too.
            assert rest.created == []


async def test_reconcile_adopts_an_existing_category(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest([GuildChannel(id="cat9", name="aid sessions", type=4)])
    channel_map = await ChannelMap.load(tmp_path / CHANNELS_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await reconcile(control, rest, "g1", "aid sessions", channel_map)
            assert channel_map.category_id == "cat9"
            assert rest.created == [
                ("g1", "echo", 0, "cat9", f"aid session echo: {await _state_word(control, 'echo')}")
            ]


async def test_channel_map_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / CHANNELS_FILE
    channel_map = await ChannelMap.load(path)
    assert channel_map.channel_of("c1") is None
    channel_map.category_id = "cat"
    channel_map.channels["echo"] = "c1"
    await channel_map.save()
    reloaded = await ChannelMap.load(path)
    assert reloaded.channel_of("c1") == "echo"
    assert reloaded.forget_channel("c1") == "echo"
    assert reloaded.forget_channel("c1") is None
    path.write_text("[not json")
    with pytest.raises(AidError, match="bad_channels"):
        await ChannelMap.load(path)
