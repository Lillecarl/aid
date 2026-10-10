"""The Discord serve loop: commands, taps, notices, dispatch routing and the channel loop."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import pytest

import aid
from aid.mutes import Mutes
from aid.plugins import Grant, discover
from aid.protocol import AidError, MessageEntry
from plugins.discord.channels import CHANNELS_FILE, ChannelMap
from plugins.discord.rest import GuildChannel, Rest, action_rows
from plugins.discord.serve import (
    EMBED_DONE,
    EMBED_FAILED,
    EMBED_OTHER,
    MUTES_FILE,
    Reply,
    _notice_parts,  # pyright: ignore[reportPrivateUsage] -- the notice shape is the unit under test
    channel_loop,
    dispatch,
    fence_output,
    handle_command,
    handle_tap,
    notice_loop,
    on_message,
    parse_command,
    read_config,
)
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pydantic import JsonValue

    from aid.client import Client
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30
GUILD = "g1"
PLUGINS_ROOT = Path(__file__).resolve().parents[1] / "plugins"


def test_discovers_discord_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    discord = discover([PLUGINS_ROOT]).plugins["discord"]
    assert (discord.version, discord.grants) == ("0.1.0", frozenset({Grant.READ, Grant.MESSAGE}))


def test_read_config_drops_secrets_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AID_DISCORD_TOKEN", "tok")
    monkeypatch.setenv("AID_DISCORD_GUILD", "123")
    monkeypatch.setenv("AID_DISCORD_CATEGORY", "played sessions")
    config = read_config()
    assert (config.token, config.guild_id, config.category) == ("tok", "123", "played sessions")
    assert "AID_DISCORD_TOKEN" not in os.environ
    assert "AID_DISCORD_GUILD" not in os.environ


def test_read_config_refuses_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AID_DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("AID_DISCORD_GUILD", raising=False)
    with pytest.raises(AidError, match="no_token"):
        read_config()
    # read_config pops what it reads: the token is gone again.
    monkeypatch.setenv("AID_DISCORD_TOKEN", "tok")
    with pytest.raises(AidError, match="bad_guild"):
        read_config()
    monkeypatch.setenv("AID_DISCORD_TOKEN", "tok")
    monkeypatch.setenv("AID_DISCORD_GUILD", "abc")
    with pytest.raises(AidError, match="bad_guild"):
        read_config()


def test_parse_command() -> None:
    assert parse_command("!status echo now") == ("status", ["echo", "now"])
    assert parse_command("!send echo hi") == ("send", ["echo", "hi"])
    assert parse_command("hello") is None
    assert parse_command("") is None


class FakeRest(Rest):
    """A scripted guild listing; sends, answers, creates and deletes recorded."""

    def __init__(self, listing: list[GuildChannel] | None = None) -> None:
        self.listing = listing if listing is not None else []
        self.sent: list[tuple[str, str, JsonValue | None, list[dict[str, JsonValue]] | None]] = []
        self.answered: list[tuple[str, str, str, bool, bool, JsonValue | None]] = []
        self.created: list[tuple[str, str, int, str | None, str | None]] = []
        self.deleted: list[str] = []

    async def guild_channels(self, guild_id: str) -> list[GuildChannel]:
        assert guild_id == GUILD
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

    async def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        components: JsonValue | None = None,
        embeds: list[dict[str, JsonValue]] | None = None,
    ) -> str:
        self.sent.append((channel_id, content, components, embeds))
        return "m1"

    async def answer_interaction(
        self,
        interaction_id: str,
        interaction_token: str,
        content: str,
        *,
        update: bool = False,
        ephemeral: bool = False,
        components: JsonValue | None = None,
    ) -> None:
        self.answered.append((interaction_id, interaction_token, content, update, ephemeral, components))


async def test_commands_answer_and_mute(daemon: Paths, tmp_path: Path) -> None:

    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:

            async def cmd(text: str, session: str | None = None) -> Reply:
                reply = await handle_command(control, mutes, text, session=session)
                assert reply is not None
                return reply

            assert (await cmd("!sessions")).content == "no sessions"
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await cmd("!sessions")
            assert listed.content == "echo idle"
            assert listed.components == action_rows(
                ((("echo", "status:echo"), ("tail", "tail:echo"), ("mute", "mute:echo")),)
            )
            assert (await cmd("!status echo")).content == "echo idle\nfunction::_echo"
            assert (await cmd("!status", session="echo")).content == "echo idle\nfunction::_echo"
            assert (await cmd("!status")).content.startswith("!sessions")
            assert (await cmd("!tail echo")).content == "echo: no output yet"
            assert (await cmd("!mute echo")).content == "muted 'echo'"
            assert (await cmd("!sessions")).content == "echo idle muted"
            muted = await cmd("!muted")
            assert muted.content == "echo"
            assert muted.components == action_rows(((("unmute echo", "unmute:echo"),),))
            assert (await cmd("!unmute echo")).content == "unmuted 'echo'"
            # !send wakes the session, so only its reply is asserted.
            assert (await cmd("!send echo hello")).content == "sent to 'echo'"
            await control.session("echo").stop()
            assert (await cmd("!sessions")).content == "echo stopped"
            assert (await cmd("!status missing")).content == "aid: no session named 'missing'"
            assert (await cmd("!frobnicate echo")).content.startswith("unknown command 'frobnicate'")
            assert await handle_command(control, mutes, "just chatting", session=None) is None


async def test_taps_read_and_toggle(daemon: Paths, tmp_path: Path) -> None:

    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            status = await handle_tap(control, mutes, "status:echo")
            assert (status.content, status.update, status.ephemeral) == ("echo idle\nfunction::_echo", False, False)
            toggled = await handle_tap(control, mutes, "mute:echo")
            assert (toggled.content, toggled.update, toggled.ephemeral) == ("echo idle muted", True, False)
            assert toggled.components == action_rows(
                ((("echo", "status:echo"), ("tail", "tail:echo"), ("unmute", "unmute:echo")),)
            )
            assert (await handle_tap(control, mutes, "nope")).ephemeral
            assert (await handle_tap(control, mutes, "bogus:echo")).ephemeral


async def test_on_message_routes_commands_and_channel_text(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, None, {"echo": "c1"})
    rest = FakeRest()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await on_message(control, rest, mutes, channel_map, channel_id="c1", content="!status")
            assert rest.sent[-1] == ("c1", "echo idle\nfunction::_echo", None, None)
            await on_message(control, rest, mutes, channel_map, channel_id="c1", content="hello")
            history = await control.session("echo").history()
            assert any(isinstance(e.item, MessageEntry) and e.item.text == "hello" for e in history.entries)
            sent = len(rest.sent)
            await on_message(control, rest, mutes, channel_map, channel_id="c9", content="hi")
            assert len(rest.sent) == sent  # Bare text outside session channels is ignored.
            await on_message(control, rest, mutes, channel_map, channel_id="c9", content="!sessions")
            assert rest.sent[-1][0] == "c9"  # Commands work in any heard channel.
            assert rest.sent[-1][3] is None  # Plain replies carry no embeds.


async def test_dispatch_routes_and_ignores(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, None, {"echo": "c1"})
    rest = FakeRest()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))

            async def incoming(event: str, data: JsonValue) -> None:
                await dispatch(control, rest, mutes, channel_map, GUILD, "cat", event, data)

            message: dict[str, Any] = {"author": {"id": "u", "bot": False}, "channel_id": "c1", "content": "!tail"}
            await incoming("MESSAGE_CREATE", {**message, "author": {"id": "b", "bot": True}, "guild_id": GUILD})
            await incoming("MESSAGE_CREATE", {**message, "guild_id": "other"})
            assert rest.sent == []
            await incoming("MESSAGE_CREATE", {**message, "guild_id": GUILD})
            assert rest.sent == [("c1", "echo: no output yet", None, None)]
            await incoming("MESSAGE_CREATE", {k: v for k, v in message.items() if k != "guild_id"})
            assert len(rest.sent) == 2  # DMs have no guild id and still work.
            await incoming(
                "INTERACTION_CREATE",
                {"id": "i1", "token": "t1", "data": {"custom_id": "status:echo"}},
            )
            assert rest.answered == [("i1", "t1", "echo idle\nfunction::_echo", False, False, None)]
            await incoming("TYPING_START", {})
            assert len(rest.sent) == 2 and len(rest.answered) == 1


async def test_channel_delete_heals(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest([GuildChannel(id="cat", name="aid sessions", type=4)])
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, "cat", {"echo": "cdead"})
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await dispatch(control, rest, mutes, channel_map, GUILD, "aid sessions", "CHANNEL_DELETE", {"id": "cdead"})
            assert channel_map.channels == {"echo": "new-0"}
            assert rest.created == [("g1", "echo", 0, "cat", f"aid session echo: {await _topic_word(control, 'echo')}")]


async def _topic_word(control: Client, session: str) -> str:
    status = await control.session(session).status()
    if not status.running:
        return "stopped"
    return "busy" if status.working else "idle"


async def test_notice_posts_to_session_channel(daemon: Paths, tmp_path: Path) -> None:

    rest = FakeRest()
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, None, {"echo": "c1"})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            session = await control.create("echo", py_spec(tmp_path, "agents:echo"))
            async with anyio.create_task_group() as tg:
                tg.start_soon(partial(notice_loop, control, rest, mutes, channel_map, idle_window=0.5, tick=0.1))
                await anyio.sleep(0.3)  # Time for the subscription to reach the daemon.
                assert "echo hello" in (await session.run("hello")).text
                await anyio.sleep(1.5)  # The turn settles past its idle window, then posts.
                tg.cancel_scope.cancel()
    [(channel_id, content, components, embeds)] = rest.sent
    assert channel_id == "c1"
    assert "echo: turn ended" in content
    assert "```" not in content
    assert embeds is not None and len(embeds) == 1
    assert embeds[0]["color"] == EMBED_DONE
    assert "echo hello" in str(embeds[0]["description"])
    assert str(embeds[0]["description"]).count("```") == 2
    assert components == action_rows(((("mute", "mute:echo"),),))


async def test_notice_reports_a_failed_turn(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest()
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, None, {"oops": "c1"})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            session = await control.create("oops", py_spec(tmp_path, "agents:failer"))
            async with anyio.create_task_group() as tg:
                tg.start_soon(partial(notice_loop, control, rest, mutes, channel_map, idle_window=0.5, tick=0.1))
                await anyio.sleep(0.3)  # Time for the subscription to reach the daemon.
                with pytest.raises(AidError, match="agent_error"):
                    await session.run("go")
                await anyio.sleep(1.5)  # The turn settles past its idle window, then posts.
                tg.cancel_scope.cancel()
    [(channel_id, content, components, embeds)] = rest.sent
    assert channel_id == "c1"
    assert "oops: turn failed (agent_error)" in content
    assert "```" not in content
    assert embeds is not None and len(embeds) == 1
    assert embeds[0]["color"] == EMBED_FAILED
    assert "boom" in str(embeds[0]["description"])
    assert str(embeds[0]["description"]).count("```") == 2
    assert components == action_rows(((("mute", "mute:oops"),),))


async def test_muted_sessions_stay_silent(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest()
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    await mutes.mute("echo")
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE, None, {"echo": "c1"})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            session = await control.create("echo", py_spec(tmp_path, "agents:echo"))
            async with anyio.create_task_group() as tg:
                tg.start_soon(partial(notice_loop, control, rest, mutes, channel_map, idle_window=0.5, tick=0.1))
                await anyio.sleep(0.3)
                assert "echo hello" in (await session.run("hello")).text
                await anyio.sleep(1.5)
                tg.cancel_scope.cancel()
    assert rest.sent == []


async def test_channel_loop_reconciles_on_list_changes(daemon: Paths, tmp_path: Path) -> None:
    rest = FakeRest([GuildChannel(id="cat", name="aid sessions", type=4)])
    channel_map = ChannelMap(tmp_path / CHANNELS_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            async with anyio.create_task_group() as tg:
                tg.start_soon(channel_loop, control, rest, GUILD, "aid sessions", channel_map)
                await anyio.sleep(0.3)  # The loop reconciles once, then follows the session list.
                await control.create("echo", py_spec(tmp_path, "agents:echo"))
                await anyio.sleep(1.0)  # The list change wakes the loop, which creates the channel.
                tg.cancel_scope.cancel()
            assert channel_map.channels.keys() == {"echo"}
            assert rest.created == [("g1", "echo", 0, "cat", f"aid session echo: {await _topic_word(control, 'echo')}")]


def test_fence_output_wraps_and_escapes_inner_fences() -> None:
    assert fence_output("hi") == "```\nhi\n```"
    assert fence_output("a ``` b") == "```\na ``\u200b` b\n```"


async def test_tail_fences_machine_output(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await control.session("echo").run("hello")
            tail = await handle_command(control, mutes, "!tail echo", session=None)
            assert tail is not None
            assert tail.content == "echo: tail"
            assert tail.embeds is not None and len(tail.embeds) == 1
            assert tail.embeds[0]["color"] == EMBED_OTHER
            description = str(tail.embeds[0]["description"])
            assert description.startswith("```\n")
            assert description.endswith("\n```")
            assert "echo hello" in description


async def test_tail_escapes_inner_fences(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await control.session("echo").run("say ``` hi ```")
            tail = await handle_command(control, mutes, "!tail echo", session=None)
            assert tail is not None
            assert tail.content == "echo: tail"
            assert tail.embeds is not None and len(tail.embeds) == 1
            description = str(tail.embeds[0]["description"])
            assert description.count("```") == 2
            assert "\u200b" in description
            assert "say" in description


async def test_notice_fences_output_and_leaves_heads_plain(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / MUTES_FILE)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await control.session("echo").run("hello")
            content, embeds = await _notice_parts(control.session("echo"), None)
            head, _, _ = content.partition("\n")
            assert "```" not in head
            assert embeds is not None and len(embeds) == 1
            assert embeds[0]["color"] == EMBED_OTHER
            description = str(embeds[0]["description"])
            assert "echo hello" in description
            assert description.count("```") == 2
            status = await handle_command(control, mutes, "!status echo", session=None)
            assert status is not None and "```" not in status.content
            assert status.embeds is None
            listed = await handle_command(control, mutes, "!sessions", session=None)
            assert listed is not None and "```" not in listed.content
            assert listed.embeds is None
            empty = await handle_command(control, mutes, "!tail fresh", session=None)
            assert empty is not None and empty.content == "aid: no session named 'fresh'"
            assert "```" not in empty.content
            assert empty.embeds is None
