from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import anyio.lowlevel
import httpx
import pytest

import aid
from aid.client import register_plugin
from aid.plugins import Grant, discover
from aid.protocol import AidError
from plugins.telegram.bot import (
    Bot,
    BotError,
    Mutes,
    Tracker,
    Update,
    daemon_call,
    handle_command,
    parse_command,
    poll_delay,
    read_config,
    route_topic,
    serve,
    watch,
)
from tests.conftest import py_spec

if TYPE_CHECKING:
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30
CHAT = 1234
PLUGINS_ROOT = Path(__file__).resolve().parents[1] / "plugins"


def test_discovers_telegram_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    telegram = discover([PLUGINS_ROOT]).plugins["telegram"]
    assert (telegram.version, telegram.grants) == ("0.1.0", frozenset({Grant.READ, Grant.MESSAGE}))


def test_read_config_drops_secrets_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AID_TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("AID_TELEGRAM_CHATS", "1, 22")
    config = read_config()
    assert (config.token, config.chats) == ("tok", frozenset({1, 22}))
    assert "AID_TELEGRAM_TOKEN" not in os.environ
    assert "AID_TELEGRAM_CHATS" not in os.environ


def test_read_config_refuses_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AID_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("AID_TELEGRAM_CHATS", raising=False)
    with pytest.raises(AidError, match="no_token"):
        read_config()
    monkeypatch.setenv("AID_TELEGRAM_TOKEN", "tok")
    with pytest.raises(AidError, match="no_chats"):
        read_config()
    # read_config pops what it reads: the token is gone again.
    monkeypatch.setenv("AID_TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("AID_TELEGRAM_CHATS", "abc")
    with pytest.raises(AidError, match="bad_chats"):
        read_config()


def test_parse_command() -> None:
    assert parse_command("/status echo now") == ("status", ["echo", "now"])
    assert parse_command("/send@bot echo hi") == ("send", ["echo", "hi"])
    assert parse_command("hello") is None
    assert parse_command("") is None


def test_route_names_session_topics() -> None:
    assert route_topic(b"session/echo/turn/") == ("echo", "turn")
    assert route_topic(b"session/echo/supervision/") == ("echo", "supervision")
    assert route_topic(b"session/echo/status/") == ("echo", "status")
    assert route_topic(b"sessions/") is None
    assert route_topic(b"session/echo/tool/") is None
    assert route_topic(b"session/echo/turn/extra/") is None


def _transport(calls: list[tuple[str, str, Any]], result: Any) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls.append((request.method, request.url.path, json.loads(request.content)))
        else:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"ok": True, "result": result})

    return httpx.MockTransport(handler)


async def test_send_posts_to_the_bot_api() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with httpx.AsyncClient(transport=_transport(calls, True)) as http:
        await Bot("tok", client=http).send(7, "hi")
    assert calls == [("POST", "/bottok/sendMessage", {"chat_id": 7, "text": "hi"})]


async def test_updates_skips_what_has_no_message_text() -> None:
    calls: list[tuple[str, str, Any]] = []
    result = [
        {"update_id": 1, "message": {"chat": {"id": 7}, "text": "/sessions"}},
        {"update_id": 2, "edited_message": {"chat": {"id": 7}, "text": "late"}},
        {"update_id": 3, "message": {"chat": {"id": 7}}},
        {"update_id": 4},
    ]
    async with httpx.AsyncClient(transport=_transport(calls, result)) as http:
        updates, offset = await Bot("tok", client=http).updates(0)
    assert updates == [Update(update_id=1, chat_id=7, text="/sessions")]
    assert offset == 5
    assert ("GET", "/bottok/getUpdates", {"offset": "0", "timeout": "25"}) in calls


async def test_bot_api_failure_raises() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bottok/sendMessage":
            return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})
        return httpx.Response(200, json={"ok": False, "description": "Bad Request: chat not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        bot = Bot("tok", client=http)
        with pytest.raises(BotError) as refused:
            await bot.send(7, "hi")
        assert poll_delay(refused.value) is None
        with pytest.raises(BotError, match="chat not found"):
            await bot.updates(0)


async def test_throttled_poll_waits_out_retry_after() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"ok": False, "description": "Too Many Requests: retry after 3", "parameters": {"retry_after": 3}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(BotError) as throttled:
            await Bot("tok", client=http).updates(0)
    assert poll_delay(throttled.value) == 3.0


def test_poll_delay_exits_on_dead_requests() -> None:
    assert poll_delay(BotError(401, "Unauthorized")) is None
    assert poll_delay(BotError(404, "Not Found")) is None
    assert poll_delay(BotError(429, "Too Many Requests")) == 60.0
    assert poll_delay(BotError(500, "Server Error")) == 5.0


async def test_daemon_call_exits_when_the_daemon_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plugins.telegram.bot.CALL_TIMEOUT", 0.1)

    async def hang() -> None:
        await anyio.sleep_forever()

    with pytest.raises(SystemExit, match="answered nothing"):
        await daemon_call(hang())


def test_tracker_settles_an_unsupervised_turn() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_started("s")
    tracker.turn_ended("s", "t1", now=0)
    assert tracker.tick(9) == []
    assert tracker.tick(10) == ["s"]
    assert tracker.tick(100) == []


def test_tracker_revival_cancels_a_pending_notice() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_ended("s", "t1", now=0)
    tracker.turn_started("s")
    assert tracker.tick(100) == []


def test_tracker_waits_out_a_silent_overseer() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_ended("s", "t1", now=0)
    tracker.firing("s", "v", "t1", now=1)
    assert tracker.tick(200) == []
    tracker.turn_started("v")
    tracker.turn_ended("v", "vt1", now=210)
    assert tracker.tick(215) == []
    assert tracker.tick(220) == ["s", "v"]


def test_tracker_nudge_cancels_the_notice() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_ended("s", "t1", now=0)
    tracker.firing("s", "v", "t1", now=1)
    tracker.nudge("s")
    assert tracker.tick(1000) == []


def test_tracker_verdict_timeout_settles_on_idleness() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_ended("s", "t1", now=0)
    tracker.firing("s", "v", "t1", now=1)
    assert tracker.tick(301) == ["s"]


def test_tracker_ignores_a_firing_for_another_turn() -> None:
    tracker = Tracker(idle_window=10, verdict_timeout=300)
    tracker.turn_ended("s", "t2", now=0)
    tracker.turn_started("s")
    tracker.turn_ended("s", "t3", now=5)
    tracker.firing("s", "v", "t2", now=6)
    assert tracker.tick(15) == ["s"]


async def test_mutes_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "mutes.json"
    mutes = await Mutes.load(path)
    assert "s" not in mutes
    await mutes.mute("s")
    assert "s" in (await Mutes.load(path))
    await mutes.unmute("s")
    assert (await Mutes.load(path)).names() == []


async def test_mutes_refuses_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "mutes.json"
    path.write_text("[not json")
    with pytest.raises(AidError, match="bad_mutes"):
        await Mutes.load(path)


async def test_commands_answer_and_mute(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / "mutes.json")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            assert await handle_command(control, mutes, "/sessions") == "no sessions"
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            assert await handle_command(control, mutes, "/sessions") == "echo idle"
            assert await handle_command(control, mutes, "/status echo") == "echo idle\nfunction::_echo"
            assert await handle_command(control, mutes, "/tail echo") == "echo: no output yet"
            assert await handle_command(control, mutes, "/mute echo") == "muted 'echo'"
            assert await handle_command(control, mutes, "/sessions") == "echo idle muted"
            assert await handle_command(control, mutes, "/muted") == "echo"
            assert await handle_command(control, mutes, "/unmute echo") == "unmuted 'echo'"
            # /send wakes the session, so only its reply is asserted.
            assert await handle_command(control, mutes, "/send echo hello") == "sent to 'echo'"
            assert await handle_command(control, mutes, "/status missing") == "aid: no session named 'missing'"
            assert await handle_command(control, mutes, "/frobnicate echo") == (
                "unknown command 'frobnicate'; sessions, status <name>, tail <name>, "
                "send <name> <text>, mute <name>, unmute <name>, muted"
            )
            assert await handle_command(control, mutes, "just chatting") is None


class ScriptBot(Bot):
    """Serve-loop driver: each poll drains the queue, so the loop idles after."""

    def __init__(self, updates: list[Update]) -> None:
        super().__init__("test")
        self._queued = updates
        self.sent: list[tuple[int, str]] = []
        self.got = anyio.Event()

    async def send(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))
        self.got.set()

    async def updates(self, offset: int) -> tuple[list[Update], int]:
        await anyio.lowlevel.checkpoint()  # A checkpoint per poll: the real one awaits the network.
        out, self._queued = self._queued, []
        last = offset
        for update in out:
            last = max(last, update.update_id + 1)
        return out, last


async def test_serve_answers_allowlisted_chats(daemon: Paths, tmp_path: Path) -> None:
    bot = ScriptBot(
        [
            Update(update_id=1, chat_id=CHAT, text="/muted"),
            Update(update_id=2, chat_id=999, text="/sessions"),
        ]
    )
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as control, anyio.create_task_group() as tg:
                tg.start_soon(serve, control, bot, {CHAT}, await Mutes.load(tmp_path / "mutes.json"))
                await bot.got.wait()
                await anyio.sleep(0.5)  # Time for the dropped chat to pass through.
                tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    assert bot.sent == [(CHAT, "nothing muted")]


class FakeBot(Bot):
    def __init__(self) -> None:
        super().__init__("test")
        self.sent: list[tuple[int, str]] = []
        self.got = anyio.Event()

    async def send(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))
        self.got.set()

    async def updates(self, offset: int) -> tuple[list[Update], int]:
        return [], offset


async def test_watch_notifies_a_settled_turn(daemon: Paths, tmp_path: Path) -> None:
    bot = FakeBot()
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as control:
                session = await control.create("echo", py_spec(tmp_path, "agents:echo"))
                await register_plugin(control, daemon, "telegram", frozenset({Grant.READ, Grant.MESSAGE}))
                async with aid.connect(daemon, plugin="telegram") as plugin, anyio.create_task_group() as tg:
                    tg.start_soon(
                        partial(
                            watch,
                            plugin,
                            bot,
                            {CHAT},
                            await Mutes.load(tmp_path / "mutes.json"),
                            idle_window=0.5,
                            tick=0.2,
                        )
                    )
                    await anyio.sleep(0.5)  # Time for the subscription to reach the daemon.
                    assert "echo hello" in (await session.run("hello")).text
                    await bot.got.wait()
                    tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    [(chat, text)] = bot.sent
    assert chat == CHAT
    assert "echo: turn ended" in text and "echo hello" in text
