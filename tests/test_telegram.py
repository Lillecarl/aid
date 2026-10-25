from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import anyio.lowlevel
import httpx
import pytest

import aid
from aid.client import daemon_call, register_plugin
from aid.discovery import discover
from aid.mutes import Mutes
from aid.plugins import Grant
from aid.protocol import AidError
from aid.settle import Tracker, route_topic
from plugins.telegram.bot import (
    COMMANDS,
    DESCRIPTION,
    SHORT_DESCRIPTION,
    Bot,
    BotError,
    Button,
    Buttons,
    Callback,
    Reply,
    Update,
    handle_callback,
    handle_command,
    parse_command,
    poll_delay,
    read_config,
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


def test_discovers_telegram_on_import() -> None:
    catalog = discover([PLUGINS_ROOT], only=["telegram"])
    telegram = catalog.plugins["telegram"]
    assert (catalog.roots["telegram"].manifest.version, telegram.grants) == (
        "0.1.0",
        frozenset({Grant.READ, Grant.MESSAGE}),
    )
    assert telegram.plugin_name() == "telegram"


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
        bot = Bot("tok", client=http)
        await bot.send(7, "hi")
        await bot.set_commands()
    assert calls == [
        ("POST", "/bottok/sendMessage", {"chat_id": 7, "text": "hi"}),
        (
            "POST",
            "/bottok/setMyCommands",
            {
                "commands": [
                    {"command": "sessions", "description": "list sessions"},
                    {"command": "status", "description": "a session's state"},
                    {"command": "tail", "description": "a session's last output"},
                    {"command": "send", "description": "message a session"},
                    {"command": "mute", "description": "silence a session"},
                    {"command": "unmute", "description": "listen to a session again"},
                    {"command": "muted", "description": "list silenced sessions"},
                    {"command": "help", "description": "this list"},
                ]
            },
        ),
    ]


async def test_updates_skips_what_has_no_message_text() -> None:
    calls: list[tuple[str, str, Any]] = []
    result = [
        {"update_id": 1, "message": {"chat": {"id": 7}, "text": "/sessions"}},
        {"update_id": 2, "edited_message": {"chat": {"id": 7}, "text": "late"}},
        {"update_id": 3, "message": {"chat": {"id": 7}}},
        {"update_id": 4},
    ]
    async with httpx.AsyncClient(transport=_transport(calls, result)) as http:
        messages, taps, offset = await Bot("tok", client=http).updates(0)
    assert messages == [Update(update_id=1, chat_id=7, text="/sessions")]
    assert taps == []
    assert offset == 5
    assert ("GET", "/bottok/getUpdates", {"offset": "0", "timeout": "25"}) in calls


async def test_updates_returns_taps_and_skips_the_unanswerable() -> None:
    calls: list[tuple[str, str, Any]] = []
    result = [
        {
            "update_id": 1,
            "callback_query": {
                "id": "q1",
                "data": "tok",
                "message": {"message_id": 5, "chat": {"id": CHAT}},
            },
        },
        {"update_id": 2, "callback_query": {"id": "q2"}},  # No data and no message.
        {"update_id": 3, "callback_query": {"id": "q3", "data": "tok"}},  # No message: unanswerable.
        {"nonsense": True},  # No update_id: dropped, and the offset untouched.
    ]
    async with httpx.AsyncClient(transport=_transport(calls, result)) as http:
        messages, taps, offset = await Bot("tok", client=http).updates(0)
    assert messages == []
    assert taps == [Callback(update_id=1, query_id="q1", chat_id=CHAT, message_id=5, data="tok")]
    assert offset == 4


async def test_button_calls_post_to_the_bot_api() -> None:
    calls: list[tuple[str, str, Any]] = []
    keyboard = {"inline_keyboard": [[{"text": "a", "callback_data": "t1"}, {"text": "b", "callback_data": "t2"}]]}
    async with httpx.AsyncClient(transport=_transport(calls, True)) as http:
        bot = Bot("tok", client=http)
        await bot.send(7, "hi", ((Button("a", "t1"), Button("b", "t2")),))
        await bot.edit(7, 42, "hola", ((Button("a", "t1"), Button("b", "t2")),))
        await bot.answer_callback("q1", "toast")
        await bot.answer_callback("q2")
        await bot.set_description()
    assert calls == [
        ("POST", "/bottok/sendMessage", {"chat_id": 7, "text": "hi", "reply_markup": keyboard}),
        ("POST", "/bottok/editMessageText", {"chat_id": 7, "message_id": 42, "text": "hola", "reply_markup": keyboard}),
        ("POST", "/bottok/answerCallbackQuery", {"callback_query_id": "q1", "text": "toast", "show_alert": False}),
        ("POST", "/bottok/answerCallbackQuery", {"callback_query_id": "q2"}),
        ("POST", "/bottok/setMyDescription", {"description": DESCRIPTION}),
        ("POST", "/bottok/setMyShortDescription", {"short_description": SHORT_DESCRIPTION}),
    ]


async def test_send_takes_the_sent_message_telegram_answers() -> None:
    sent = {"message_id": 1, "date": 0, "chat": {"id": 7, "type": "private"}, "text": "hi"}
    async with httpx.AsyncClient(transport=_transport([], sent)) as http:
        await Bot("tok", client=http).send(7, "hi")


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
    monkeypatch.setattr("aid.client.DAEMON_CALL_TIMEOUT", 0.1)

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
    buttons = Buttons()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:

            async def cmd(text: str) -> Reply:
                reply = await handle_command(control, mutes, buttons, text)
                assert reply is not None
                return reply

            assert (await cmd("/sessions")).text == "no sessions"
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await cmd("/sessions")
            assert listed.text == "echo idle"
            [row] = listed.markup
            [status_button, tail_button, mute_button] = row
            assert (status_button.text, buttons.take(status_button.callback)) == ("echo", ("status", "echo"))
            assert (tail_button.text, buttons.take(tail_button.callback)) == ("tail", ("tail", "echo"))
            assert (mute_button.text, buttons.take(mute_button.callback)) == ("mute", ("mute", "echo"))
            assert (await cmd("/status echo")).text == "echo idle\nfunction::_echo"
            assert (await cmd("/tail echo")).text == "echo: no output yet"
            assert (await cmd("/mute echo")).text == "muted 'echo'"
            assert (await cmd("/sessions")).text == "echo idle muted"
            muted = await cmd("/muted")
            assert muted.text == "echo"
            [[unmute_button]] = muted.markup
            assert (unmute_button.text, buttons.take(unmute_button.callback)) == ("unmute echo", ("unmute", "echo"))
            assert (await cmd("/unmute echo")).text == "unmuted 'echo'"
            # /send wakes the session, so only its reply is asserted.
            assert (await cmd("/send echo hello")).text == "sent to 'echo'"
            await control.session("echo").stop()
            assert (await cmd("/sessions")).text == "echo stopped"
            assert (await cmd("/status missing")).text == "aid: no session named 'missing'"
            assert (await cmd("/frobnicate echo")).text == (
                "unknown command 'frobnicate'; sessions, status <name>, tail <name>, "
                "send <name> <text>, mute <name>, unmute <name>, muted"
            )
            assert await handle_command(control, mutes, buttons, "just chatting") is None


def test_buttons_mint_resolve_and_expire() -> None:
    buttons = Buttons()
    token = buttons.mint("status", "echo")
    assert len(token) <= 64  # callback_data holds 1-64 bytes.
    assert buttons.take(token) == ("status", "echo")
    assert buttons.take("nope") is None


async def test_taps_read_and_toggle(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / "mutes.json")
    buttons = Buttons()

    def tap(data: str) -> Callback:
        return Callback(update_id=1, query_id="q", chat_id=CHAT, message_id=7, data=data)

    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await handle_command(control, mutes, buttons, "/sessions")
            assert listed is not None
            [[status_button, _, mute_button]] = listed.markup
            status = await handle_callback(control, mutes, buttons, tap(status_button.callback))
            assert status is not None
            assert (status.text, status.edit) == ("echo idle\nfunction::_echo", False)
            toggled = await handle_callback(control, mutes, buttons, tap(mute_button.callback))
            assert toggled is not None
            assert (toggled.text, toggled.edit) == ("echo idle muted", True)
            [[_, _, unmute_button]] = toggled.markup
            assert buttons.take(unmute_button.callback) == ("unmute", "echo")
            assert await handle_callback(control, mutes, buttons, tap("nope")) is None


class ScriptBot(Bot):
    """Serve-loop driver: each poll drains the queue, so the loop idles after."""

    def __init__(self, updates: list[Update | Callback]) -> None:
        super().__init__("test")
        self._queued = updates
        self.sent: list[tuple[int, str, tuple[tuple[Button, ...], ...]]] = []
        self.answered: list[tuple[str, str | None, bool]] = []
        self.edited: list[tuple[int, int, str]] = []
        self.menu: tuple[tuple[str, str], ...] = ()
        self.described = False
        self.got = anyio.Event()

    async def send(self, chat_id: int, text: str, markup: tuple[tuple[Button, ...], ...] = ()) -> None:
        self.sent.append((chat_id, text, markup))
        self.got.set()

    async def edit(self, chat_id: int, message_id: int, text: str, markup: tuple[tuple[Button, ...], ...] = ()) -> None:
        self.edited.append((chat_id, message_id, text))
        self.got.set()

    async def answer_callback(self, query_id: str, text: str | None = None, *, alert: bool = False) -> None:
        self.answered.append((query_id, text, alert))

    async def set_commands(self, commands: tuple[tuple[str, str], ...] = COMMANDS) -> None:
        self.menu = commands

    async def set_description(self) -> None:
        self.described = True

    async def updates(self, offset: int) -> tuple[list[Update], list[Callback], int]:
        await anyio.lowlevel.checkpoint()  # A checkpoint per poll: the real one awaits the network.
        out, self._queued = self._queued, []
        last = offset
        for update in out:
            last = max(last, update.update_id + 1)
        return (
            [update for update in out if isinstance(update, Update)],
            [update for update in out if isinstance(update, Callback)],
            last,
        )


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
                tg.start_soon(serve, control, bot, {CHAT}, await Mutes.load(tmp_path / "mutes.json"), Buttons())
                await bot.got.wait()
                await anyio.sleep(0.5)  # Time for the dropped chat to pass through.
                tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    assert bot.sent == [(CHAT, "nothing muted", ())]
    assert bot.menu == COMMANDS
    assert bot.described


async def test_serve_answers_taps(daemon: Paths, tmp_path: Path) -> None:
    mutes = await Mutes.load(tmp_path / "mutes.json")
    buttons = Buttons()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await handle_command(control, mutes, buttons, "/sessions")
            assert listed is not None
            [[_, _, mute_button]] = listed.markup
            bot = ScriptBot(
                [
                    Callback(update_id=1, query_id="q1", chat_id=CHAT, message_id=42, data=mute_button.callback),
                    Callback(update_id=2, query_id="q2", chat_id=CHAT, message_id=43, data="nope"),
                ]
            )
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(serve, control, bot, {CHAT}, mutes, buttons)
                    await anyio.sleep(1)  # Both taps answer and the mute tap edits, locally and at once.
                    tg.cancel_scope.cancel()
            finally:
                await bot.aclose()
    [(chat_id, message_id, text)] = bot.edited
    assert (chat_id, message_id, text) == (CHAT, 42, "echo idle muted")
    assert bot.answered == [("q1", None, False), ("q2", "Those buttons expired — send /sessions again", True)]
    assert bot.sent == []


class FakeBot(Bot):
    def __init__(self) -> None:
        super().__init__("test")
        self.sent: list[tuple[int, str]] = []
        self.got = anyio.Event()

    async def send(self, chat_id: int, text: str, markup: tuple[tuple[Button, ...], ...] = ()) -> None:
        self.sent.append((chat_id, text))
        self.got.set()

    async def updates(self, offset: int) -> tuple[list[Update], list[Callback], int]:
        return [], [], offset


async def test_watch_notifies_a_settled_turn(daemon: Paths, tmp_path: Path) -> None:
    bot = FakeBot()
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as control:
                session = await control.create("echo", py_spec(tmp_path, "agents:echo"))
                child = py_spec(tmp_path, "agents:echo").model_copy(update={"parent": "echo"})
                fork = await control.create("echo.fork-abcdef", child)
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
                    await fork.run("unseen")
                    await anyio.sleep(1.0)  # Past the idle window: a child's turn would have settled by now.
                    assert "echo hello" in (await session.run("hello")).text
                    await bot.got.wait()
                    tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    [(chat, text)] = bot.sent
    assert chat == CHAT
    assert "echo: turn ended" in text and "echo hello" in text


async def test_watch_skips_a_child_made_after_it_started(daemon: Paths, tmp_path: Path) -> None:
    """A classifier is made on first use, after the watcher read the list: its turns are no person's notice."""
    bot = FakeBot()
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as control:
                session = await control.create("boss", py_spec(tmp_path, "agents:echo"))
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
                    child = py_spec(tmp_path, "agents:echo").model_copy(update={"parent": "boss"})
                    classifier = await control.create("boss.classifier", child)
                    await classifier.run("unseen")
                    await anyio.sleep(1.0)  # Past the idle window: a child's turn would have settled by now.
                    await session.run("hello")
                    await bot.got.wait()
                    tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    assert [text.split(":", 1)[0] for _, text in bot.sent] == ["boss"]


async def test_watch_notifies_a_failed_turn(daemon: Paths, tmp_path: Path) -> None:
    bot = FakeBot()
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as control:
                session = await control.create("oops", py_spec(tmp_path, "agents:failer"))
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
                    with pytest.raises(AidError, match="agent_error"):
                        await session.run("go")
                    await bot.got.wait()
                    tg.cancel_scope.cancel()
    finally:
        await bot.aclose()
    [(chat, text)] = bot.sent
    assert chat == CHAT
    assert "oops: turn failed (agent_error)" in text and "boom" in text
