"""Telegram notifications for settled turn ends, with basic commands back.

Only plugs into aid: it subscribes to the `session/` ZeroMQ topics and sends through the message grant.
Turns that end but come back — a supervisor nudge starting a wake turn — never notify; the settle tracker
below is that filter.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import anyio
import httpx
import structlog
from pydantic import BaseModel, ConfigDict, JsonValue

from aid.protocol import (
    AidError,
    Output,
    SessionInfo,
    SessionStatus,
    SupervisionEvent,
    TurnEvent,
    clip,
    tool_text,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine, Set
    from pathlib import Path

    from aid.client import Client, Session

log = structlog.get_logger(__name__)

TOKEN_ENV: Final = "AID_TELEGRAM_TOKEN"
CHATS_ENV: Final = "AID_TELEGRAM_CHATS"
MUTES_FILE: Final = "mutes.json"
IDLE_WINDOW: Final = 10.0
"""An unsupervised turn ended: nothing waking it within this long means it stays ended."""
VERDICT_TIMEOUT: Final = 300.0
"""A firing this old with no verdict from its overseer settles on the session's idleness instead."""
RECONCILE_INTERVAL: Final = 60.0
"""How often the watch loop re-reads the session list, repairing turns a dropped event hid."""
TICK_INTERVAL: Final = 2.0
"""How often a quiet watch re-checks for settled turns: notices lag a tick behind settling, at most."""
POLL_TIMEOUT: Final = 25
"""Seconds each Telegram `getUpdates` call waits for a message."""
RETRY_DELAY: Final = 5.0
"""Seconds between Telegram calls after a failure that heals itself."""
FLOOD_DELAY: Final = 60.0
"""Seconds to wait when Telegram throttles without naming a delay."""


def poll_delay(error: BotError) -> float | None:
    """How long the updates loop waits after `error`, or None when the request itself is dead: a 4xx other
    than throttling names a config error that retrying never heals, so the process exits loudly instead."""
    if error.status == 429:
        return error.retry_after or FLOOD_DELAY
    return None if 400 <= error.status < 500 else RETRY_DELAY


TAIL_SCAN: Final = 20
"""Newest history entries searched for the turn's output when composing a notice."""
CALL_TIMEOUT: Final = 30.0
"""A daemon call slower than this means the daemon is gone: exit loudly and let the unit restart into a
fresh baseline, rather than hang on a reply a restarted daemon never sends."""
API_BASE: Final = "https://api.telegram.org"

COMMANDS: Final = (
    ("sessions", "list sessions"),
    ("status", "a session's state"),
    ("tail", "a session's last output"),
    ("send", "message a session"),
    ("mute", "silence a session"),
    ("unmute", "listen to a session again"),
    ("muted", "list silenced sessions"),
    ("help", "this list"),
)
"""The bot's slash-command menu, published for autocompletion on startup."""


class TelegramConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    chats: frozenset[int]


def read_config() -> TelegramConfig:
    """The bot token and chat allowlist, dropped from the environment once read so neither reaches a worker."""
    token = os.environ.pop(TOKEN_ENV, "")
    if not token:
        raise AidError("no_token", f"{TOKEN_ENV} names no bot token")
    try:
        chats = frozenset(int(part) for part in os.environ.pop(CHATS_ENV, "").split(",") if part.strip())
    except ValueError:
        raise AidError("bad_chats", f"{CHATS_ENV} holds comma-separated chat ids") from None
    if not chats:
        raise AidError("no_chats", f"{CHATS_ENV} names no chat")
    return TelegramConfig(token=token, chats=chats)


class BotError(Exception):
    def __init__(self, status: int, description: str, parameters: dict[str, JsonValue] | None = None) -> None:
        super().__init__(f"Telegram {status}: {description}")
        self.status = status
        self.description = description
        self.parameters = parameters

    @property
    def retry_after(self) -> float | None:
        """Seconds Telegram asks to wait before retrying a throttled call, if it named any."""
        if self.parameters is None:
            return None
        retry_after = self.parameters.get("retry_after")
        if not isinstance(retry_after, int) or isinstance(retry_after, bool):
            return None
        return float(retry_after)


class Update(BaseModel):
    model_config = ConfigDict(frozen=True)

    update_id: int
    chat_id: int
    text: str


class _ApiBody(BaseModel):
    ok: bool
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None


class _SendBody(_ApiBody):
    result: bool = False


class _ChatBody(BaseModel):
    id: int


class _MessageBody(BaseModel):
    chat: _ChatBody
    text: str | None = None


class _UpdateBody(BaseModel):
    update_id: int
    message: _MessageBody | None = None


class _UpdatesBody(_ApiBody):
    result: list[Any] = []


def _body[M: _ApiBody](model: type[M], response: httpx.Response) -> M:
    try:
        body = model.model_validate(response.json())
    except ValueError:
        raise BotError(response.status_code, "not Telegram JSON") from None
    if not body.ok:
        raise BotError(response.status_code, body.description or "not ok", body.parameters)
    return body


class Bot:
    """The two Bot API calls the plugin needs: sending, and long-polling for messages."""

    def __init__(self, token: str, *, base_url: str = API_BASE, client: httpx.AsyncClient | None = None) -> None:
        self._base = f"{base_url}/bot{token}"
        self._client = client if client is not None else httpx.AsyncClient(timeout=POLL_TIMEOUT + 10)
        self._owned = client is None

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()

    async def send(self, chat_id: int, text: str) -> None:
        _body(_SendBody, await self._client.post(f"{self._base}/sendMessage", json={"chat_id": chat_id, "text": text}))

    async def set_commands(self, commands: tuple[tuple[str, str], ...] = COMMANDS) -> None:
        """Publish the slash-command menu, so chats complete it. Idempotent: Telegram keeps the last menu."""
        _body(
            _SendBody,
            await self._client.post(
                f"{self._base}/setMyCommands",
                json={"commands": [{"command": name, "description": text} for name, text in commands]},
            ),
        )

    async def updates(self, offset: int) -> tuple[list[Update], int]:
        """Messages since `offset`, oldest first, with the offset past the newest update seen: unparsable
        updates are skipped but never re-polled."""
        body = _body(
            _UpdatesBody,
            await self._client.get(f"{self._base}/getUpdates", params={"offset": offset, "timeout": POLL_TIMEOUT}),
        )
        out: list[Update] = []
        last = offset
        for raw in body.result:
            fields = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
            update_id = fields.get("update_id")
            if isinstance(update_id, int) and not isinstance(update_id, bool):
                last = max(last, update_id + 1)
            try:
                item = _UpdateBody.model_validate(raw)
            except ValueError:
                log.warning("telegram_update_dropped")
                continue
            if item.message is not None and item.message.text is not None:
                out.append(Update(update_id=item.update_id, chat_id=item.message.chat.id, text=item.message.text))
        return out, last


async def daemon_call[T](call: Coroutine[Any, Any, T]) -> T:
    """Await a daemon call, exiting loudly when the daemon answers nothing: a restarted daemon never replies
    to requests sent before its restart, so hanging would outlive the outage silently."""
    try:
        with anyio.fail_after(CALL_TIMEOUT):
            return await call
    except TimeoutError:
        raise SystemExit("aid telegram: the daemon answered nothing for 30s") from None


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """A Telegram message as a command and its arguments, or None when it is no command."""
    if not text.startswith("/"):
        return None
    head, *rest = text[1:].split()
    return head.split("@")[0], rest


@dataclass
class _Pending:
    turn: str
    deadline: float
    supervisor: str | None = None
    """The overseer whose verdict is awaited; None once it stayed silent or no firing came."""
    verdict_open: bool = False
    """The overseer's verdict turn runs now: its end without a nudge settles the episode."""


class Tracker:
    """Which sessions sit in an ended turn that nothing revived.

    Fed by a session's turn and supervision events; `tick` names the sessions whose episode settled.
    Time is the caller's monotonic clock, so tests drive it exactly.
    """

    def __init__(self, *, idle_window: float = IDLE_WINDOW, verdict_timeout: float = VERDICT_TIMEOUT) -> None:
        self._idle_window = idle_window
        self._verdict_timeout = verdict_timeout
        self._turn: dict[str, bool] = {}
        self._pending: dict[str, _Pending] = {}

    def set_busy(self, session: str, busy: bool) -> None:
        self._turn[session] = busy

    def turn_started(self, session: str) -> None:
        self._turn[session] = True
        self._pending.pop(session, None)
        for pending in self._pending.values():
            if pending.supervisor == session and not pending.verdict_open:
                pending.verdict_open = True

    def turn_ended(self, session: str, turn: str, now: float) -> None:
        for pending in self._pending.values():
            if pending.supervisor == session and pending.verdict_open:
                pending.supervisor = None
                pending.verdict_open = False
                pending.deadline = now + self._idle_window
        self._turn[session] = False
        self._pending[session] = _Pending(turn=turn, deadline=now + self._idle_window)

    def firing(self, session: str, supervisor: str, turn: str, now: float) -> None:
        pending = self._pending.get(session)
        if pending is None or pending.turn != turn:
            return
        pending.supervisor = supervisor
        pending.verdict_open = False
        pending.deadline = now + self._verdict_timeout

    def nudge(self, session: str) -> None:
        self._pending.pop(session, None)

    def tick(self, now: float) -> list[str]:
        """Sessions whose episode settled and that sit idle: each is named once."""
        return [
            session
            for session, pending in list(self._pending.items())
            if now >= pending.deadline
            and not self._turn.get(session, False)
            and self._pending.pop(session, None) is not None
        ]


class Mutes:
    """Sessions that stay silent, chosen at runtime; everything else notifies. Kept on disk so a bot
    restart keeps them."""

    def __init__(self, path: Path, muted: frozenset[str] = frozenset()) -> None:
        self._path = path
        self._muted = muted

    @classmethod
    async def load(cls, path: Path) -> Mutes:
        try:
            muted = frozenset(json.loads(await anyio.Path(path).read_text()).get("muted", []))
        except FileNotFoundError:
            return cls(path)
        except ValueError, AttributeError:
            raise AidError("bad_mutes", f"{path} holds no mute list") from None
        if not all(isinstance(name, str) for name in muted):
            raise AidError("bad_mutes", f"{path} holds no mute list")
        return cls(path, muted)

    def __contains__(self, session: object) -> bool:
        return session in self._muted

    def names(self) -> list[str]:
        return sorted(self._muted)

    async def mute(self, session: str) -> None:
        self._muted |= {session}
        await self._save()

    async def unmute(self, session: str) -> None:
        self._muted -= {session}
        await self._save()

    async def _save(self) -> None:
        path = anyio.Path(self._path)
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_text(json.dumps({"muted": sorted(self._muted)}))


async def _last_output(session: Session) -> Output | None:
    page = await daemon_call(session.history(limit=TAIL_SCAN))
    for entry in reversed(page.entries):
        if isinstance(entry.item, Output):
            return entry.item
    return None


async def _notice_text(session: Session, reason: str | None) -> str:
    """What Telegram hears about a settled turn: how it ended, and what it said last."""
    head = f"{session.name}: turn ended ({reason or 'unknown'})"
    try:
        status = await daemon_call(session.status())
        output = await _last_output(session)
    except AidError as error:
        return f"{head}\ngone since: {error.message}"
    lines = [head]
    if output is not None and (text := tool_text(output.output)) is not None:
        lines.append(clip(text))
    if status.model is not None:
        lines.append(status.model)
    if status.cost:
        lines.append(", ".join(f"{amount:g} {currency}" for currency, amount in sorted(status.cost.items())))
    return "\n".join(lines)


async def _notify(client: Client, bot: Bot, chats: Set[int], mutes: Mutes, session: str, reason: str | None) -> None:
    if session in mutes:
        return
    try:
        text = await _notice_text(client.session(session), reason)
    except Exception:
        log.exception("telegram_notice_failed", session=session)
        return
    for chat in chats:
        try:
            await bot.send(chat, text)
        except BotError, httpx.HTTPError:
            log.exception("telegram_send_failed", session=session, chat=chat)


def route_topic(topic: bytes) -> tuple[str, str] | None:
    """A session topic as its session and kind (`turn`, `supervision` or `status`); None for anything else."""
    parts = topic.split(b"/")
    if len(parts) != 4 or parts[0] != b"session" or parts[2] not in (b"turn", b"supervision", b"status"):
        return None
    return parts[1].decode(), parts[2].decode()


async def watch(
    client: Client,
    bot: Bot,
    chats: Set[int],
    mutes: Mutes,
    *,
    idle_window: float = IDLE_WINDOW,
    verdict_timeout: float = VERDICT_TIMEOUT,
    tick: float = TICK_INTERVAL,
    reconcile: float = RECONCILE_INTERVAL,
) -> None:
    """Tell Telegram about turns that ended and stayed ended; a nudge reviving the loop cancels the notice."""
    tracker = Tracker(idle_window=idle_window, verdict_timeout=verdict_timeout)
    reasons: dict[str, str | None] = {}
    async with client.subscribed(b"session/", idle=tick) as published:
        for info in await daemon_call(client.sessions()):
            tracker.set_busy(info.name, info.working)
        quiet = 0.0
        async for message in published:
            now = time.monotonic()
            if message is None:
                quiet += tick
                if quiet >= reconcile:
                    quiet = 0.0
                    for info in await daemon_call(client.sessions()):
                        tracker.set_busy(info.name, info.working)
            else:
                quiet = 0.0
                try:
                    topic, payload = message
                    routed = route_topic(topic)
                    if routed is not None:
                        name, kind = routed
                        if kind == "turn":
                            event = TurnEvent.model_validate_json(payload)
                            if event.phase == "started":
                                tracker.turn_started(name)
                                reasons.pop(name, None)
                            else:
                                tracker.turn_ended(name, event.turn, now)
                                reasons[name] = event.reason
                        elif kind == "supervision":
                            event = SupervisionEvent.model_validate_json(payload)
                            if event.firing is not None:
                                tracker.firing(event.session, event.supervisor, event.firing.turn, now)
                            if event.nudge is not None:
                                tracker.nudge(event.session)
                        elif kind == "status":
                            tracker.set_busy(name, SessionStatus.model_validate_json(payload).busy)
                except Exception:
                    log.exception("telegram_event_dropped")
                    continue
            for session in tracker.tick(now):
                await _notify(client, bot, chats, mutes, session, reasons.pop(session, None))


HELP: Final = """sessions, status <name>, tail <name>, send <name> <text>, mute <name>, unmute <name>, muted"""


def _state(info: SessionInfo) -> str:
    """A session's one-word state: stopped beats idle, busy beats both."""
    if not info.running:
        return "stopped"
    return "busy" if info.working else "idle"


async def handle_command(client: Client, mutes: Mutes, text: str) -> str | None:
    """An inbound Telegram message as a reply, or None when it is no command."""
    parsed = parse_command(text)
    if parsed is None:
        return None
    command, args = parsed
    if command in ("start", "help"):
        return HELP
    if command == "sessions":
        infos = await daemon_call(client.sessions())
        if not infos:
            return "no sessions"
        return "\n".join(f"{info.name} {_state(info)}{' muted' if info.name in mutes else ''}" for info in infos)
    if command == "muted":
        names = mutes.names()
        return "nothing muted" if not names else "\n".join(names)
    if not args:
        return HELP
    name, rest = args[0], args[1:]
    if command == "mute":
        await mutes.mute(name)
        return f"muted {name!r}"
    if command == "unmute":
        await mutes.unmute(name)
        return f"unmuted {name!r}"
    try:
        session = client.session(name)
        if command == "status":
            status = await daemon_call(session.status())
            lines = [f"{name} {'busy' if status.busy else 'idle'}"]
            if status.model is not None:
                lines.append(status.model)
            if status.permissions:
                lines.append(f"{len(status.permissions)} permission(s) waiting")
            if status.attention is not None:
                lines.append(f"attention: {status.attention}")
            return "\n".join(lines)
        if command == "tail":
            output = await _last_output(session)
            if output is None or (rendered := tool_text(output.output)) is None:
                return f"{name}: no output yet"
            return clip(rendered)
        if command == "send":
            if not rest:
                return HELP
            await daemon_call(client.send_message(name, " ".join(rest)))
            return f"sent to {name!r}"
    except AidError as error:
        return f"aid: {error.message}"
    return f"unknown command {command!r}; {HELP}"


async def serve(client: Client, bot: Bot, chats: Set[int], mutes: Mutes) -> None:
    """Answer allowlisted chats; everything else is dropped on the floor."""
    try:
        await bot.set_commands()
    except BotError, httpx.HTTPError:
        log.exception("telegram_commands_unpublished")
    offset = 0
    while True:
        try:
            updates, offset = await bot.updates(offset)
        except BotError as error:
            if (delay := poll_delay(error)) is None:
                raise SystemExit(
                    f"aid telegram: Telegram refused the updates ({error.status}: {error.description})"
                ) from error
            log.warning("telegram_poll_failed", error=error.description, retry_in=delay)
            await anyio.sleep(delay)
            continue
        except httpx.HTTPError as error:
            log.warning("telegram_network_failed", error=str(error))
            await anyio.sleep(RETRY_DELAY)
            continue
        for update in updates:
            if update.chat_id not in chats:
                log.info("telegram_chat_dropped", chat=update.chat_id)
                continue
            try:
                reply = await handle_command(client, mutes, update.text)
            except Exception:
                log.exception("telegram_command_failed", chat=update.chat_id)
                reply = "that failed; the log knows why"
            if reply is None:
                continue
            try:
                await bot.send(update.chat_id, reply)
            except BotError, httpx.HTTPError:
                log.exception("telegram_send_failed", chat=update.chat_id)


async def run_loops(client: Client, config: TelegramConfig, mutes: Mutes) -> None:
    """Both loops on the one client: the watch notices settled turns, the server answers chats."""
    bot = Bot(config.token)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(watch, client, bot, config.chats, mutes)
            tg.start_soon(serve, client, bot, config.chats, mutes)
    finally:
        await bot.aclose()
