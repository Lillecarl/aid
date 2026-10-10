"""The Discord plugin's loops: gateway dispatches in, REST out, settled notices to session channels.

Session channels take bare text as messages to their session; `!`-commands work everywhere (DMs and
guild channels alike), defaulting to the channel's session when argless. A DM takes commands only:
bare text there is ignored, as in Telegram. Taps answer in place or anew; stale ones toast alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import anyio
import httpx
import structlog
from pydantic import BaseModel, ConfigDict

from aid.client import daemon_call
from aid.protocol import AidError, Output, TurnError, tool_text
from aid.settle import IDLE_WINDOW, TICK_INTERVAL, VERDICT_TIMEOUT, describe_end, settled_turns

from .channels import reconcile, state_word
from .gateway import Gateway
from .rest import DiscordError, Rest, action_rows

if TYPE_CHECKING:
    from pydantic import JsonValue

    from aid.client import Client, Session
    from aid.mutes import Mutes

    from .channels import ChannelMap

log = structlog.get_logger(__name__)

TOKEN_ENV: Final = "AID_DISCORD_TOKEN"
GUILD_ENV: Final = "AID_DISCORD_GUILD"
CATEGORY_ENV: Final = "AID_DISCORD_CATEGORY"
DEFAULT_CATEGORY: Final = "aid sessions"
MUTES_FILE: Final = "mutes.json"
TAIL_SCAN: Final = 20
"""Newest history entries searched for the turn's output when composing a notice."""
_SUBSCRIBE_IDLE: Final = 60.0
"""Seconds a quiet reconcile loop waits per poll: publishes still arrive at once, and cancellation lands
in the poll rather than in a bare receive, which pyzmq cannot unwind cleanly."""
NOTICE_LIMIT: Final = 1900
"""Discord messages hold 2000 characters: notices clip under it with room for the head."""
HELP: Final = """!sessions, !status [name], !tail [name], !send <name> <text>, !mute [name], !unmute [name], !muted"""
STALE_TAP: Final = "Those buttons expired — ask again"


class DiscordConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    guild_id: str
    category: str = DEFAULT_CATEGORY


def read_config() -> DiscordConfig:
    """The bot token and guild, dropped from the environment once read so neither reaches a worker."""
    token = os.environ.pop(TOKEN_ENV, "")
    if not token:
        raise AidError("no_token", f"{TOKEN_ENV} names no bot token")
    guild = os.environ.pop(GUILD_ENV, "").strip()
    if not guild.isdigit():
        raise AidError("bad_guild", f"{GUILD_ENV} holds a guild snowflake")
    category = os.environ.pop(CATEGORY_ENV, DEFAULT_CATEGORY).strip() or DEFAULT_CATEGORY
    return DiscordConfig(token=token, guild_id=guild, category=category)


@dataclass(frozen=True)
class Reply:
    """What answers a command or a tap: content plus optional buttons; a tap may rewrite instead (`update`)
    or toast the tapper alone (`ephemeral`)."""

    content: str
    components: JsonValue | None = None
    update: bool = False
    ephemeral: bool = False


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """A Discord message as a command and its arguments, or None when it holds none."""
    if not text.startswith("!"):
        return None
    head, *rest = text[1:].split()
    return head.split("@")[0], rest


def _clip(text: str) -> str:
    if len(text) <= NOTICE_LIMIT:
        return text
    return f"{text[:NOTICE_LIMIT]}\n… {len(text) - NOTICE_LIMIT} more characters"


def fence_output(text: str) -> str:
    """Wrap machine output in triple-backtick fences; inner fences split with a zero-width space survive."""
    return f"```\n{text.replace('```', '``\u200b`')}\n```"


async def _last_output(session: Session) -> Output | None:
    page = await daemon_call(session.history(limit=TAIL_SCAN))
    for entry in reversed(page.entries):
        if isinstance(entry.item, Output):
            return entry.item
    return None


async def _last_terminal(session: Session) -> Output | TurnError | None:
    """The turn's newest terminal entry: its output, or its failure when that came last. One history
    read serves the whole notice."""
    page = await daemon_call(session.history(limit=TAIL_SCAN))
    for entry in reversed(page.entries):
        if isinstance(entry.item, (Output, TurnError)):
            return entry.item
    return None


async def _status_text(session: Session, name: str) -> str:
    status = await daemon_call(session.status())
    lines = [f"{name} {'busy' if status.busy else 'idle'}"]
    if status.model is not None:
        lines.append(status.model)
    if status.permissions:
        lines.append(f"{len(status.permissions)} permission(s) waiting")
    if status.attention is not None:
        lines.append(f"attention: {status.attention}")
    return "\n".join(lines)


async def _tail_text(session: Session, name: str) -> str:
    output = await _last_output(session)
    if output is None or (rendered := tool_text(output.output)) is None:
        return f"{name}: no output yet"
    return fence_output(_clip(rendered))


async def _notice_text(session: Session, reason: str | None) -> str:
    """What a session channel hears about a settled turn: how it ended, and what it said last."""
    head = f"{session.name}: {describe_end(reason)}"
    try:
        status = await daemon_call(session.status())
        terminal = await _last_terminal(session)
    except AidError as error:
        return f"{head}\ngone since: {error.message}"
    lines = [head]
    if isinstance(terminal, TurnError):
        lines.append(fence_output(_clip(terminal.message)))
    elif terminal is not None and (text := tool_text(terminal.output)) is not None:
        lines.append(fence_output(_clip(text)))
    if status.model is not None:
        lines.append(status.model)
    if status.cost:
        lines.append(", ".join(f"{amount:g} {currency}" for currency, amount in sorted(status.cost.items())))
    return "\n".join(lines)


async def _sessions_reply(client: Client, mutes: Mutes) -> Reply:
    """The session list, each row tapping to its status, tail, and mute state."""
    infos = await daemon_call(client.sessions())
    if not infos:
        return Reply("no sessions")
    rows = tuple(
        (
            (info.name, f"status:{info.name}"),
            ("tail", f"tail:{info.name}"),
            (
                "unmute" if info.name in mutes else "mute",
                f"{'unmute' if info.name in mutes else 'mute'}:{info.name}",
            ),
        )
        for info in infos
    )
    text = "\n".join(f"{info.name} {state_word(info)}{' muted' if info.name in mutes else ''}" for info in infos)
    return Reply(text, action_rows(rows))


async def handle_command(client: Client, mutes: Mutes, text: str, *, session: str | None) -> Reply | None:
    """An inbound message as a reply, or None when it holds no command. Argless reads default to the
    channel's session; `send` always names its session, so a wake never lands somewhere unmeant."""
    parsed = parse_command(text)
    if parsed is None:
        return None
    command, args = parsed
    if command in ("start", "help"):
        return Reply(HELP)
    if command == "sessions":
        return await _sessions_reply(client, mutes)
    if command == "muted":
        names = mutes.names()
        if not names:
            return Reply("nothing muted")
        rows = tuple(((f"unmute {name}", f"unmute:{name}"),) for name in names)
        return Reply("\n".join(names), action_rows(rows))
    name = args[0] if args else session
    if name is None:
        return Reply(HELP)
    rest = args[1:] if args else []
    if command == "mute":
        await mutes.mute(name)
        return Reply(f"muted {name!r}")
    if command == "unmute":
        await mutes.unmute(name)
        return Reply(f"unmuted {name!r}")
    try:
        target = client.session(name)
        if command == "status":
            return Reply(await _status_text(target, name))
        if command == "tail":
            return Reply(await _tail_text(target, name))
        if command == "send":
            if not rest:
                return Reply(HELP)
            await daemon_call(client.send_message(name, " ".join(rest)))
            return Reply(f"sent to {name!r}")
    except AidError as error:
        return Reply(f"aid: {error.message}")
    return Reply(f"unknown command {command!r}; {HELP}")


async def handle_tap(client: Client, mutes: Mutes, custom_id: str) -> Reply:
    """A button tap as a reply: reads answer anew, mute taps rewrite with fresh buttons, unknown taps
    toast alone."""
    action, sep, name = custom_id.partition(":")
    if not sep or not name or action not in ("status", "tail", "mute", "unmute"):
        return Reply(STALE_TAP, ephemeral=True)
    try:
        target = client.session(name)
        if action == "status":
            return Reply(await _status_text(target, name))
        if action == "tail":
            return Reply(await _tail_text(target, name))
        if action == "mute":
            await mutes.mute(name)
        else:
            await mutes.unmute(name)
        refreshed = await _sessions_reply(client, mutes)
        return Reply(refreshed.content, refreshed.components, update=True)
    except AidError as error:
        return Reply(f"aid: {error.message}", ephemeral=True)


async def on_message(
    client: Client, rest: Rest, mutes: Mutes, channel_map: ChannelMap, *, channel_id: str, content: str
) -> None:
    """A message in a heard channel: commands everywhere, bare text only in session channels."""
    session = channel_map.channel_of(channel_id)
    if parse_command(content) is None:
        if session is None:
            return
        await daemon_call(client.send_message(session, content))
        return
    reply = await handle_command(client, mutes, content, session=session)
    if reply is None:
        return
    await rest.send_message(channel_id, reply.content, components=reply.components)


async def on_interaction(
    client: Client, rest: Rest, mutes: Mutes, *, interaction_id: str, token: str, custom_id: str
) -> None:
    reply = await handle_tap(client, mutes, custom_id)
    await rest.answer_interaction(
        interaction_id,
        token,
        reply.content,
        update=reply.update,
        ephemeral=reply.ephemeral,
        components=reply.components,
    )


async def dispatch(
    client: Client,
    rest: Rest,
    mutes: Mutes,
    channel_map: ChannelMap,
    guild_id: str,
    category: str,
    event: str,
    data: JsonValue,
) -> None:
    """One gateway dispatch: messages, taps and channel deletions in, REST out. Anything else passes."""
    try:
        if event == "MESSAGE_CREATE" and isinstance(data, dict):
            author = data.get("author")
            if isinstance(author, dict) and author.get("bot") is True:
                return
            channel_id = data.get("channel_id")
            content = data.get("content", "")
            if not isinstance(channel_id, str) or not isinstance(content, str):
                return
            if data.get("guild_id") not in (None, guild_id):
                return
            await on_message(client, rest, mutes, channel_map, channel_id=channel_id, content=content)
        elif event == "INTERACTION_CREATE" and isinstance(data, dict):
            interaction_id = data.get("id")
            token = data.get("token")
            inner = data.get("data")
            custom_id = inner.get("custom_id") if isinstance(inner, dict) else None
            if not isinstance(interaction_id, str) or not isinstance(token, str) or not isinstance(custom_id, str):
                return
            await on_interaction(client, rest, mutes, interaction_id=interaction_id, token=token, custom_id=custom_id)
        elif event == "CHANNEL_DELETE" and isinstance(data, dict):
            channel_id = data.get("id")
            if isinstance(channel_id, str) and channel_map.forget_channel(channel_id) is not None:
                await reconcile(client, rest, guild_id, category, channel_map)
    except anyio.get_cancelled_exc_class():
        raise
    except Exception:
        log.exception("discord_dispatch_failed", event=event)


async def channel_loop(client: Client, rest: Rest, guild_id: str, category: str, channel_map: ChannelMap) -> None:
    """Reconcile now, then on every session-list change; a failure logs and waits for the next change."""
    try:
        await reconcile(client, rest, guild_id, category, channel_map)
    except anyio.get_cancelled_exc_class():
        raise
    except Exception:
        log.exception("discord_reconcile_failed")
    async with client.subscribed(b"sessions/", idle=_SUBSCRIBE_IDLE) as published:
        async for message in published:
            if message is None:
                continue  # The idle wakes only so cancellation lands cleanly; publishes arrive at once.
            try:
                await reconcile(client, rest, guild_id, category, channel_map)
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                log.exception("discord_reconcile_failed")


async def notice_loop(
    client: Client,
    rest: Rest,
    mutes: Mutes,
    channel_map: ChannelMap,
    *,
    idle_window: float = IDLE_WINDOW,
    verdict_timeout: float = VERDICT_TIMEOUT,
    tick: float = TICK_INTERVAL,
) -> None:
    """Post settled turns to their session channels; muted sessions and missing channels stay silent."""

    async def notify(session: str, reason: str | None) -> None:
        if session in mutes:
            return
        channel_id = channel_map.channels.get(session)
        if channel_id is None:
            log.warning("discord_no_channel", session=session)
            return
        try:
            text = await _notice_text(client.session(session), reason)
        except Exception:
            log.exception("discord_notice_failed", session=session)
            return
        try:
            await rest.send_message(channel_id, text, components=action_rows(((("mute", f"mute:{session}"),),)))
        except DiscordError, httpx.HTTPError:
            log.exception("discord_send_failed", session=session)

    await settled_turns(client, notify, idle_window=idle_window, verdict_timeout=verdict_timeout, tick=tick)


async def run_loops(client: Client, config: DiscordConfig, mutes: Mutes, channel_map: ChannelMap) -> None:
    """Three loops on the one client: the gateway hears, channels reconcile, settled turns post."""
    rest = Rest(config.token)
    try:
        try:
            url = await rest.gateway_url()
        except DiscordError as error:
            raise SystemExit(f"aid discord: the token proved bad ({error.status}: {error.message})") from error

        async def on_dispatch(event: str, data: JsonValue) -> None:
            await dispatch(client, rest, mutes, channel_map, config.guild_id, config.category, event, data)

        gateway = Gateway(config.token, url=url)
        async with anyio.create_task_group() as tg:
            tg.start_soon(gateway.run, on_dispatch)
            tg.start_soon(channel_loop, client, rest, config.guild_id, config.category, channel_map)
            tg.start_soon(notice_loop, client, rest, mutes, channel_map)
    finally:
        await rest.aclose()
