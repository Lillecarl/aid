"""Discord gateway over anyio streams: wsproto frames the wire, anyio owns the sockets.

No Discord library: the maintained ones own an asyncio event loop, which a supervised task cannot give
them, and buy far more than a session bot needs. The gateway surface here is small — hello, identify,
heartbeats, resume, dispatches — and wsproto (in the closure via httpx) frames it with no loop of its own.
JSON stays the encoding, as on every other aid wire: throughput is control-plane either way.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlsplit

import anyio
import structlog
from anyio.streams.tls import TLSStream
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
from wsproto.events import (
    AcceptConnection,
    BytesMessage,
    CloseConnection,
    Ping,
    Pong,
    Request,
    TextMessage,
)
from wsproto.utilities import RemoteProtocolError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anyio.abc import AnyByteStream
    from pydantic import JsonValue

log = structlog.get_logger(__name__)

GATEWAY_URL: Final = "wss://gateway.discord.gg/?v=10&encoding=json"
INTENTS: Final = (1 << 0) | (1 << 9) | (1 << 12) | (1 << 15)
"""Guilds, guild messages, direct messages, and message content: what session channels and DMs need."""
FATAL_CLOSE: Final = frozenset({4004, 4010, 4011, 4012, 4013, 4014})
"""Token, shard, version and intent death: reconnecting never heals these."""
_FIRST_BACKOFF: Final = 1.0
_MAX_BACKOFF: Final = 60.0
"""Fresh identifies are globally limited (1000 a day): back off across consecutive failures."""


class WSClosed(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(f"websocket closed {code}: {reason or 'no reason'}")
        self.code = code
        self.reason = reason


class FatalGatewayError(Exception):
    """Token or config death: reconnecting never heals it, so the plugin exits loudly instead."""


class _Resume(Exception):
    """The session dropped but lives server-side: reconnect and resume."""


class _Fresh(Exception):
    """Start over with a fresh identify: the session is gone or never was."""


def _unwrap_session(group: BaseExceptionGroup) -> BaseException:
    """A task-group unwind's meaning: the severest control exception it holds (a receive outcome beats its
    heartbeat's collateral crash), cancellation if that is all it holds, else the group itself."""
    for kind in (FatalGatewayError, _Fresh, _Resume):
        for error in group.exceptions:
            if isinstance(error, kind):
                return error
    if all(isinstance(error, anyio.get_cancelled_exc_class()) for error in group.exceptions):
        return group.exceptions[0]
    return group


class Channel(Protocol):
    """What the gateway speaks to: a real `WebSocket`, or a scripted fake in tests."""

    async def send_json(self, payload: JsonValue) -> None: ...
    async def recv_json(self) -> JsonValue: ...
    async def aclose(self, code: int | None = 1000) -> None: ...


class WebSocket(Channel):
    """A client websocket over an anyio byte stream. Answers pings, reassembles fragments, raises
    `WSClosed` on close or EOF. `open` dials one (`ws://` plain for loopback tests, `wss://` with TLS)."""

    def __init__(self, stream: AnyByteStream, ws: WSConnection) -> None:
        self._stream = stream
        self._ws = ws
        self._send_lock = anyio.Lock()

    @classmethod
    async def open(cls, url: str) -> WebSocket:
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss") or not parts.hostname:
            raise ValueError(f"{url!r} is no websocket URL")
        target = parts.path or "/"
        if parts.query:
            target += f"?{parts.query}"
        tcp = await anyio.connect_tcp(parts.hostname, parts.port or 443)
        stream: AnyByteStream = tcp
        if parts.scheme == "wss":
            try:
                stream = await TLSStream.wrap(tcp, hostname=parts.hostname)
            except BaseException:
                await tcp.aclose()
                raise
        ws = WSConnection(ConnectionType.CLIENT)
        try:
            await stream.send(ws.send(Request(host=parts.hostname, target=target)))
            head = bytearray()
            while b"\r\n\r\n" not in head:
                try:
                    head += await stream.receive()
                except anyio.EndOfStream, anyio.ClosedResourceError:
                    raise WSClosed(1006, "eof during handshake") from None
            try:
                ws.receive_data(bytes(head))
            except RemoteProtocolError as error:
                raise WSClosed(1006, f"handshake refused: {error}") from error
        except BaseException:
            await stream.aclose()
            raise
        for event in ws.events():
            if isinstance(event, AcceptConnection):
                return cls(stream, ws)
        await stream.aclose()
        raise WSClosed(1006, "handshake refused")

    async def send_json(self, payload: JsonValue) -> None:
        async with self._send_lock:
            await self._stream.send(self._ws.send(TextMessage(json.dumps(payload))))

    async def recv_json(self) -> JsonValue:
        """The next whole text message: answers pings, reassembles fragments, raises `WSClosed` on close."""
        parts: list[str] = []
        while True:
            try:
                self._ws.receive_data(await self._stream.receive())
            except anyio.EndOfStream, anyio.ClosedResourceError:
                raise WSClosed(1006, "eof") from None
            for event in self._ws.events():
                if isinstance(event, TextMessage):
                    parts.append(event.data)
                    if event.message_finished:
                        text: JsonValue = json.loads("".join(parts))
                        return text
                elif isinstance(event, BytesMessage):
                    continue  # JSON encoding negotiated: binary never arrives.
                elif isinstance(event, Ping):
                    async with self._send_lock:
                        await self._stream.send(self._ws.send(Pong(event.payload)))
                elif isinstance(event, CloseConnection):
                    await self._answer_close(event)
                    raise WSClosed(event.code, event.reason or "")

    async def _answer_close(self, event: CloseConnection) -> None:
        """Echo the close frame while the connection still takes one, best-effort either way."""
        try:
            if self._ws.state is ConnectionState.OPEN:
                async with self._send_lock:
                    await self._stream.send(self._ws.send(event.response()))
        except Exception:
            log.warning("discord_close_unanswered")

    async def aclose(self, code: int | None = 1000) -> None:
        """Close the stream, with a close frame unless `code` is None: a bare drop keeps the session
        resumable, while 1000 invalidates it (a clean offline on shutdown)."""
        if code is not None:
            try:
                async with self._send_lock:
                    await self._stream.send(self._ws.send(CloseConnection(code)))
            except Exception:
                log.warning("discord_close_unsent")
        await self._stream.aclose()


@dataclass
class _Heart:
    """One session's heartbeat state, shared by its heartbeat and receive tasks."""

    acked: bool = True
    beat: anyio.Event = field(default_factory=anyio.Event)


class Gateway:
    """One bot's gateway session: connects, identifies or resumes, and calls `on_dispatch` per dispatch.

    Reconnects internally; raises `FatalGatewayError` on token or config death. Handler failures drop the
    session and resume past the failed event: side effects happen at most once, losses log loudly.
    """

    def __init__(
        self, token: str, *, url: str = GATEWAY_URL, open_ws: Callable[[str], Awaitable[Channel]] | None = None
    ) -> None:
        self._token = token
        self._url = url
        self._open = open_ws if open_ws is not None else WebSocket.open
        self._seq: int | None = None
        self._session_id: str | None = None
        self._resume_url = url
        self._fresh_backoff = _FIRST_BACKOFF

    async def run(self, on_dispatch: Callable[[str, JsonValue], Awaitable[None]]) -> None:
        """Hold the session until cancelled."""
        can_resume = False
        while True:
            try:
                if can_resume and self._session_id is not None:
                    await self._session(self._resume_url, True, on_dispatch)
                else:
                    await self._session(self._url, False, on_dispatch)
            except _Resume as dropped:
                log.info("discord_reconnect", reason=str(dropped))
                can_resume = True
            except _Fresh as stale:
                log.warning("discord_reidentify", reason=str(stale))
                can_resume = False
                self._session_id = None
                self._seq = None
                await anyio.sleep(self._fresh_backoff + random.uniform(0, 1))
                self._fresh_backoff = min(self._fresh_backoff * 2, _MAX_BACKOFF)

    async def _session(self, url: str, resume: bool, on_dispatch: Callable[[str, JsonValue], Awaitable[None]]) -> None:
        """One websocket lifetime: hello, identify or resume, then dispatches. Raises `_Resume`, `_Fresh`
        or `FatalGatewayError`; never returns."""
        ws = await self._open(url)
        closing: int | None = 1000
        try:
            hello = await ws.recv_json()
            details = hello.get("d") if isinstance(hello, dict) else None
            interval_ms = details.get("heartbeat_interval") if isinstance(details, dict) else None
            if not isinstance(interval_ms, (int, float)) or isinstance(interval_ms, bool) or interval_ms <= 0:
                raise _Fresh(f"no heartbeat interval: {hello!r}")
            heart = _Heart()
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._heartbeat, ws, interval_ms / 1000, heart)
                    if resume and self._session_id is not None:
                        await ws.send_json(
                            {
                                "op": 6,
                                "d": {"token": self._token, "session_id": self._session_id, "seq": self._seq},
                            }
                        )
                    else:
                        self._session_id = None
                        self._seq = None
                        await ws.send_json(
                            {
                                "op": 2,
                                "d": {
                                    "token": self._token,
                                    "intents": INTENTS,
                                    "properties": {"os": sys.platform, "browser": "aid", "device": "aid"},
                                },
                            }
                        )
                    try:
                        await self._receive(ws, on_dispatch, heart)
                    except _Resume, _Fresh, FatalGatewayError:
                        raise
                    except anyio.get_cancelled_exc_class():
                        raise
                    except Exception as error:
                        log.exception("discord_receive_failed", error=f"{type(error).__name__}: {error}")
                        raise _Resume(f"receive failed: {error}") from error
            except BaseExceptionGroup as group:
                raise _unwrap_session(group) from None
        except _Resume, _Fresh:
            closing = None
            raise
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await ws.aclose(closing)
                except Exception:
                    log.warning("discord_close_failed")

    async def _heartbeat(self, ws: Channel, interval: float, heart: _Heart) -> None:
        """Heartbeats until cancelled or zombied: a missing ack drops the TCP bare (resumable) and returns,
        leaving the receive side to drive the resume."""
        await anyio.sleep(random.uniform(0, interval))
        while True:
            if not heart.acked:
                await ws.aclose(None)
                return
            heart.acked = False
            heart.beat = anyio.Event()
            try:
                await ws.send_json({"op": 1, "d": self._seq})
            except anyio.get_cancelled_exc_class():
                raise
            except Exception as error:
                raise _Resume(f"heartbeat unsent: {error}") from error
            with anyio.move_on_after(interval):
                await heart.beat.wait()

    async def _receive(
        self, ws: Channel, on_dispatch: Callable[[str, JsonValue], Awaitable[None]], heart: _Heart
    ) -> None:
        """Dispatches until the connection dies: raises `_Resume`, `_Fresh` or `FatalGatewayError`."""
        while True:
            try:
                payload = await ws.recv_json()
            except WSClosed as closed:
                if closed.code in FATAL_CLOSE:
                    raise FatalGatewayError(f"discord refused the gateway ({closed.code}: {closed.reason})") from closed
                if closed.code in (1000, 1001):
                    raise _Fresh(f"session invalidated ({closed.code})") from closed
                raise _Resume(f"connection dropped ({closed.code})") from closed
            if not isinstance(payload, dict):
                continue
            op = payload.get("op")
            if op == 0:
                event = payload.get("t")
                if not isinstance(event, str):
                    continue
                seq = payload.get("s")
                if isinstance(seq, int) and not isinstance(seq, bool):
                    self._seq = seq
                data = payload.get("d")
                if event == "READY" and isinstance(data, dict):
                    session_id = data.get("session_id")
                    resume_url = data.get("resume_gateway_url")
                    if not isinstance(session_id, str):
                        raise _Fresh("ready names no session")
                    self._session_id = session_id
                    self._resume_url = resume_url if isinstance(resume_url, str) else GATEWAY_URL
                    self._fresh_backoff = _FIRST_BACKOFF
                elif event == "RESUMED":
                    self._fresh_backoff = _FIRST_BACKOFF
                else:
                    try:
                        await on_dispatch(event, data if data is not None else {})
                    except anyio.get_cancelled_exc_class():
                        raise
                    except Exception:
                        log.exception("discord_dispatch_failed", event=event)
            elif op == 1:
                heart.beat.set()
            elif op == 7:
                raise _Resume("server asked to reconnect")
            elif op == 9:
                if payload.get("d") is True:
                    raise _Resume("invalid session, resumable")
                raise _Fresh("invalid session, start anew")
            elif op == 11:
                heart.acked = True
            elif op == 10:
                continue  # A hello mid-session answers nothing.
            else:
                log.warning("discord_unknown_op", op=op)
