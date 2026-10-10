"""The Discord gateway: handshake, heartbeats, resume and death, against scripted wires."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import anyio
import pytest
from anyio.abc import SocketAttribute
from wsproto import ConnectionType, WSConnection
from wsproto.events import AcceptConnection, Request, TextMessage

from plugins.discord.gateway import (
    INTENTS,
    FatalGatewayError,
    Gateway,
    WebSocket,
    WSClosed,
)

if TYPE_CHECKING:
    from anyio.abc import AnyByteStream, SocketStream
    from pydantic import JsonValue

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def _hello(interval_ms: int = 60) -> JsonValue:
    return {"op": 10, "d": {"heartbeat_interval": interval_ms}}


def _ready(seq: int = 1) -> JsonValue:
    return {
        "op": 0,
        "s": seq,
        "t": "READY",
        "d": {
            "session_id": "s1",
            "resume_gateway_url": "wss://resume.example/?v=10&encoding=json",
            "user": {},
            "guilds": [],
        },
    }


async def _ignore(event: str, data: JsonValue) -> None:
    del event, data


class ScriptWS:
    """A scripted websocket: queued payloads out, recorded payloads in, closes recorded. Heartbeats are
    acked like the real gateway, unless `ack` is off (the zombie test). Queued payloads wake the reader,
    as network bytes would."""

    def __init__(self, incoming: list[JsonValue | WSClosed], *, ack: bool = True) -> None:
        self._incoming = incoming
        self._ack = ack
        self._arrived = anyio.Event()
        self.sent: list[JsonValue] = []
        self.closed: list[int | None] = []

    async def send_json(self, payload: JsonValue) -> None:
        self.sent.append(payload)
        if self._ack and isinstance(payload, dict) and payload.get("op") == 1:
            self._incoming.insert(0, {"op": 11})
        self._arrived.set()

    async def recv_json(self) -> JsonValue:
        while not self._incoming:
            await self._arrived.wait()
        item = self._incoming.pop(0)
        if isinstance(item, WSClosed):
            raise item
        return item

    async def aclose(self, code: int | None = 1000) -> None:
        self.closed.append(code)


class Opener:
    """Hands scripted websockets to one gateway, recording the URLs it dials."""

    def __init__(self, *conns: ScriptWS) -> None:
        self.urls: list[str] = []
        self._conns = list(conns)

    async def __call__(self, url: str) -> ScriptWS:
        self.urls.append(url)
        return self._conns.pop(0)


def _beats(sent: list[JsonValue]) -> list[JsonValue]:
    return [payload for payload in sent if isinstance(payload, dict) and payload.get("op") == 1]


async def test_identifies_dispatches_and_resumes() -> None:
    first = ScriptWS(
        [
            _hello(),
            _ready(),
            {"op": 0, "s": 2, "t": "MESSAGE_CREATE", "d": {"id": "m1", "content": "hi"}},
            WSClosed(1006, "boom"),
        ]
    )
    second = ScriptWS([_hello()])
    opener = Opener(first, second)
    dispatched: list[tuple[str, JsonValue]] = []

    async def on_dispatch(event: str, data: JsonValue) -> None:
        dispatched.append((event, data))

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            tg.start_soon(Gateway("tok", open_ws=opener).run, on_dispatch)
            await anyio.sleep(0.5)
            tg.cancel_scope.cancel()
    assert first.sent[0] == {
        "op": 2,
        "d": {
            "token": "tok",
            "intents": INTENTS,
            "properties": {"os": sys.platform, "browser": "aid", "device": "aid"},
        },
    }
    beats = _beats(second.sent)
    assert len(beats) >= 2
    assert all(isinstance(beat, dict) and beat.get("d") == 2 for beat in beats)
    assert dispatched == [("MESSAGE_CREATE", {"id": "m1", "content": "hi"})]
    assert opener.urls == [
        "wss://gateway.discord.gg/?v=10&encoding=json",
        "wss://resume.example/?v=10&encoding=json",
    ]
    assert second.sent[0] == {"op": 6, "d": {"token": "tok", "session_id": "s1", "seq": 2}}
    assert first.closed == [None]  # A resume keeps the session: the TCP drops bare, no close frame.


async def test_server_reconnect_asks_resume() -> None:
    first = ScriptWS([_hello(), _ready(), {"op": 7}])
    second = ScriptWS([_hello()])
    opener = Opener(first, second)
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            tg.start_soon(Gateway("tok", open_ws=opener).run, _ignore)
            await anyio.sleep(0.3)
            tg.cancel_scope.cancel()
    assert isinstance(second.sent[0], dict) and second.sent[0].get("op") == 6


async def test_invalid_session_starts_fresh() -> None:
    first = ScriptWS([_hello(), {"op": 9, "d": False}])
    second = ScriptWS([_hello()])
    opener = Opener(first, second)
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            tg.start_soon(Gateway("tok", open_ws=opener).run, _ignore)
            await anyio.sleep(2.5)  # The fresh identify backs off past a jittered second.
            tg.cancel_scope.cancel()
    assert opener.urls[1] == "wss://gateway.discord.gg/?v=10&encoding=json"
    assert isinstance(second.sent[0], dict) and second.sent[0].get("op") == 2


async def test_bad_token_exits_loudly() -> None:
    opener = Opener(ScriptWS([_hello(), WSClosed(4004, "auth")]))
    with anyio.fail_after(TIMEOUT):
        with pytest.raises(FatalGatewayError, match="4004"):
            await Gateway("bad", open_ws=opener).run(_ignore)


async def test_zombie_drops_bare_and_resumes() -> None:
    first = ScriptWS([_hello(50), _ready()], ack=False)
    opener = Opener(first, ScriptWS([_hello()]))
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            tg.start_soon(Gateway("tok", open_ws=opener).run, _ignore)
            await anyio.sleep(0.4)
            tg.cancel_scope.cancel()
    assert len(_beats(first.sent)) == 1  # The second check finds no ack and drops instead of beating.
    assert first.closed[0] is None  # Teardown closes clean (1000) after the zombie drop.


async def test_websocket_handshake_validates_and_frames() -> None:
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0)
    port = listener.extra(SocketAttribute.local_port)
    server_ws = WSConnection(ConnectionType.SERVER)
    try:
        async with anyio.create_task_group() as tg:

            async def serve(stream: SocketStream) -> None:
                await _answer_upgrade(stream, server_ws)
                while True:
                    server_ws.receive_data(await stream.receive())
                    if any(isinstance(event, TextMessage) for event in server_ws.events()):
                        break
                await stream.send(server_ws.send(TextMessage('{"op": 11}')))

            tg.start_soon(listener.serve, serve)  # pyrefly: ignore[bad-argument-type]
            client = await WebSocket.open(f"ws://127.0.0.1:{port}/?v=10&encoding=json")
            await client.send_json({"op": 1, "d": None})
            assert await client.recv_json() == {"op": 11}
            tg.cancel_scope.cancel()
    finally:
        await listener.aclose()


async def _answer_upgrade(stream: AnyByteStream, server_ws: WSConnection) -> None:
    head = bytearray()
    while b"\r\n\r\n" not in head:
        head += await stream.receive()
    server_ws.receive_data(bytes(head))
    for event in server_ws.events():
        if isinstance(event, Request):
            await stream.send(server_ws.send(AcceptConnection()))
            return
    raise AssertionError("no upgrade request")


async def test_websocket_handshake_refuses_a_bad_status() -> None:
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0)
    port = listener.extra(SocketAttribute.local_port)
    try:
        async with anyio.create_task_group() as tg:

            async def serve(stream: SocketStream) -> None:
                await stream.receive()
                await stream.send(b"HTTP/1.1 400 Bad\r\nContent-Length: 0\r\n\r\n")

            tg.start_soon(listener.serve, serve)  # pyrefly: ignore[bad-argument-type]
            with pytest.raises(WSClosed, match="handshake refused"):
                await WebSocket.open(f"ws://127.0.0.1:{port}/")
            tg.cancel_scope.cancel()
    finally:
        await listener.aclose()
