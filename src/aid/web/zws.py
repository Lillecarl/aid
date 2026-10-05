"""The daemon's sockets for the page, as ZWS 2.0 (zeromq over WebSocket) through aid web.

Each WebSocket message is one frame: a flags byte (MORE, COMMAND), then the body; a multipart message is several,
MORE on all but the last. The subprotocol is bare `ZWS2.0`: no mechanism and no handshake, because the login and the
Origin check here are the security. libzmq's own ws listener and jszmq are not used: libzmq ignores the cookie and the
Origin, and runs no ZAP on bare ZWS2.0 (4.3.5, ws_engine.cpp).

- `/api/zws/control`: a DEALER on the daemon's control socket. Each request must be a `PageRequest`, which names
  what the page may ask; any other gets a Failure and never reaches the daemon. The relay gives each request an id of
  its own and puts the page's back on the replies.
- `/api/zws/events`: a SUB on the daemon's events (`aid.events`). A frame `\\x01topic` subscribes, `\\x00topic`
  unsubscribes, as in ZMTP 3.0.

A WebSocket stays authorised for its life: a logout does not close one already open. The daemon sees aid web, not the
person; nothing names them to it yet.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Annotated, Final

import anyio
import zmq
import zmq.asyncio
from pydantic import Field, TypeAdapter, ValidationError
from starlette.websockets import WebSocketDisconnect

from aid.protocol import (
    AnswerPermission,
    Cancel,
    CompactSession,
    CreateSession,
    DeleteSession,
    Done,
    Event,
    Failure,
    GetHistory,
    GetStatus,
    GetSummary,
    ListAgents,
    ListSessions,
    Prompt,
    StartSession,
    StopSession,
    decode_reply,
    encode,
)
from aid.web import auth

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from starlette.websockets import WebSocket

    from aid.client import Client

SUBPROTOCOL: Final = "ZWS2.0"
MORE: Final = 0x01
COMMAND: Final = 0x02
SUBSCRIBE: Final = 0x01
UNSUBSCRIBE: Final = 0x00
PING: Final = b"\x04PING"
PONG: Final = b"\x04PONG"
# ZMTP 3.1: PING is its name, a 2-byte TTL and up to 16 bytes of context, which PONG sends back.
PING_CONTEXT_AT: Final = len(PING) + 2
PING_CONTEXT_MAX: Final = 16
# A request is one frame; a subscription message is too, but a page may send several at once.
MAX_PARTS: Final = 64
WS_POLICY_VIOLATION: Final = 1008

type PageRequest = Annotated[
    ListSessions
    | ListAgents
    | CreateSession
    | GetStatus
    | GetHistory
    | GetSummary
    | Prompt
    | Cancel
    | CompactSession
    | AnswerPermission
    | StartSession
    | StopSession
    | DeleteSession,
    Field(discriminator="op"),
]
"""What the page may ask the daemon. Not `Hook` (a claude-tty worker's), `SendMessage` (it names its sender),
`ReceiveMessages` (it takes a session's messages), the pane requests (aid web relays the pane itself), nor the plugin
registry (it grants what the page itself cannot check)."""

type PageReply = Annotated[Event | Done | Failure, Field(discriminator="reply")]
"""What the daemon answers a page's request with."""

PageRequestAdapter: TypeAdapter[PageRequest] = TypeAdapter(PageRequest)


class ProtocolError(Exception):
    """The page broke ZWS: the relay closes the WebSocket."""


def frames(parts: Sequence[bytes]) -> list[bytes]:
    """One multipart message as WebSocket messages."""
    return [bytes([MORE if i < len(parts) - 1 else 0]) + part for i, part in enumerate(parts)]


def pong(command: bytes) -> bytes | None:
    """The PONG frame that answers a PING command's body, or None for any other command."""
    if not command.startswith(PING) or len(command) < PING_CONTEXT_AT:
        return None
    return bytes([COMMAND]) + PONG + command[PING_CONTEXT_AT:][:PING_CONTEXT_MAX]


def refusal(payload: bytes, message: str) -> bytes:
    """A Failure for a request the page may not send, under the request's id if it has one."""
    try:
        said = json.loads(payload)
    except ValueError:
        said = None
    found = said.get("id") if isinstance(said, dict) else None  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType] -- any JSON
    request_id = found if isinstance(found, str) else ""
    return encode(Failure(id=request_id, code="invalid_request", message=message))


class _Link:
    """One page's WebSocket: whole messages in, whole messages out. A send holds a lock, so a PONG never lands inside
    a multipart message."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._send_lock = anyio.Lock()

    async def send(self, *parts: bytes) -> None:
        async with self._send_lock:
            for frame in frames(parts):
                await self.websocket.send_bytes(frame)

    async def _send_frame(self, frame: bytes) -> None:
        async with self._send_lock:
            await self.websocket.send_bytes(frame)

    async def messages(self) -> AsyncIterator[list[bytes]]:
        """Each multipart message the page sends, until it disconnects. Commands are answered here, never yielded."""
        parts: list[bytes] = []
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data: bytes | None = message.get("bytes")
            if not data:
                raise ProtocolError("a ZWS frame is binary and holds its flags")
            flags, body = data[0], data[1:]
            if flags & COMMAND:
                if (answer := pong(body)) is not None:
                    await self._send_frame(answer)
                continue
            parts.append(body)
            if len(parts) > MAX_PARTS:
                raise ProtocolError("too many frames in one message")
            if not flags & MORE:
                yield parts
                parts = []


async def _accept(websocket: WebSocket) -> _Link | None:
    """The login and the Origin, as for every WebSocket here, and the ZWS subprotocol."""
    offered: list[str] = websocket.scope.get("subprotocols", [])
    if (
        auth.current_user(websocket) is None
        or not auth.same_origin(websocket, websocket.app.state.oidc_config)
        or SUBPROTOCOL not in offered
    ):
        await websocket.close(code=WS_POLICY_VIOLATION)
        return None
    await websocket.accept(subprotocol=SUBPROTOCOL)
    return _Link(websocket)


async def _relay(
    link: _Link,
    sock: zmq.asyncio.Socket,
    handle: Callable[[list[bytes]], Awaitable[None]],
    forward: Callable[[list[bytes]], list[bytes] | None] = lambda parts: parts,
) -> None:
    """The daemon's messages to the page through `forward` (None drops one), and the page's through `handle`, until
    either side ends. The socket stays in `app.state.open_sockets` throughout, so shutdown can close it."""
    link.websocket.app.state.open_sockets.add(link.websocket)
    try:
        problem: str | None = None
        try:
            async with anyio.create_task_group() as tg:

                async def to_page() -> None:
                    while True:
                        if (parts := forward(await sock.recv_multipart())) is not None:
                            await link.send(*parts)

                tg.start_soon(to_page)
                try:
                    async for parts in link.messages():
                        await handle(parts)
                except ProtocolError as error:
                    problem = str(error)
                tg.cancel_scope.cancel()
        except* WebSocketDisconnect:
            problem = None
        with contextlib.suppress(RuntimeError):  # Already closed by the page.
            if problem is None:
                await link.websocket.close()
            else:
                await link.websocket.close(code=WS_POLICY_VIOLATION, reason=problem)
    finally:
        link.websocket.app.state.open_sockets.discard(link.websocket)


async def control(websocket: WebSocket) -> None:
    if (link := await _accept(websocket)) is None:
        return
    client: Client = websocket.app.state.client
    # The daemon's request id for each of the page's. The daemon routes replies and names turns by request id,
    # whichever client sent it, so a page that chose one could take another client's replies.
    page_ids: dict[str, str] = {}
    async with client.open_socket(zmq.DEALER) as sock:

        async def handle(parts: list[bytes]) -> None:
            if len(parts) != 1:
                raise ProtocolError("a request is one frame")
            try:
                request = PageRequestAdapter.validate_json(parts[0])
            except ValidationError as error:
                await link.send(refusal(parts[0], str(error)))
                return
            ours = uuid.uuid4().hex
            page_ids[ours] = request.id
            await sock.send(encode(request.model_copy(update={"id": ours})))

        def forward(parts: list[bytes]) -> list[bytes] | None:
            reply = decode_reply(parts[0])
            if not isinstance(reply, Event | Done | Failure) or reply.id not in page_ids:
                return None
            page_id = page_ids[reply.id] if isinstance(reply, Event) else page_ids.pop(reply.id)
            return [encode(reply.model_copy(update={"id": page_id}))]

        await _relay(link, sock, handle, forward)


async def events(websocket: WebSocket) -> None:
    if (link := await _accept(websocket)) is None:
        return
    client: Client = websocket.app.state.client
    async with client.open_socket(zmq.SUB) as sock:

        async def handle(parts: list[bytes]) -> None:
            for part in parts:
                if part[:1] == bytes([SUBSCRIBE]):
                    sock.setsockopt(zmq.SUBSCRIBE, part[1:])
                elif part[:1] == bytes([UNSUBSCRIBE]):
                    sock.setsockopt(zmq.UNSUBSCRIBE, part[1:])
                else:
                    raise ProtocolError("an events frame subscribes (\\x01) or unsubscribes (\\x00)")

        await _relay(link, sock, handle)
