"""CURVE keys and the ZAP handler for the workers socket.

Every worker connection is CURVE, local ones too, so a remote transport changes nothing here. The daemon issues
a keypair per worker launch; the ZAP handler accepts only issued keys and names the session as the connection's
User-Id. The daemon routes by that, never by a routing id a worker picks: that id is a claim, the key is proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import zmq
import zmq.asyncio
import zmq.utils.z85

ZAP_ENDPOINT: Final = "inproc://zeromq.zap.01"
DOMAIN: Final = b"aid-workers"
_VERSION: Final = b"1.0"


@dataclass(frozen=True)
class Keypair:
    """Z85 text, so it travels in `WorkerArgs`."""

    public: str
    secret: str

    @classmethod
    def new(cls) -> Keypair:
        public, secret = zmq.curve_keypair()
        return cls(public=public.decode(), secret=secret.decode())


class Keys:
    """The keys issued to running workers, by session name."""

    def __init__(self) -> None:
        self.server = Keypair.new()
        self._sessions: dict[str, str] = {}

    def issue(self, session: str) -> Keypair:
        keys = Keypair.new()
        self._sessions = {k: v for k, v in self._sessions.items() if v != session} | {keys.public: session}
        return keys

    def revoke(self, session: str, public: str) -> None:
        if self._sessions.get(public) == session:
            del self._sessions[public]

    def reply(self, request: list[bytes]) -> list[bytes]:
        """A ZAP reply (RFC 27): 200 with the session as User-Id for an issued CURVE key, else 400."""
        version, request_id, domain, _address, _identity, mechanism, *credentials = request
        session = None
        if version == _VERSION and domain == DOMAIN and mechanism == b"CURVE" and len(credentials) == 1:
            key = zmq.utils.z85.encode(credentials[0])  # pyright: ignore[reportUnknownMemberType] -- untyped in pyzmq
            session = self._sessions.get(key.decode())
        if session is None:
            return [_VERSION, request_id, b"400", b"key not issued", b"", b""]
        return [_VERSION, request_id, b"200", b"", session.encode(), b""]


def serve_curve(sock: zmq.asyncio.Socket, keys: Keys) -> None:
    sock.curve_server = True
    sock.curve_secretkey = keys.server.secret.encode()
    sock.curve_publickey = keys.server.public.encode()
    sock.zap_domain = DOMAIN


async def handle(zap: zmq.asyncio.Socket, keys: Keys) -> None:
    """Answer ZAP requests on a REP socket bound to `ZAP_ENDPOINT` in the workers socket's context."""
    while True:
        await zap.send_multipart(keys.reply(await zap.recv_multipart()))  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
