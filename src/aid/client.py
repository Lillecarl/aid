"""Async Python API for the aid daemon.

async with aid.connect() as client:
    session = await client.create("reviewer", PydanticAISpec(cwd=".", target="agents:reviewer"))
    result = await session.run("Review this diff", output_type=Review)
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import anyio
import zmq
import zmq.asyncio
from pydantic import JsonValue, TypeAdapter

from aid.paths import default_paths
from aid.protocol import (
    AgentCatalog,
    AidError,
    Cancel,
    CreateSession,
    DeleteSession,
    Done,
    Event,
    Failure,
    GetHistory,
    HistoryPage,
    ListAgents,
    ListSessions,
    MessageEntry,
    Output,
    Prompt,
    ReceiveMessages,
    Reply,
    SendMessage,
    SessionInfo,
    StopSession,
    TextDelta,
    decode_reply,
    encode,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from aid.paths import Paths
    from aid.protocol import Request, SessionEvent
    from aid.spec import AgentSpec

_SESSION_INFOS = TypeAdapter(list[SessionInfo])
_MESSAGES = TypeAdapter(list[MessageEntry])


@dataclass(frozen=True)
class RunResult[T]:
    output: T
    text: str
    stop_reason: str


@asynccontextmanager
async def connect(paths: Paths | None = None) -> AsyncGenerator[Client]:
    paths = paths or default_paths()
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(paths.control)
    client = Client(sock)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.read_replies)
            try:
                yield client
            finally:
                tg.cancel_scope.cancel()
    except BaseExceptionGroup as group:
        # The reader's task group wraps the caller's own exception; hand it back as the caller raised it.
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        raise
    finally:
        sock.close(linger=0)
        ctx.term()


class Client:
    def __init__(self, sock: zmq.asyncio.Socket) -> None:
        self._sock = sock
        self._send_lock = anyio.Lock()
        self._pending: dict[str, MemoryObjectSendStream[Reply]] = {}

    async def read_replies(self) -> None:
        while True:
            reply = decode_reply(await self._sock.recv())
            if isinstance(reply, Event | Done | Failure) and (stream := self._pending.get(reply.id)) is not None:
                stream.send_nowait(reply)

    @asynccontextmanager
    async def _exchange(self, request: Request) -> AsyncGenerator[MemoryObjectReceiveStream[Reply]]:
        send, receive = anyio.create_memory_object_stream[Reply](math.inf)
        self._pending[request.id] = send
        try:
            async with self._send_lock:
                await self._sock.send(encode(request))
            yield receive
        finally:
            del self._pending[request.id]
            send.close()
            receive.close()

    async def _events(self, request: Request) -> AsyncIterator[SessionEvent | Done]:
        async with self._exchange(request) as replies:
            async for reply in replies:
                match reply:
                    case Event():
                        yield reply.event
                    case Done():
                        yield reply
                        return
                    case Failure():
                        raise AidError(reply.code, reply.message)
                    case _:
                        pass

    async def call(self, request: Request) -> JsonValue:
        async for item in self._events(request):
            if isinstance(item, Done):
                return item.data
        raise AidError("closed", "daemon closed the exchange without a reply")

    async def create(self, name: str, spec: AgentSpec) -> Session:
        await self.call(CreateSession(name=name, spec=spec))
        return Session(self, name)

    async def sessions(self) -> list[SessionInfo]:
        return _SESSION_INFOS.validate_python(await self.call(ListSessions()))

    async def agents(self) -> AgentCatalog:
        """The `aid.PydanticAgent`s on the daemon's agents path, and what failed to load."""
        return AgentCatalog.model_validate(await self.call(ListAgents()))

    def session(self, name: str) -> Session:
        return Session(self, name)

    async def send_message(self, to: str, text: str, *, sender: str | None = None) -> None:
        """Leave a message for session `to`; the daemon wakes it. `sender` names the session it is from, or None
        for a person. Returns once the message is recorded, not answered."""
        await self.call(SendMessage(to=to, text=text, sender=sender))

    async def receive_messages(self, session: str, *, wait: float = 25) -> list[MessageEntry]:
        """Take the messages waiting for interactive Claude session `session`, waiting up to `wait` seconds."""
        return _MESSAGES.validate_python(await self.call(ReceiveMessages(session=session, wait=wait)))


class Session:
    def __init__(self, client: Client, name: str) -> None:
        self._client = client
        self.name = name

    async def stream(self, text: str) -> AsyncIterator[SessionEvent]:
        """Yield the prompt's events. The last one is an `Output`."""
        async for item in self._client._events(Prompt(session=self.name, text=text)):  # pyright: ignore[reportPrivateUsage] -- Session is Client's own facade
            if not isinstance(item, Done):
                yield item

    @overload
    async def run(self, text: str) -> RunResult[JsonValue]: ...
    @overload
    async def run[T](self, text: str, output_type: type[T]) -> RunResult[T]: ...
    async def run[T](self, text: str, output_type: type[T] | None = None) -> RunResult[T] | RunResult[JsonValue]:
        chunks: list[str] = []
        output: Output | None = None
        async for event in self.stream(text):
            match event:
                case TextDelta():
                    chunks.append(event.text)
                case Output():
                    output = event
                case _:
                    pass
        if output is None:
            raise AidError("no_output", f"prompt to {self.name!r} ended without output")
        if output_type is None:
            return RunResult(output.output, "".join(chunks), output.stop_reason)
        typed = TypeAdapter(output_type).validate_python(output.output)
        return RunResult(typed, "".join(chunks), output.stop_reason)

    async def history(self, *, before: int | None = None, after: int | None = None, limit: int = 100) -> HistoryPage:
        """A page of this session's history: the newest entries before `before`, the oldest after `after`, or
        the newest of all. Entry `seq` numbers are what `before` and `after` take."""
        request = GetHistory(session=self.name, before=before, after=after, limit=limit)
        return HistoryPage.model_validate(await self._client.call(request))

    async def cancel(self) -> None:
        await self._client.call(Cancel(session=self.name))

    async def stop(self) -> None:
        await self._client.call(StopSession(session=self.name))

    async def delete(self) -> None:
        await self._client.call(DeleteSession(session=self.name))
