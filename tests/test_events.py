from __future__ import annotations

import contextlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import pytest
import zmq
import zmq.asyncio

import aid
from aid import events
from aid.paths import Paths
from tests.conftest import py_spec

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from aid.protocol import HistoryEntry

pytestmark = pytest.mark.anyio

TIMEOUT = 30
# A SUB socket misses what is published before its subscription reaches the PUB: the slow joiner.
JOIN = 0.3


class Subscriber:
    def __init__(self, sock: zmq.asyncio.Socket) -> None:
        self._sock = sock

    async def next(self, wait: float = 5) -> tuple[bytes, Any]:
        with anyio.fail_after(wait):
            topic, payload = await self._sock.recv_multipart()
        return topic, json.loads(payload)

    async def quiet(self, wait: float = 0.5) -> bool:
        with anyio.move_on_after(wait):
            await self._sock.recv_multipart()
            return False
        return True

    async def until(self, topic: bytes, match: Callable[[Any], object]) -> Any:
        while True:
            got, payload = await self.next()
            if got == topic and match(payload):
                return payload


@asynccontextmanager
async def subscribed(paths: Paths, *prefixes: bytes) -> AsyncGenerator[Subscriber]:
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(paths.events)
    for prefix in prefixes:
        sock.setsockopt(zmq.SUBSCRIBE, prefix)
    await anyio.sleep(JOIN)
    try:
        yield Subscriber(sock)
    finally:
        sock.close(linger=0)
        ctx.term()


async def test_the_daemon_publishes_changes(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with (
            subscribed(daemon, events.SESSIONS, events.status_topic("echo")) as sub,
            aid.connect(daemon) as client,
        ):
            session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await sub.until(events.SESSIONS, lambda infos: infos and infos[0]["running"])
            await session.run("hi")
            idle = await sub.until(events.status_topic("echo"), lambda s: not s["working"] and s["model"])
            while not await sub.quiet():  # What the turn's end still sends.
                pass
            await session.status()
            await session.status()
            nothing_new = await sub.quiet()
            await session.stop()
            stopped = await sub.until(events.SESSIONS, lambda infos: not infos[0]["running"])
    assert listed == [
        {"name": "echo", "kind": "pydantic-ai", "running": True, "permissions": 0, "working": False, "attention": None}
    ]
    assert idle["name"] == "echo"
    assert nothing_new
    assert stopped[0]["running"] is False


async def test_every_history_entry_is_published_with_its_seq(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with subscribed(daemon, events.history_topic("echo")) as sub, aid.connect(daemon) as client:
            session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
            await session.run("one")
            recorded = (await session.history()).entries
            published = [(await sub.next())[1] for _ in recorded]
            extra = await sub.quiet()
    assert [p["seq"] for p in published] == [e.seq for e in recorded] == list(range(len(recorded)))
    assert [p["item"]["type"] for p in published] == [e.item.type for e in recorded]
    assert extra


async def test_following_gives_the_baseline_then_each_change(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client, contextlib.aclosing(client.follow_sessions()) as lists:
            first = await anext(lists)
            await client.create("echo", py_spec(tmp_path, "agents:echo"))
            listed = await anext(lists)
            session = client.session("echo")
            async with contextlib.aclosing(session.follow_status()) as statuses:
                now = await anext(statuses)
                await session.stop()
                while (status := await anext(statuses)).running:
                    pass
    assert first == []
    assert [info.name for info in listed] == ["echo"]
    assert now.running
    assert not status.running


@dataclass(frozen=True)
class _OwnEvents(Paths):
    """The daemon's control socket, and a PUB socket the test publishes on itself."""

    fake: str = ""

    @property
    def events(self) -> str:
        return self.fake


async def test_a_dropped_entry_is_fetched(daemon: Paths, tmp_path: Path) -> None:
    # A PUB drops what a slow reader cannot take; here the test publishes around a gap on purpose.
    paths = _OwnEvents(daemon.runtime_dir, daemon.state_dir, fake=f"ipc://{tmp_path / 'fake-events.sock'}")
    ctx = zmq.asyncio.Context()
    publisher = ctx.socket(zmq.PUB)
    publisher.bind(paths.events)
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(paths) as client:
                session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
                await session.run("before")
                start = (await session.history()).entries[-1].seq
                got: list[HistoryEntry] = []
                recorded: list[HistoryEntry] = []
                until: list[int] = []
                done = anyio.Event()
                async with contextlib.aclosing(session.follow_history(after=start)) as entries:

                    async def take() -> None:
                        async for entry in entries:
                            got.append(entry)
                            if until and entry.seq >= until[0]:
                                done.set()
                                return

                    async with anyio.create_task_group() as tg:
                        tg.start_soon(take)
                        await anyio.sleep(JOIN)  # Its baseline is empty; it waits on the PUB, which is ours.
                        await session.run("after")  # The daemon publishes this turn on its own PUB, unseen.
                        recorded += (await session.history(after=start)).entries
                        until.append(recorded[-1].seq)
                        # Only the newest: every entry before it is a gap the follower must fetch.
                        topic = events.history_topic("echo")
                        await publisher.send_multipart([topic, recorded[-1].model_dump_json().encode()])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
                        await done.wait()
    finally:
        publisher.close(linger=0)
        ctx.term()
    assert got == recorded
