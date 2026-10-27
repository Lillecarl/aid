from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import Daemon
from aid.protocol import AidError, ListModelSlots, SlotUse
from aid.slots import ModelSlots
from tests.conftest import py_spec
from tests.models import HOST

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


async def test_slots_go_first_come_first_served_and_come_back_however_a_holder_ends() -> None:
    slots = ModelSlots({"gpu": 1})
    assert slots.capacities() == {"gpu": 1}
    order: list[str] = []

    async def take(token: str, session: str) -> None:
        try:
            await slots.acquire("gpu", token, session)
            order.append(token)
        except AidError as error:
            order.append(f"{token}: {error.code}")

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await slots.acquire("gpu", "a", "one")
            for token, session in (("b", "two"), ("c", "three"), ("d", "two")):
                tg.start_soon(take, token, session)
                await anyio.sleep(0.01)
            slots.release("c")  # Gives up waiting.
            slots.release("a")  # Next in line: b.
            await anyio.sleep(0.01)
            slots.release_session("two")  # b held, d waited: both go, and nobody is left.
        with pytest.raises(AidError, match="slot_released"):
            slots.release("late")
            await slots.acquire("gpu", "late", "one")
        await slots.acquire("gpu", "e", "one")
    assert order == ["c: slot_released", "b", "d: slot_released"]


@pytest.fixture
async def daemon(paths: Paths, launcher: ForkserverLauncher) -> AsyncIterator[Paths]:
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[], model_slots={HOST: 1}).serve)
        yield paths
        tg.cancel_scope.cancel()


async def test_one_slot_holds_the_second_session_until_the_first_request_ends(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            first = await client.create("first", py_spec(tmp_path, "agents:echo"))
            second = await client.create("second", py_spec(tmp_path, "agents:echo"))
            async with anyio.create_task_group() as tg:
                tg.start_soon(first.run, "slow")  # Streams, then waits forever, holding the host's one slot.
                async with contextlib.aclosing(first.follow_status()) as statuses:
                    async for status in statuses:
                        if status.busy:
                            break
                await anyio.sleep(0.5)
                waited = anyio.Event()

                async def ask() -> None:
                    await second.run("hi")
                    waited.set()

                tg.start_soon(ask)
                with anyio.move_on_after(1):
                    await waited.wait()
                assert not waited.is_set()
                listed = await client.call(ListModelSlots())
                assert isinstance(listed, list)
                assert [SlotUse.model_validate(use) for use in listed] == [
                    SlotUse(host=HOST, capacity=1, holders=["first"], waiting=["second"])
                ]
                await first.cancel()  # The cancelled request gives its slot back.
                await waited.wait()
            assert (await second.history(limit=5)).entries[-1].item.type == "output"
