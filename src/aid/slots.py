"""Model slots: at most N requests in flight to one model host, across every session of the daemon.

A self-hosted server serves best at a steady load: more requests than its batch queue, less throughput for each. The
operator names hosts and counts (`aid daemon --model-slots HOST=N`). A native worker whose model sends to such a host
knows them from its launch, and takes a slot around each request (`Slotted`), first come, first served, giving it
back when the request ends, however it ends. The daemon owns the table (`ModelSlots`): a worker that exits frees
every slot its session held or waited for.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Final

import anyio
from pydantic import TypeAdapter, ValidationError

from aid.client import connect
from aid.control import ENV_MODEL_SLOTS
from aid.protocol import AcquireModelSlot, AidError, ReleaseModelSlot, SlotUse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from aid.messages import ModelMessage, ModelResponse
    from aid.model import Model, ModelInfo, StreamDelta


RELEASED_KEPT: Final = 1024
_CAPACITIES: Final = TypeAdapter(dict[str, int])


class _Host:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.holders: dict[str, str] = {}
        """Token → session."""
        self.waiting: OrderedDict[str, tuple[str, anyio.Event]] = OrderedDict()


class ModelSlots:
    """The daemon's table: hosts with their capacity, who holds a slot and who waits, in order."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        self._hosts = {host: _Host(n) for host, n in capacities.items()}
        self._tokens: dict[str, _Host] = {}
        self._released: OrderedDict[str, None] = OrderedDict()
        """Tokens released before the daemon saw them asked for: acquire and release travel on separate connections,
        so a cancelled wait's release can overtake its acquire. A late acquire for one is refused, not granted to a
        holder nobody would release before its worker exits."""

    def use(self) -> list[SlotUse]:
        return [
            SlotUse(
                host=host,
                capacity=slots.capacity,
                holders=list(slots.holders.values()),
                waiting=[session for session, _ in slots.waiting.values()],
            )
            for host, slots in self._hosts.items()
        ]

    def capacities(self) -> dict[str, int]:
        return {host: slots.capacity for host, slots in self._hosts.items()}

    async def acquire(self, host: str, token: str, session: str) -> None:
        """Return holding a slot of `host` for `token`; `AidError` if the token was released while it waited."""
        if (slots := self._hosts.get(host)) is None:
            raise AidError("unlimited", f"model host {host!r} has no slots")
        if token in self._tokens:
            raise AidError("duplicate_id", f"a model slot {token!r} is taken or wanted already")
        if self._released.pop(token, False) is None:
            raise AidError("slot_released", f"model slot {token!r} was given up before it was asked for")
        self._tokens[token] = slots
        if len(slots.holders) < slots.capacity and not slots.waiting:
            slots.holders[token] = session
            return
        granted = anyio.Event()
        slots.waiting[token] = (session, granted)
        try:
            await granted.wait()
        except BaseException:
            self.release(token)
            raise
        if token not in slots.holders:
            raise AidError("slot_released", f"model slot {token!r} was given up while it waited")

    def release(self, token: str) -> None:
        """Give back a held slot to the next in line, or stop waiting; an unknown token is already gone."""
        slots = self._tokens.pop(token, None)
        if slots is None:
            self._released[token] = None
            while len(self._released) > RELEASED_KEPT:
                self._released.popitem(last=False)
            return
        if slots.holders.pop(token, None) is None:
            waited = slots.waiting.pop(token, None)
            if waited is not None:
                waited[1].set()
            return
        while slots.waiting and len(slots.holders) < slots.capacity:
            next_token, (session, granted) = slots.waiting.popitem(last=False)
            slots.holders[next_token] = session
            granted.set()

    def release_session(self, session: str) -> None:
        """Everything `session` holds or waits for: its worker is gone."""
        for slots in self._hosts.values():
            # Waits first: a held slot released before them would go to the session's own next waiter.
            for token in [t for t, (s, _) in slots.waiting.items() if s == session]:
                self.release(token)
            for token in [t for t, s in slots.holders.items() if s == session]:
                self.release(token)


def limited_hosts() -> frozenset[str]:
    """The hosts the daemon limits, as it launched this worker; none outside a worker."""
    try:
        return frozenset(_CAPACITIES.validate_json(os.environ.get(ENV_MODEL_SLOTS) or "{}"))
    except ValidationError:
        return frozenset()


class Slotted:
    """A model whose every request holds one of the daemon's slots for `host`, for as long as the request runs."""

    def __init__(self, model: Model, host: str) -> None:
        self.model_name = model.model_name
        self.cache_reads_in_prompt: bool = getattr(model, "cache_reads_in_prompt", False)
        self._model = model
        self._host = host

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        async with self._slot():
            return await self._model.request(messages, info)

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        async with self._slot():
            async for delta in self._model.stream(messages, info):
                yield delta

    async def aclose(self) -> None:
        if (close := getattr(self._model, "aclose", None)) is not None:
            await close()

    @contextlib.asynccontextmanager
    async def _slot(self) -> AsyncGenerator[None]:
        token = uuid.uuid4().hex
        try:
            async with connect() as client:
                await client.call(AcquireModelSlot(host=self._host, token=token))
            yield
        finally:
            # Also when the wait was cancelled: the daemon may grant the token after this side stopped listening.
            with anyio.CancelScope(shield=True), contextlib.suppress(AidError, OSError, TimeoutError):
                async with connect() as client:
                    await client.call(ReleaseModelSlot(token=token))
