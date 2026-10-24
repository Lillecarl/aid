from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel, ConfigDict

from aid.protocol import Delivery, MessageEntry, TurnError

if TYPE_CHECKING:
    from aid.history import HistoryLog
    from aid.protocol import HistoryEntry

INBOX_FILE = "inbox.json"


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    through: int = -1
    delivered: tuple[int, ...] = ()
    """Seqs after `through` already dispatched: interrupting messages a turn took past ones waiting for it."""
    turn: str | None = None
    previous: int | None = None
    previous_delivered: tuple[int, ...] = ()


class Inbox:
    """Messages in history after a durable dispatch checkpoint. Uncertain dispatches are never replayed."""

    def __init__(self, path: anyio.Path, history: HistoryLog) -> None:
        self._path = path
        self._history = history
        self._checkpoint = _Checkpoint()
        self._pending: list[HistoryEntry] = []
        self._claimed: list[HistoryEntry] = []
        self._lock = anyio.Lock()

    def __len__(self) -> int:
        return len(self._pending)

    def ready(self, *, interrupts_only: bool = False) -> bool:
        return any(_takes(entry, interrupts_only) for entry in self._pending)

    async def _save(self, checkpoint: _Checkpoint) -> None:
        temporary = self._path.with_suffix(".tmp")
        with anyio.CancelScope(shield=True):
            await temporary.write_bytes(checkpoint.model_dump_json().encode())
            await temporary.replace(self._path)
            self._checkpoint = checkpoint

    async def initialize(self) -> None:
        """Create the checkpoint without resetting messages accepted during session startup."""
        async with self._lock:
            await self._save(self._checkpoint)

    async def restore(self) -> str | None:
        """Restore pending messages; report interrupted dispatch without repeating its possible effects."""
        if not await self._path.exists():
            # Existing sessions have no dispatch ledger. Historical messages may already have effects.
            latest = await self._history.page(limit=1)
            await self._save(_Checkpoint(through=latest.total - 1))
            return None
        self._checkpoint = _Checkpoint.model_validate_json(await self._path.read_bytes())
        after = self._checkpoint.through
        delivered = set(self._checkpoint.delivered)
        while True:
            page = await self._history.page(after=after, limit=1000)
            self._pending.extend(
                entry for entry in page.entries if isinstance(entry.item, MessageEntry) and entry.seq not in delivered
            )
            if not page.has_newer:
                break
            after = page.entries[-1].seq
        if (turn := self._checkpoint.turn) is None:
            return None
        attention = f"Message dispatch {turn} was interrupted; inspect history before repeating its work"
        await self._history.append(TurnError(code="delivery_uncertain", message=attention), turn)
        await self.finish(turn)
        return attention

    async def accept(self, message: MessageEntry, turn: str) -> None:
        """Record a message before accepting it for delivery."""
        async with self._lock:
            with anyio.CancelScope(shield=True):
                entry = await self._history.append(message, turn)
                self._pending.append(entry)

    async def claim(self, turn: str, *, interrupts_only: bool = False) -> list[MessageEntry]:
        """Checkpoint one batch before dispatch. A crash after this point cannot replay the batch.
        `interrupts_only`: a running turn takes only `Delivery.INTERRUPT` messages; the rest stay for its end."""
        async with self._lock:
            taken = [entry for entry in self._pending if _takes(entry, interrupts_only)]
            if not taken:
                return []
            left = [entry for entry in self._pending if not _takes(entry, interrupts_only)]
            old = self._checkpoint
            last = taken[-1].seq
            through = min(last, left[0].seq - 1) if left else max(old.through, last, *old.delivered)
            done = {*old.delivered, *(entry.seq for entry in taken)}
            with anyio.CancelScope(shield=True):
                await self._save(
                    _Checkpoint(
                        through=through,
                        delivered=tuple(sorted(seq for seq in done if seq > through)),
                        turn=turn,
                        previous=old.through,
                        previous_delivered=old.delivered,
                    )
                )
                self._pending = left
                self._claimed = taken
            return [entry.item for entry in taken if isinstance(entry.item, MessageEntry)]

    async def release(self, turn: str) -> None:
        """Return a batch only when the transport proves that dispatch failed."""
        async with self._lock:
            current = self._checkpoint
            if current.turn == turn and current.previous is not None:
                with anyio.CancelScope(shield=True):
                    await self._save(_Checkpoint(through=current.previous, delivered=current.previous_delivered))
                    self._pending = sorted(self._claimed + self._pending, key=lambda entry: entry.seq)
                    self._claimed = []

    async def finish(self, turn: str) -> None:
        """Close only the dispatch belonging to this turn."""
        async with self._lock:
            if self._checkpoint.turn == turn:
                await self._save(_Checkpoint(through=self._checkpoint.through, delivered=self._checkpoint.delivered))
                self._claimed = []


def _takes(entry: HistoryEntry, interrupts_only: bool) -> bool:
    return not interrupts_only or (isinstance(entry.item, MessageEntry) and entry.item.delivery is Delivery.INTERRUPT)
