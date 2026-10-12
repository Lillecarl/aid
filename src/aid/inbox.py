from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel, ConfigDict

from aid.protocol import MessageEntry, TurnError

if TYPE_CHECKING:
    from aid.history import HistoryLog
    from aid.protocol import HistoryEntry

INBOX_FILE = "inbox.json"


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    through: int = -1
    turn: str | None = None
    previous: int | None = None


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
        while True:
            page = await self._history.page(after=after, limit=1000)
            self._pending.extend(entry for entry in page.entries if isinstance(entry.item, MessageEntry))
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

    async def claim(self, turn: str) -> list[MessageEntry]:
        """Checkpoint one batch before dispatch. A crash after this point cannot replay the batch."""
        async with self._lock:
            if not self._pending:
                return []
            taken = self._pending
            with anyio.CancelScope(shield=True):
                await self._save(
                    _Checkpoint(through=self._pending[-1].seq, turn=turn, previous=self._checkpoint.through)
                )
                self._pending = []
                self._claimed = taken
            return [entry.item for entry in taken if isinstance(entry.item, MessageEntry)]

    async def release(self, turn: str) -> None:
        """Return a batch only when the transport proves that dispatch failed."""
        async with self._lock:
            if self._checkpoint.turn == turn and self._checkpoint.previous is not None:
                with anyio.CancelScope(shield=True):
                    await self._save(_Checkpoint(through=self._checkpoint.previous))
                    self._pending = self._claimed + self._pending
                    self._claimed = []

    async def finish(self, turn: str) -> None:
        """Close only the dispatch belonging to this turn."""
        async with self._lock:
            if self._checkpoint.turn == turn:
                await self._save(_Checkpoint(through=self._checkpoint.through))
                self._claimed = []
