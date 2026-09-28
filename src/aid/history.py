"""Per-session history: what was asked and answered, whichever client asked.

The daemon records it, so it covers every backend and every client alike. It lives in `history.jsonl` in the
session's state directory: one `HistoryEntry` per line, numbered from 0. Text arrives in many small deltas
(an ACP agent sends one per token), so `Recorder` joins consecutive deltas into one entry before writing.

Reading a page seeks straight to it through an index of line offsets, built once per log by one scan of the
file. A session can grow long; a page read stays the size of the page.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Final

import anyio
from pydantic import ValidationError

from aid.protocol import HistoryEntry, HistoryPage, PromptEntry, TextDelta, ThoughtDelta, TurnError

if TYPE_CHECKING:
    from aid.protocol import HistoryItem, SessionEvent

log = logging.getLogger(__name__)

HISTORY_FILE: Final = "history.jsonl"
MAX_PAGE: Final = 1000


class HistoryLog:
    def __init__(self, path: anyio.Path) -> None:
        self._path = path
        self._offsets: list[int] | None = None
        """Byte offset of each entry's line, then of the end of the last complete line."""
        self._lock = anyio.Lock()

    async def _index(self) -> list[int]:
        if self._offsets is not None:
            return self._offsets
        offsets = [0]
        if await self._path.exists():
            data = await self._path.read_bytes()
            position = 0
            while (end := data.find(b"\n", position)) != -1:
                position = end + 1
                offsets.append(position)
            if position != len(data):
                # A line cut short by a crash. Appending after it would glue the next entry onto it.
                log.warning("%s: dropping %d bytes of an unfinished line", self._path, len(data) - position)
                async with await anyio.open_file(self._path, "r+b") as f:
                    await f.truncate(position)
        self._offsets = offsets
        return offsets

    async def append(self, item: HistoryItem, turn: str) -> HistoryEntry:
        async with self._lock:
            offsets = await self._index()
            entry = HistoryEntry(seq=len(offsets) - 1, at=time.time(), turn=turn, item=item)
            line = entry.model_dump_json().encode() + b"\n"
            await self._path.parent.mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(self._path, "ab") as f:
                await f.write(line)
            offsets.append(offsets[-1] + len(line))
            return entry

    async def page(self, *, before: int | None = None, after: int | None = None, limit: int = 100) -> HistoryPage:
        """Up to `limit` entries: the newest before `before`, the oldest after `after`, or the newest of all."""
        if before is not None and after is not None:
            raise ValueError("give before or after, not both")
        limit = max(1, min(limit, MAX_PAGE))
        async with self._lock:
            offsets = await self._index()
            total = len(offsets) - 1
            if after is not None:
                start = max(0, after + 1)
                end = min(total, start + limit)
            else:
                end = total if before is None else max(0, min(before, total))
                start = max(0, end - limit)
            entries: list[HistoryEntry] = []
            if start < end:
                async with await anyio.open_file(self._path, "rb") as f:
                    await f.seek(offsets[start])
                    data = await f.read(offsets[end] - offsets[start])
                for line in data.splitlines():
                    try:
                        entries.append(HistoryEntry.model_validate_json(line))
                    except ValidationError:
                        log.exception("%s: skipping an unreadable entry", self._path)
        return HistoryPage(entries=entries, has_older=start > 0, has_newer=end < total, total=total)


class Recorder:
    """Turns one prompt's events into history entries, joining consecutive text or thought deltas."""

    def __init__(self, history: HistoryLog, turn: str) -> None:
        self._history = history
        self._turn = turn
        self._pending: TextDelta | ThoughtDelta | None = None

    async def flush(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            await self._history.append(pending, self._turn)

    async def event(self, event: SessionEvent) -> None:
        if isinstance(event, TextDelta | ThoughtDelta):
            if type(self._pending) is type(event) and self._pending is not None:
                self._pending = type(event)(text=self._pending.text + event.text)
                return
            await self.flush()
            self._pending = event
            return
        await self.flush()
        await self._history.append(event, self._turn)

    async def prompt(self, text: str) -> None:
        await self._history.append(PromptEntry(text=text), self._turn)

    async def error(self, code: str, message: str) -> None:
        await self.flush()
        await self._history.append(TurnError(code=code, message=message), self._turn)
