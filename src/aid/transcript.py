"""Follow a Claude Code session transcript and turn its entries into aid events.

The transcript is `<config dir>/projects/<cwd, / as ->/<session id>.jsonl`,
append-only JSONL. It is Claude Code's internal format, not an API; checked
against 2.1.283. What this module relies on:

- `assistant` entries carry one content block each (`text`, `thinking`,
  `tool_use`), written when the block completes. No token deltas.
- A `user` entry with `toolUseResult` carries the `tool_result` blocks.
- `system` / `turn_duration` ends a turn, once, after any Stop-hook
  continuations. `assistant` with `stop_reason: end_turn` does not: a hook can
  block the stop and the model goes on.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field
from watchfiles import (
    awatch,  # pyright: ignore[reportUnknownVariableType] -- its stop_event type names trio, which is not installed
)

from aid.protocol import TextDelta, ThoughtDelta, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aid.protocol import SessionEvent

log = logging.getLogger(__name__)

INTERRUPTED_MARKER: Final = "[Request interrupted by user"


@dataclass(frozen=True)
class TurnEnded:
    interrupted: bool = False


type TranscriptItem = SessionEvent | TurnEnded


def config_dir(env: dict[str, str] | None = None) -> Path:
    env = dict(os.environ) if env is None else env
    return Path(env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def find_transcript(base: Path, session_id: str) -> Path | None:
    return next((base / "projects").glob(f"*/{session_id}.jsonl"), None)


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class _Block(_Lenient):
    type: str = ""
    text: str = ""
    thinking: str = ""
    id: str = ""
    name: str | None = None
    tool_use_id: str = ""
    is_error: bool = False


class _Message(_Lenient):
    content: str | list[_Block] = ""

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else "".join(b.text for b in self.content)


class _Entry(_Lenient):
    type: str = ""
    subtype: str | None = None
    is_sidechain: bool = Field(False, alias="isSidechain")
    message: _Message = Field(default_factory=_Message)
    tool_use_result: Any = Field(None, alias="toolUseResult")


def items_from_entry(raw: dict[str, Any]) -> list[TranscriptItem]:
    entry = _Entry.model_validate(raw)
    if entry.is_sidechain:
        return []
    blocks = entry.message.content if isinstance(entry.message.content, list) else []
    match entry.type:
        case "assistant":
            items: list[TranscriptItem] = []
            for block in blocks:
                if block.type == "text" and block.text:
                    items.append(TextDelta(text=block.text))
                elif block.type == "thinking" and block.thinking:
                    items.append(ThoughtDelta(text=block.thinking))
                elif block.type == "tool_use":
                    items.append(ToolCall(tool_call_id=block.id, title=block.name, status="in_progress"))
            return items
        case "user" if "tool_use_result" in entry.model_fields_set:
            return [
                ToolCall(tool_call_id=b.tool_use_id, status="failed" if b.is_error else "completed")
                for b in blocks
                if b.type == "tool_result"
            ]
        case "user" if entry.message.text().startswith(INTERRUPTED_MARKER):
            return [TurnEnded(interrupted=True)]
        case "system" if entry.subtype == "turn_duration":
            return [TurnEnded()]
        case _:
            return []


class TranscriptFollower:
    """Reads a transcript from a byte offset on, one complete line at a time."""

    def __init__(self, path: Path, offset: int = 0) -> None:
        self.path = path
        self.offset = offset
        self._partial = b""

    async def size(self) -> int:
        return (await anyio.Path(self.path).stat()).st_size

    async def read_new(self) -> list[dict[str, Any]]:
        async with await anyio.open_file(self.path, "rb") as f:
            await f.seek(self.offset)
            data = await f.read()
        self.offset += len(data)
        *lines, self._partial = (self._partial + data).split(b"\n")
        entries: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping a transcript line that is not JSON: %.120r", line)
        return entries

    async def follow(self) -> AsyncIterator[dict[str, Any]]:
        """Yield entries as they are appended. Runs until cancelled."""
        for entry in await self.read_new():
            yield entry
        # The timeout re-reads even without an event: a line written between the read above and the watch
        # starting raises none, and the turn's last line is often that one.
        async for _changes in awatch(self.path, debounce=20, step=10, rust_timeout=500, yield_on_timeout=True):
            for entry in await self.read_new():
                yield entry
