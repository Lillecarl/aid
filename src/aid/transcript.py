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
- Every `assistant` entry of one API message repeats that message's `id`,
  `model` and whole `usage` (2 to 16 entries per id measured): count usage once
  per id.
- `toolUseResult` for Edit: `filePath`, `oldString`, `newString`; for Write:
  `type` create|update, `filePath`, `content`, `originalFile`; for Read:
  `file.filePath`. Checked against 2.1.283.
- Subagents write `<session>/subagents/agent-<id>.jsonl`, not this file.
- A prompt a person typed, and one aid pasted, is a `user` entry with
  `origin.kind == "human"`. Other turns open on other origins: `channel`,
  `task-notification`. Slash commands and their output carry no origin.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import anyio
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from watchfiles import (
    awatch,  # pyright: ignore[reportUnknownVariableType] -- its stop_event type names trio, which is not installed
)

from aid.protocol import TextDelta, ThoughtDelta, ToolCall, ToolDiff, Usage, clip, to_json

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aid.protocol import SessionEvent

log = structlog.get_logger(__name__)

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


# Enough of a transcript's end to hold its last few entries; one entry can be a whole file's diff.
_TAIL: Final = 256 * 1024


def last_version(path: Path) -> str | None:
    """The Claude Code version that wrote the transcript's latest entry: every entry names it."""
    with path.open("rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - _TAIL))
        tail = f.read()
    for line in reversed(tail.splitlines()):
        try:
            version = json.loads(line).get("version")
        except json.JSONDecodeError, AttributeError:
            continue
        if isinstance(version, str):
            return version
    return None


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class _ResultBlock(_Lenient):
    type: str = ""
    text: str = ""


class _Block(_Lenient):
    type: str = ""
    text: str = ""
    thinking: str = ""
    id: str = ""
    name: str | None = None
    tool_use_id: str = ""
    is_error: bool = False
    input: Any = None
    """A `tool_use` block's arguments."""
    content: str | list[_ResultBlock] | None = None
    """A `tool_result` block's result."""

    def result_text(self) -> str | None:
        if isinstance(self.content, list):
            return "\n".join(b.text for b in self.content if b.type == "text" and b.text) or None
        return self.content


class _OutputDetails(_Lenient):
    thinking_tokens: int = 0


class _Usage(_Lenient):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens_details: _OutputDetails = Field(default_factory=_OutputDetails)


class _Message(_Lenient):
    id: str = ""
    model: str = ""
    usage: _Usage | None = None
    content: str | list[_Block] = ""

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else "".join(b.text for b in self.content)


class _Origin(_Lenient):
    kind: str = ""


class _Entry(_Lenient):
    type: str = ""
    subtype: str | None = None
    is_sidechain: bool = Field(False, alias="isSidechain")
    is_meta: bool = Field(False, alias="isMeta")
    origin: _Origin | None = None
    message: _Message = Field(default_factory=_Message)
    tool_use_result: Any = Field(None, alias="toolUseResult")


def human_prompt(raw: dict[str, Any]) -> str | None:
    """The text of a prompt a person gave: typed, or pasted as aid pastes. None for anything else, such as a
    channel event, a task notification, a slash command's output or a tool result."""
    entry = _Entry.model_validate(raw)
    if entry.type != "user" or entry.is_meta or entry.origin is None or entry.origin.kind != "human":
        return None
    return entry.message.text()


class _ToolResult(_Lenient):
    """The fields of a `toolUseResult` that say which file changed and how."""

    type: str = ""
    file_path: str = Field("", alias="filePath")
    old_string: str | None = Field(None, alias="oldString")
    new_string: str | None = Field(None, alias="newString")
    content: str | None = None
    original_file: str | None = Field(None, alias="originalFile")


# Claude Code's tool names to ACP's tool kinds, which the web UI already reads.
TOOL_KINDS: Final = {
    "Read": "read",
    "Edit": "edit",
    "MultiEdit": "edit",
    "Write": "edit",
    "NotebookEdit": "edit",
    "Bash": "execute",
    "Grep": "search",
    "Glob": "search",
    "WebFetch": "fetch",
    "WebSearch": "fetch",
    "Task": "think",
    "Agent": "think",
}


def tool_paths(input: Any) -> list[str]:
    if not isinstance(input, dict):
        return []
    arguments = cast("dict[str, Any]", input)
    return [p for key in ("file_path", "notebook_path") if isinstance(p := arguments.get(key), str)]


def tool_diffs(result: Any) -> list[ToolDiff]:
    """An Edit's or Write's change. The strings are the edit's own, never the whole file an Edit changed."""
    if not isinstance(result, dict):
        return []
    try:
        found = _ToolResult.model_validate(result)
    except ValidationError:
        return []
    if not found.file_path:
        return []
    if found.old_string is not None and found.new_string is not None:
        return [ToolDiff(path=found.file_path, old=clip(found.old_string), new=clip(found.new_string))]
    if found.type in ("create", "update") and found.content is not None:
        old = None if found.type == "create" or found.original_file is None else clip(found.original_file)
        return [ToolDiff(path=found.file_path, old=old, new=clip(found.content))]
    return []


class TurnUsage:
    """The Usage of one turn's transcript entries, each API message counted once."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._models: dict[str, None] = {}
        self._usage = Usage(
            input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, thought_tokens=0, requests=0
        )

    def add(self, raw: dict[str, Any]) -> None:
        entry = _Entry.model_validate(raw)
        message = entry.message
        if entry.type != "assistant" or message.usage is None or not message.id or message.id in self._seen:
            return
        self._seen.add(message.id)
        if message.model and not message.model.startswith("<"):  # "<synthetic>" marks a message no model wrote
            self._models[message.model] = None
        tokens, total = message.usage, self._usage
        self._usage = total.model_copy(
            update={
                "input_tokens": (total.input_tokens or 0) + tokens.input_tokens,
                "output_tokens": (total.output_tokens or 0) + tokens.output_tokens,
                "cache_read_tokens": (total.cache_read_tokens or 0) + tokens.cache_read_input_tokens,
                "cache_write_tokens": (total.cache_write_tokens or 0) + tokens.cache_creation_input_tokens,
                "thought_tokens": (total.thought_tokens or 0) + tokens.output_tokens_details.thinking_tokens,
                "requests": (total.requests or 0) + 1,
            }
        )

    def usage(self) -> Usage:
        return self._usage.model_copy(update={"models": list(self._models)})


def usage_since(path: Path, offset: int) -> tuple[Usage, int]:
    """The Usage of a transcript's complete lines from byte `offset` on, and the offset after them. A subagent's
    transcript (`<session>/subagents/agent-<id>.jsonl`) is shaped like the session's."""
    usage = TurnUsage()
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    complete = data[: data.rfind(b"\n") + 1]
    for line in complete.splitlines():
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if isinstance(raw, dict):
            usage.add(cast("dict[str, Any]", raw))
    return usage.usage(), offset + len(complete)


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
                    items.append(
                        ToolCall(
                            tool_call_id=block.id,
                            title=block.name,
                            kind=TOOL_KINDS.get(block.name or "", "other"),
                            status="in_progress",
                            input=to_json(block.input),
                            paths=tool_paths(block.input),
                        )
                    )
            return items
        case "user" if "tool_use_result" in entry.model_fields_set:
            results = [b for b in blocks if b.type == "tool_result"]
            # One toolUseResult per entry: it belongs to a block only when the entry has one.
            diffs = tool_diffs(entry.tool_use_result) if len(results) == 1 else []
            return [
                ToolCall(
                    tool_call_id=b.tool_use_id,
                    status="failed" if b.is_error else "completed",
                    output=clip(text) if (text := b.result_text()) else None,
                    diffs=diffs,
                )
                for b in results
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
                log.warning("transcript_line_skipped", line=line[:120])
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
