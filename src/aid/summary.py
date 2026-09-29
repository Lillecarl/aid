"""A session's history, totalled: tokens, models, agent sessions, cost, and the files its tool calls touched."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from aid.protocol import FileActivity, Output, SessionSummary, Started, ToolCall, Usage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aid.protocol import HistoryEntry

WRITE_KINDS: Final = frozenset({"edit", "delete", "move"})
_LINE: Final = re.compile(r":\d+$")
_TOKENS: Final = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "thought_tokens")


@dataclass
class _Call:
    kind: str | None = None
    paths: set[str] = field(default_factory=set[str])
    written: set[str] = field(default_factory=set[str])
    """Paths with a diff."""
    seq: int = 0


def file_of(path: str) -> str:
    """A tool call's `path:line` as the path."""
    return _LINE.sub("", path)


def summarize(entries: Iterable[HistoryEntry]) -> SessionSummary:
    turns = starts = requests = 0
    tokens: Counter[str] = Counter()
    models: Counter[str] = Counter()
    cost: Counter[str] = Counter()
    agent_sessions: dict[str, None] = {}
    calls: dict[str, _Call] = {}
    # The running agent session's latest cost per currency, added to `cost` when the next session starts.
    latest: dict[str, float] = {}

    for entry in entries:
        match entry.item:
            case Started() as started:
                cost.update(latest)
                latest.clear()
                starts += 1
                if started.agent_session:
                    agent_sessions[started.agent_session] = None
            case Output():
                turns += 1
            case Usage() as usage:
                tokens.update({name: getattr(usage, name) or 0 for name in _TOKENS})
                requests += usage.requests or 0
                models.update(usage.models)
                if usage.session_cost is not None:
                    latest[usage.session_cost.currency] = usage.session_cost.amount
            case ToolCall() as tool:
                call = calls.setdefault(tool.tool_call_id, _Call())
                call.kind = call.kind or tool.kind
                call.paths.update(file_of(p) for p in tool.paths)
                call.written.update(d.path for d in tool.diffs)
                call.seq = entry.seq
            case _:
                pass
    cost.update(latest)
    return SessionSummary(
        turns=turns,
        starts=starts,
        requests=requests,
        **{name: tokens[name] for name in _TOKENS},
        models=dict(models),
        cost=dict(cost),
        agent_sessions=list(agent_sessions),
        files=files_of(calls.values()),
    )


def files_of(calls: Iterable[_Call]) -> list[FileActivity]:
    reads: Counter[str] = Counter()
    writes: Counter[str] = Counter()
    last: dict[str, int] = {}
    for call in calls:
        written = call.written | (call.paths if call.kind in WRITE_KINDS else set[str]())
        read = call.paths - written if call.kind == "read" else set[str]()
        writes.update(written)
        reads.update(read)
        for path in written | read:
            last[path] = max(last.get(path, 0), call.seq)
    found = [FileActivity(path=p, reads=reads[p], writes=writes[p], last_seq=seq) for p, seq in last.items()]
    return sorted(found, key=lambda a: a.last_seq, reverse=True)
