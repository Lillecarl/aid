from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.protocol import (
    Cost,
    FileActivity,
    HistoryEntry,
    HistoryItem,
    Output,
    PromptEntry,
    Started,
    ToolCall,
    ToolDiff,
    Usage,
)
from aid.summary import summarize
from tests.conftest import acp_spec
from tests.fake_acp_agent import RESOLVED_MODEL

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

TIMEOUT = 30


def history(*items: HistoryItem) -> list[HistoryEntry]:
    return [HistoryEntry(seq=i, at=float(i), turn="t", item=item) for i, item in enumerate(items)]


def turn_usage(cost: float, model: str = "m1") -> Usage:
    usd = Cost(amount=cost, currency="USD")
    return Usage(input_tokens=10, output_tokens=2, cache_read_tokens=100, requests=1, models=[model], session_cost=usd)


def test_summary_totals_a_history() -> None:
    summary = summarize(
        history(
            Started(pid=1, agent_session="s1"),
            PromptEntry(text="a"),
            ToolCall(tool_call_id="read", kind="read", status="in_progress", paths=["/a.py:12"]),
            ToolCall(tool_call_id="read", status="completed"),
            ToolCall(tool_call_id="edit", kind="edit", status="in_progress", paths=["/b.py"]),
            ToolCall(tool_call_id="edit", status="completed", diffs=[ToolDiff(path="/b.py", old="x", new="y")]),
            turn_usage(0.25),
            Output(output="", stop_reason="end_turn"),
            turn_usage(0.75),
            Output(output="", stop_reason="end_turn"),
            # The agent restarts its session, and its running cost with it.
            Started(pid=2, agent_session="s1", resumed=True),
            ToolCall(tool_call_id="shell", kind="execute", status="completed", paths=["/c.py"]),
            ToolCall(tool_call_id="read2", kind="read", paths=["/b.py"]),
            turn_usage(0.5, model="m2"),
            Output(output="", stop_reason="end_turn"),
        )
    )
    assert (summary.turns, summary.starts, summary.requests) == (3, 2, 3)
    assert (summary.input_tokens, summary.output_tokens, summary.cache_read_tokens) == (30, 6, 300)
    assert summary.models == {"m1": 2, "m2": 1}
    assert summary.cost == {"USD": 1.25}
    assert summary.agent_sessions == ["s1"]
    assert summary.files == [
        FileActivity(path="/b.py", reads=1, writes=1, last_seq=12),
        FileActivity(path="/a.py", reads=1, last_seq=3),
    ]


def test_summary_of_nothing() -> None:
    assert summarize([]).turns == 0


@pytest.mark.anyio
async def test_daemon_summary(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("sum", acp_spec(tmp_path))
            await session.run("usage")
            await session.run("usage")
            summary = await session.summary()
    assert (summary.turns, summary.starts) == (2, 1)
    assert (summary.input_tokens, summary.output_tokens) == (20, 12)
    assert summary.models == {RESOLVED_MODEL: 2}
    # One agent session: its running cost counts once.
    assert summary.cost == {"USD": 0.5}
