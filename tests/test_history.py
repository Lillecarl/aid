from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.history import HistoryLog, Recorder
from aid.protocol import Output, PromptEntry, Started, TextDelta, ThoughtDelta, ToolCall, TurnError, Usage
from tests.conftest import acp_spec, py_spec
from tests.fake_acp_agent import AGENT_NAME, AGENT_VERSION, CHUNKS, MODEL

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


async def filled(path: Path, count: int) -> HistoryLog:
    log = HistoryLog(anyio.Path(path))
    for i in range(count):
        await log.append(PromptEntry(text=f"p{i}"), turn="t")
    return log


async def test_pages(tmp_path: Path) -> None:
    log = await filled(tmp_path / "h.jsonl", 10)

    newest = await log.page(limit=3)
    assert [e.seq for e in newest.entries] == [7, 8, 9]
    assert (newest.has_older, newest.has_newer, newest.total) == (True, False, 10)

    older = await log.page(before=7, limit=3)
    assert [e.seq for e in older.entries] == [4, 5, 6]
    assert (older.has_older, older.has_newer) == (True, True)

    first = await log.page(before=2, limit=5)
    assert [e.seq for e in first.entries] == [0, 1]
    assert first.has_older is False

    newer = await log.page(after=4, limit=3)
    assert [e.seq for e in newer.entries] == [5, 6, 7]
    assert newer.has_newer is True

    everything_after_start = await log.page(after=-1, limit=100)
    assert [e.seq for e in everything_after_start.entries] == list(range(10))


async def test_empty_log(tmp_path: Path) -> None:
    page = await HistoryLog(anyio.Path(tmp_path / "none.jsonl")).page()
    assert (page.entries, page.has_older, page.has_newer, page.total) == ([], False, False, 0)


async def test_reopened_log_continues_numbering(tmp_path: Path) -> None:
    await filled(tmp_path / "h.jsonl", 3)
    log = HistoryLog(anyio.Path(tmp_path / "h.jsonl"))
    entry = await log.append(PromptEntry(text="again"), turn="t")
    assert entry.seq == 3
    assert [e.item for e in (await log.page(after=2)).entries] == [PromptEntry(text="again")]


async def test_unfinished_line_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    await filled(path, 2)
    with path.open("ab") as f:
        f.write(b'{"seq": 2, "at": 0, "tu')
    log = HistoryLog(anyio.Path(path))
    entry = await log.append(PromptEntry(text="after crash"), turn="t")
    page = await log.page()
    assert entry.seq == 2
    assert [e.seq for e in page.entries] == [0, 1, 2]


async def test_recorder_joins_deltas(tmp_path: Path) -> None:
    log = HistoryLog(anyio.Path(tmp_path / "h.jsonl"))
    recorder = Recorder(log, "turn-1")
    await recorder.prompt("q")
    for event in (
        ThoughtDelta(text="hm"),
        ThoughtDelta(text="m"),
        TextDelta(text="a"),
        TextDelta(text="b"),
        ToolCall(tool_call_id="t1", status="completed"),
        TextDelta(text="c"),
    ):
        await recorder.event(event)
    await recorder.event(Output(output="abc", stop_reason="end_turn"))
    await recorder.error("x", "later failure")
    items = [e.item for e in (await log.page()).entries]
    assert items == [
        PromptEntry(text="q"),
        ThoughtDelta(text="hmm"),
        TextDelta(text="ab"),
        ToolCall(tool_call_id="t1", status="completed"),
        TextDelta(text="c"),
        Output(output="abc", stop_reason="end_turn"),
        TurnError(code="x", message="later failure"),
    ]
    assert {e.turn for e in (await log.page()).entries} == {"turn-1"}


async def test_daemon_records_every_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            acp = await client.create("acp", acp_spec(tmp_path))
            await acp.run("count")
            acp_history = await acp.history()

            echo = await client.create("echo", py_spec(tmp_path, "agents:echo"))
            for prompt in ("a", "b", "c"):
                await echo.run(prompt)
            latest = await echo.history(limit=4)
            earlier = await echo.history(before=latest.entries[0].seq, limit=100)
            with pytest.raises(aid.AidError):
                await client.session("missing").history()

    counted = "".join(f"{i} " for i in range(CHUNKS))
    start, *turn = [e.item for e in acp_history.entries]
    assert isinstance(start, Started)
    assert (start.agent, start.model, start.resumed) == (f"{AGENT_NAME} {AGENT_VERSION}", MODEL, False)
    assert turn == [
        PromptEntry(text="count"),
        TextDelta(text=counted),
        Output(output=counted, stop_reason="end_turn"),
    ]
    assert acp_history.entries[0].turn != acp_history.entries[1].turn
    # A start, then prompt, text, usage and output per turn.
    assert latest.total == 1 + 3 * 4
    prompt, text, usage, output = [e.item for e in latest.entries]
    assert (prompt, text, output) == (
        PromptEntry(text="c"),
        TextDelta(text="turn 3: echo c"),
        Output(output="turn 3: echo c", stop_reason="end_turn"),
    )
    assert isinstance(usage, Usage)
    assert usage.requests == 1
    assert latest.has_older
    assert [e.seq for e in earlier.entries] == list(range(9))
    assert earlier.has_older is False


async def test_history_from_before_usage_loads(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text(
        '{"seq": 0, "at": 1.0, "turn": "t", "item": {"type": "prompt", "text": "old"}}\n'
        '{"seq": 1, "at": 2.0, "turn": "t", "item": {"type": "output", "output": "x", "stop_reason": "end_turn"}}\n'
    )
    page = await HistoryLog(anyio.Path(path)).page()
    assert [e.item for e in page.entries] == [PromptEntry(text="old"), Output(output="x", stop_reason="end_turn")]
