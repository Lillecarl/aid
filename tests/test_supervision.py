from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import anyio
import pytest

import aid
from aid.daemon import (
    SUPERVISION_ITEM_CHARS,
    SUPERVISION_LOOP_THRESHOLD,
    add_goal,
    set_goal,
    supervision_cycle,
    supervision_text,
)
from aid.protocol import (
    AddGoal,
    Goal,
    GoalState,
    HistoryEntry,
    HistoryPage,
    MessageEntry,
    Output,
    PromptEntry,
    SetGoal,
    SupervisedHistory,
    TextDelta,
    ToolCall,
    Usage,
)
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths
    from aid.protocol import HistoryItem

pytestmark = pytest.mark.anyio

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))


def test_supervision_cycle() -> None:
    assert not supervision_cycle({}, "a", "boss")
    assert not supervision_cycle({"boss": None}, "a", "boss")
    assert supervision_cycle({"boss": "a"}, "a", "boss")
    assert supervision_cycle({"boss": "mid", "mid": "a"}, "a", "boss")
    assert not supervision_cycle({"boss": "mid", "mid": None}, "a", "boss")
    assert not supervision_cycle({"boss": "gone"}, "a", "boss")


def _entry(seq: int, item: HistoryItem) -> HistoryEntry:
    return HistoryEntry(seq=seq, at=float(seq), turn="t", item=item)


def test_supervision_text_clips_and_scores() -> None:
    repeated = "turn 1: echo do the thing with extra words to separate it a little"
    page = HistoryPage(
        entries=[
            _entry(0, PromptEntry(text="do the thing")),
            _entry(1, TextDelta(text="x" * (SUPERVISION_ITEM_CHARS + 10))),
            _entry(2, Output(output=repeated, stop_reason="end_turn")),
            _entry(3, Output(output=repeated, stop_reason="end_turn")),
            _entry(4, Usage(input_tokens=3, output_tokens=4)),
        ],
        has_older=False,
        has_newer=False,
        total=5,
    )
    text = supervision_text("coder", "done", page, [])
    assert text.startswith("Supervised session 'coder' ended a turn (done).")
    assert "> do the thing" in text
    assert "more characters" in text
    assert "[usage" not in text
    assert "No open goals." in text
    assert f"stuck at {SUPERVISION_LOOP_THRESHOLD:.2f}" in text


def test_supervision_text_without_outputs() -> None:
    page = HistoryPage(entries=[_entry(0, PromptEntry(text="hi"))], has_older=False, has_newer=False, total=1)
    assert "Loop score: n/a" in supervision_text("coder", "done", page, [])


def test_supervision_text_lists_open_goals() -> None:
    page = HistoryPage(entries=[_entry(0, PromptEntry(text="hi"))], has_older=False, has_newer=False, total=1)
    goals = [
        Goal(id="a1b2c3d4e5", text="ship it", state=GoalState.OPEN, updated_at=1.0),
        Goal(id="f6g7h8i9j0", text="skip it", state=GoalState.DROPPED, updated_at=2.0),
    ]
    text = supervision_text("coder", "done", page, [g for g in goals if g.state is GoalState.OPEN])
    assert "- a1b2c3d4: ship it" in text
    assert "skip it" not in text


def test_goal_mutators() -> None:
    goals: list[Goal] = []
    first = add_goal(goals, "ship it")
    assert first.state is GoalState.OPEN
    done = set_goal(goals, first.id, GoalState.DONE)
    assert done.state is GoalState.DONE
    assert goals[0].state is GoalState.DONE
    with pytest.raises(aid.AidError, match="unknown_goal"):
        set_goal(goals, "nope", GoalState.DROPPED)


async def supervised_firing(session: aid.Session, name: str) -> HistoryEntry:
    """The firing message the supervisor got about `name`, once it arrives."""
    while True:
        entries = (await session.history(limit=1000)).entries
        found = [e for e in entries if isinstance(e.item, MessageEntry) and e.item.sender == name]
        if found:
            return found[0]
        await anyio.sleep(0.05)


async def output_after(session: aid.Session, seq: int) -> None:
    """Wait until an Output lands after `seq`: the wake turn over the firing finished."""
    while True:
        entries = (await session.history(limit=1000)).entries
        if any(isinstance(e.item, Output) and e.seq > seq for e in entries):
            return
        await anyio.sleep(0.05)


async def supervised_firings(session: aid.Session, name: str, count: int) -> list[HistoryEntry]:
    """Every firing message the supervisor got about `name`, once `count` arrived."""
    while True:
        entries = (await session.history(limit=1000)).entries
        found = [e for e in entries if isinstance(e.item, MessageEntry) and e.item.sender == name]
        if len(found) >= count:
            return found
        await anyio.sleep(0.05)


def completed_outputs(items: list[HistoryItem]) -> list[str]:
    """What the supervisor's tool calls returned: completed calls carry the result as their output."""
    return [item.output for item in items if isinstance(item, ToolCall) and isinstance(item.output, str)]


async def test_supervisor_link_validation(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(aid.AidError, match="unknown_supervisor"):
                await client.create("a", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "nobody"}))
            with pytest.raises(aid.AidError, match="self_supervised"):
                await client.create("b", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "b"}))
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))


async def test_supervision_ops_refuse_strangers(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            with pytest.raises(aid.AidError, match="not_supervised"):
                await client.call(SupervisedHistory(supervisor="stranger", session="coder"))
            with pytest.raises(aid.AidError, match="not_supervised"):
                await client.call(AddGoal(supervisor="coder", session="coder", text="self-dealing"))
            with pytest.raises(aid.AidError, match="empty_goal"):
                await client.call(AddGoal(supervisor="boss", session="coder", text="  "))
            with pytest.raises(aid.AidError, match="unknown_goal"):
                await client.call(SetGoal(supervisor="boss", session="coder", goal_id="nope", state=GoalState.DONE))


def read_goals(paths: Paths, name: str) -> list[dict[str, Any]]:
    return json.loads((paths.session_dir(name) / "goals.json").read_bytes())


async def test_supervisor_reads_history_and_tracks_goals(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_probe", AID_SUPERVISED="coder"))
            coder = await client.create(
                "coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"})
            )
            await coder.run("do the thing")
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
            first_goals = read_goals(daemon, "coder")
            await coder.run("do more")
            firings = await supervised_firings(boss, "coder", 2)
            await output_after(boss, firings[1].seq)
            settled_goals = read_goals(daemon, "coder")
            await coder.run("wrap up")
            firings = await supervised_firings(boss, "coder", 3)
            await output_after(boss, firings[2].seq)
            last = [e.item for e in (await boss.history(limit=1000)).entries]
    assert isinstance(firings[0].item, MessageEntry)
    assert "No open goals." in firings[0].item.text
    outputs = completed_outputs(boss_items)
    assert any("do the thing" in out for out in outputs)
    assert sorted(g["text"] for g in first_goals) == ["ship it", "skip it"]
    assert all(g["state"] == "open" for g in first_goals)
    assert isinstance(firings[1].item, MessageEntry)
    assert "ship it" in firings[1].item.text
    assert "skip it" in firings[1].item.text
    assert {g["text"]: g["state"] for g in settled_goals} == {"ship it": "done", "skip it": "dropped"}
    assert isinstance(firings[2].item, MessageEntry)
    assert "No open goals." in firings[2].item.text
    assert isinstance(last[-1], Output)
    assert last[-1].output == "all done"


async def test_supervised_turn_end_fires_supervisor(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:echo"))
            coder = await client.create(
                "coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"})
            )
            await coder.run("do the thing")
            firing = await supervised_firing(boss, "coder")
            assert isinstance(firing.item, MessageEntry)
            await output_after(boss, firing.seq)
            await anyio.sleep(0.5)
            coder_items = [e.item for e in (await coder.history(limit=1000)).entries]
    assert "ended a turn" in firing.item.text
    assert "do the thing" in firing.item.text
    assert "Loop score" in firing.item.text
    # The echo supervisor answers with the prompt, which asks for nothing: silence, no nudge.
    assert not [i for i in coder_items if isinstance(i, MessageEntry) and i.sender == "boss"]
