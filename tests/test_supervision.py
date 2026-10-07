from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import (
    SUPERVISION_ITEM_CHARS,
    SUPERVISION_LOOP_THRESHOLD,
    supervision_cycle,
    supervision_text,
)
from aid.protocol import HistoryEntry, HistoryPage, MessageEntry, Output, PromptEntry, TextDelta, Usage
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
    text = supervision_text("coder", "done", page)
    assert text.startswith("Supervised session 'coder' ended a turn (done).")
    assert "> do the thing" in text
    assert "more characters" in text
    assert "[usage" not in text
    assert f"stuck at {SUPERVISION_LOOP_THRESHOLD:.2f}" in text


def test_supervision_text_without_outputs() -> None:
    page = HistoryPage(entries=[_entry(0, PromptEntry(text="hi"))], has_older=False, has_newer=False, total=1)
    assert "Loop score: n/a" in supervision_text("coder", "done", page)


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


async def test_supervisor_link_validation(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(aid.AidError, match="unknown_supervisor"):
                await client.create("a", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "nobody"}))
            with pytest.raises(aid.AidError, match="self_supervised"):
                await client.create("b", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "b"}))
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))


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
