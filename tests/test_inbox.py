from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from aid.history import HistoryLog
from aid.inbox import Inbox
from aid.protocol import MessageEntry, TurnError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


async def test_restore_keeps_pending_messages_and_their_order(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    path = anyio.Path(tmp_path / "inbox.json")
    inbox = Inbox(path, history)
    await inbox.initialize()
    first = MessageEntry(sender="sender", text="first")
    second = MessageEntry(sender=None, text="second")
    await inbox.accept(first, "m1")
    await inbox.accept(second, "m2")
    restored = Inbox(path, history)
    assert await restored.restore() is None
    assert len(restored) == 2
    assert await restored.claim("wake") == [first, second]
    await restored.finish("wake")
    again = Inbox(path, history)
    assert await again.restore() is None
    assert len(again) == 0


async def test_interrupted_dispatch_is_reported_without_replay(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    path = anyio.Path(tmp_path / "inbox.json")
    inbox = Inbox(path, history)
    await inbox.initialize()
    first = MessageEntry(sender="sender", text="possible effects")
    second = MessageEntry(sender=None, text="still pending")
    await inbox.accept(first, "m1")
    assert await inbox.claim("interrupted") == [first]
    await inbox.accept(second, "m2")
    restored = Inbox(path, history)
    assert "interrupted" in (await restored.restore() or "")
    assert await restored.claim("next") == [second]
    await restored.finish("next")
    entries = await history.entries()
    assert entries[-1].turn == "interrupted"
    assert isinstance(entries[-1].item, TurnError)
    assert entries[-1].item.code == "delivery_uncertain"
    again = Inbox(path, history)
    assert await again.restore() is None
    assert len(again) == 0
    assert len(await history.entries()) == len(entries)


async def test_stale_completion_does_not_close_a_new_dispatch(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    path = anyio.Path(tmp_path / "inbox.json")
    inbox = Inbox(path, history)
    await inbox.initialize()
    await inbox.accept(MessageEntry(sender=None, text="first"), "m1")
    await inbox.claim("first")
    await inbox.finish("first")
    await inbox.accept(MessageEntry(sender=None, text="second"), "m2")
    await inbox.claim("second")
    await inbox.finish("first")
    assert "second" in (await Inbox(path, history).restore() or "")


async def test_rejected_dispatch_returns_batch_ahead_of_new_messages(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    path = anyio.Path(tmp_path / "inbox.json")
    inbox = Inbox(path, history)
    await inbox.initialize()
    first = MessageEntry(sender=None, text="rejected")
    second = MessageEntry(sender=None, text="new")
    await inbox.accept(first, "m1")
    await inbox.claim("rejected")
    await inbox.accept(second, "m2")
    await inbox.release("stale")
    assert len(inbox) == 1
    await inbox.release("rejected")
    assert len(inbox) == 2
    restored = Inbox(path, history)
    assert await restored.restore() is None
    assert await restored.claim("retry") == [first, second]


async def test_migration_does_not_replay_historical_messages(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    await history.append(MessageEntry(sender=None, text="historical effects"), "old")
    inbox = Inbox(anyio.Path(tmp_path / "inbox.json"), history)
    assert await inbox.restore() is None
    assert len(inbox) == 0
    new = MessageEntry(sender=None, text="new work")
    await inbox.accept(new, "new")
    assert await inbox.claim("wake") == [new]


async def test_initialization_preserves_an_early_dispatch(tmp_path: Path) -> None:
    history = HistoryLog(anyio.Path(tmp_path / "history.jsonl"))
    path = anyio.Path(tmp_path / "inbox.json")
    inbox = Inbox(path, history)
    await inbox.accept(MessageEntry(sender=None, text="early message"), "m1")
    await inbox.claim("early")
    await inbox.initialize()
    await inbox.finish("early")
    restored = Inbox(path, history)
    assert await restored.restore() is None
    assert len(restored) == 0
