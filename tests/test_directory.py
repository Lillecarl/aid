from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.protocol import AidError, Lifecycle, MessageEntry, TextDelta, ToolCall
from aid.spec import PermissionMode
from tests.conftest import acp_spec, py_spec
from tests.test_coding import plan

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30
MOVE_SENDER = "aid:directory"


async def until_message(session: aid.Session, sender: str) -> MessageEntry:
    while True:
        entries = (await session.history(limit=1000)).entries
        if found := [e.item for e in entries if isinstance(e.item, MessageEntry) and e.item.sender == sender]:
            return found[-1]
        await anyio.sleep(0.05)


async def test_a_person_moves_a_session_and_its_worker_restarts_there(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "there").mkdir()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            before = await session.run("env")
            moved = await session.change_directory("there")
            after = await session.run("env")
            status = await session.status()
            entries = (await session.history(limit=1000)).entries
    assert moved == str((tmp_path / "there").resolve())
    assert str(before.output).endswith(str(tmp_path))
    assert str(after.output).endswith(moved)
    assert status.cwd == moved
    assert Lifecycle(event="moved", detail=moved) in [e.item for e in entries]


async def test_a_move_is_refused_where_it_cannot_happen(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("not a directory\n")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            with pytest.raises(AidError, match="not a directory"):
                await session.change_directory("file")
            async for event in session.stream("slow"):
                if event == TextDelta(text="waiting"):
                    with pytest.raises(AidError, match="in a turn"):
                        await session.change_directory("/")
                    await session.cancel()
            with pytest.raises(AidError, match="no directory request"):
                await session.answer_directory(True)
            assert (await session.status()).cwd == str(tmp_path)


async def test_an_approved_request_moves_the_session_when_its_turn_ends(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "marker.txt").write_text("here\n")
    sub = str((tmp_path / "sub").resolve())
    steps = plan(
        ("change_directory", {"path": "sub", "reason": "the work is there"}),
        ("python", {"script": "import time; time.sleep(1.5)", "mode": "sync"}),
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            during = after_answer = None
            async for event in session.stream(steps):
                if isinstance(event, ToolCall) and event.status == "completed" and during is None:
                    during = await session.status()
                    await session.answer_directory(True)
                    after_answer = await session.status()
            told = await until_message(session, MOVE_SENDER)
            while (await session.status()).busy:  # noqa: ASYNC110 -- the message wakes a turn of its own
                await anyio.sleep(0.05)
            listed = await session.run(plan(("ls", {})))
            status = await session.status()
    assert during is not None and during.directory_request is not None
    assert (during.directory_request.cwd, during.directory_request.reason) == (sub, "the work is there")
    # Approved mid-turn: the session stays put until the turn ends.
    assert after_answer is not None
    assert after_answer.directory_request is not None and after_answer.directory_request.approved
    assert after_answer.cwd == str(tmp_path)
    assert told.text == f"A person approved the move: this session's working directory is now {sub}."
    assert status.cwd == sub and status.directory_request is None
    assert str(listed.output) == "marker.txt"


async def test_a_declined_request_leaves_the_session_where_it_is(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("change_directory", {"path": "sub", "reason": "tidier"})))
            waiting = await session.status()
            await session.answer_directory(False)
            told = await until_message(session, MOVE_SENDER)
            status = await session.status()
    assert "a person decides" in str(result.output)
    # Allow-all permission does not approve it.
    assert waiting.directory_request is not None and not waiting.directory_request.approved
    assert told.text == f"A person declined moving this session to {(tmp_path / 'sub').resolve()}."
    assert status.cwd == str(tmp_path) and status.directory_request is None
