from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import wake_prompt
from aid.protocol import HistoryItem, MessageEntry, Output, PromptEntry, SessionStatus, Started, TextDelta
from aid.spec import McpHttp, PermissionMode
from tests.conftest import acp_spec, py_spec
from tests.test_coding import plan

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def test_wake_prompt() -> None:
    assert wake_prompt([MessageEntry(sender=None, text="hi")]) == "Message from a person using aid:\n\nhi"
    two = wake_prompt([MessageEntry(sender="alice", text="a"), MessageEntry(sender=None, text="b")])
    assert two.startswith("Message from aid session 'alice':\n\na\n\n---\n\nMessage from a person using aid:\n\nb")
    assert "send_message tool" in two


def test_wake_prompt_renders_background_reports() -> None:
    report = MessageEntry(sender="background task bg1", text="Background task bg1 finished: exit 0 after 0.5s")
    assert wake_prompt([report]) == "Background task bg1 finished: exit 0 after 0.5s"
    mixed = wake_prompt([report, MessageEntry(sender=None, text="hi")])
    assert mixed.startswith("Background task bg1 finished: exit 0 after 0.5s\n\n---\n\nMessage from a person")
    assert "send_message tool" not in mixed
    with_session = wake_prompt([report, MessageEntry(sender="alice", text="a")])
    assert with_session.startswith("Background task bg1 finished")
    assert "Message from aid session 'alice'" in with_session
    assert "send_message tool" in with_session


async def outputs_after_message(session: aid.Session, count: int = 1) -> list[HistoryItem]:
    """The session's history, less its worker starts, once `count` outputs follow its first message."""
    while True:
        entries = (await session.history(limit=1000)).entries
        items: list[HistoryItem] = [e.item for e in entries if not isinstance(e.item, Started)]
        first = next((i for i, item in enumerate(items) if isinstance(item, MessageEntry)), None)
        if first is not None and sum(isinstance(item, Output) for item in items[first:]) >= count:
            return items
        await anyio.sleep(0.05)


async def test_a_message_wakes_an_acp_session(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            await client.send_message("acp", "hi", sender="other")
            items = await outputs_after_message(session)
    woken = f"echo: {wake_prompt([MessageEntry(sender='other', text='hi')])}"
    assert items == [
        MessageEntry(sender="other", text="hi"),
        TextDelta(text=woken),
        Output(output=woken, stop_reason="end_turn"),
    ]


async def test_a_message_waits_for_the_running_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            async for event in session.stream("slow"):
                if event == TextDelta(text="waiting"):
                    await client.send_message("acp", "queued")
                    with pytest.raises(aid.AidError, match="busy"):
                        await session.run("second prompt")
                    await session.cancel()
            # The message lands mid-turn, so the cancelled turn's output follows it too.
            items = await outputs_after_message(session, count=2)
    assert items[0] == PromptEntry(text="slow")
    assert Output(output="waiting", stop_reason="cancelled") in items
    woken = f"echo: {wake_prompt([MessageEntry(sender=None, text='queued')])}"
    assert items[-1] == Output(output=woken, stop_reason="end_turn")


async def test_a_message_wakes_a_pydantic_ai_session(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
            await session.stop()
            await client.send_message("echo", "hello")
            items = await outputs_after_message(session)
    assert isinstance(items[-1], Output)
    assert str(items[-1].output).startswith("turn 1: echo Message from a person using aid")


async def test_refused_messages(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("acp", acp_spec(tmp_path))
            with pytest.raises(aid.AidError, match="itself"):
                await client.send_message("acp", "x", sender="acp")
            with pytest.raises(aid.AidError, match="no session"):
                await client.send_message("nobody", "x")
            with pytest.raises(aid.AidError, match="as turns"):
                await client.receive_messages("acp", wait=0)


async def test_a_pydantic_ai_agent_messages_another_session(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            recipient = await client.create("a", acp_spec(tmp_path))
            sender = await client.create("sender", py_spec(tmp_path, "agents:messenger"))
            await sender.run("go")
            items = await outputs_after_message(recipient)
    assert items[0] == MessageEntry(sender="sender", text="a")


async def test_status_shows_the_turn_and_waiting_messages(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path, SECRET="hidden").model_copy(
        update={"mcp_servers": [McpHttp(name="web", url="http://x", headers={"Authorization": "hidden"})]}
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", spec)
            idle = await session.status()
            during: list[SessionStatus] = []
            async for event in session.stream("slow"):
                if event == TextDelta(text="waiting"):
                    await client.send_message("acp", "queued")
                    during.append(await session.status())
                    await session.cancel()
            await session.stop()
            stopped = await session.status()
    assert (idle.running, idle.busy, idle.pending) == (True, False, 0)
    assert [(s.busy, s.pending) for s in during] == [(True, 1)]
    assert (stopped.running, stopped.pid) == (False, None)
    assert idle.runs.endswith("fake_acp_agent.py")
    assert idle.mcp_servers == ["web", "aid"]
    assert "hidden" not in idle.model_dump_json()


async def test_monitor_resumes_an_idle_loop(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("watcher", py_spec(tmp_path, "agents:watcher", PermissionMode.ALLOW))
            assert str((await session.run("go")).output) == "armed"
            items = await outputs_after_message(session)
    outputs = [item for item in items if isinstance(item, Output)]
    assert len(outputs) == 2
    assert str(outputs[0].output) == "armed"
    message = next(item for item in items if isinstance(item, MessageEntry))
    # Turn 1 answered before the notification arrived: the loop was idle when it resumed.
    assert items.index(outputs[0]) < items.index(message)
    assert message.sender == "background task bg1"
    assert message.text.startswith("Background task bg1 finished: exit 0")
    assert f"woke: {message.text}" in str(outputs[1].output)


async def test_monitor_waits_for_the_running_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            await session.run(
                plan(
                    ("background", {"argv": ["sleep", "0.5"]}),
                    ("monitor", {"task_id": "bg1"}),
                    ("python", {"script": "import time; time.sleep(3)", "mode": "sync"}),
                )
            )
            items = await outputs_after_message(session, count=2)
    outputs = [item for item in items if isinstance(item, Output)]
    assert len(outputs) == 2
    message = next(item for item in items if isinstance(item, MessageEntry))
    # The notification lands mid-turn: ahead of the running turn's own output.
    assert items.index(message) < items.index(outputs[0])
    assert message.sender == "background task bg1"
    assert message.text.startswith("Background task bg1 finished: exit 0")
    assert "watching bg1 as m1" in str(outputs[1].output)
