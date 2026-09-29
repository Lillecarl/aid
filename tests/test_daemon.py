from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import Daemon
from aid.protocol import AidError, Cost, Output, Started, TextDelta, Usage
from aid.spec import PermissionMode
from tests.agents import Review
from tests.conftest import acp_spec, py_spec
from tests.fake_acp_agent import AGENT_NAME, AGENT_VERSION, CHUNKS, MODEL, RESOLVED_MODEL

if TYPE_CHECKING:
    from pathlib import Path

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


async def test_acp_prompt_keeps_chunk_order(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            result = await session.run("count")
    expected = "".join(f"{i} " for i in range(CHUNKS))
    assert result.text == expected
    assert result.output == expected
    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize(("mode", "answer"), [(PermissionMode.DENY, "no"), (PermissionMode.ALLOW, "yes")])
async def test_acp_permission_policy(daemon: Paths, tmp_path: Path, mode: PermissionMode, answer: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("perm", acp_spec(tmp_path, mode))
            result = await session.run("permission")
    assert result.text == answer


async def test_acp_agent_gets_spec_env_and_cwd(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("env", acp_spec(tmp_path, AID_TEST_VAR="from-spec"))
            result = await session.run("env")
    assert result.text == f"from-spec {tmp_path}"


async def test_acp_cancel(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("slow", acp_spec(tmp_path))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("slow"):
                events.append(event)
                if event == TextDelta(text="waiting"):
                    await session.cancel()
    assert events[-1] == Output(output="waiting", stop_reason="cancelled")


async def test_acp_session_survives_stop(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("restart", acp_spec(tmp_path))
            first = await session.run("pid")
            await session.stop()
            assert [s.running for s in await client.sessions()] == [False]
            second = await session.run("pid")
            again = await session.run("hello")
    assert first.text != second.text
    assert again.text == "echo: hello"


async def test_start_runs_a_stopped_session_without_a_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("wake", acp_spec(tmp_path))
            await session.stop()
            await session.start()
            started = await session.status()
            await session.start()
            again = await session.status()
            history = await session.history()
    assert started.running
    assert not started.busy
    assert again.pid == started.pid
    first, second = [e.item for e in history.entries]
    assert isinstance(first, Started)
    assert isinstance(second, Started)
    assert not first.resumed
    assert second.resumed
    assert second.agent_session == first.agent_session == started.agent_session
    assert second.pid == started.pid


async def test_acp_usage_is_recorded(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("usage", acp_spec(tmp_path))
            events = [e async for e in session.stream("usage")]
            status = await session.status()
    assert events[-2:] == [
        Usage(
            input_tokens=10,
            output_tokens=6,
            cache_read_tokens=8,
            cache_write_tokens=2,
            models=[RESOLVED_MODEL],
            context_used=1234,
            context_size=200_000,
            session_cost=Cost(amount=0.5, currency="USD"),
        ),
        Output(output="counted", stop_reason="end_turn"),
    ]
    assert (status.agent, status.model) == (f"{AGENT_NAME} {AGENT_VERSION}", MODEL)
    assert status.agent_session


async def test_failed_start_leaves_no_session(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path).model_copy(update={"command": ["/nonexistent/agent"]})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(AidError) as error:
                await client.create("broken", spec)
            assert error.value.code == "start_failed"
            assert "/nonexistent/agent" in error.value.message
            assert await client.sessions() == []


async def test_error_leaves_connect_unwrapped(daemon: Paths) -> None:
    with anyio.fail_after(TIMEOUT), pytest.raises(AidError) as error:
        async with aid.connect(daemon) as client:
            await client.session("missing").run("x")
    assert error.value.code == "not_found"


async def test_pydantic_ai_typed_output(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("review", py_spec(tmp_path, "agents:reviewer"))
            result = await session.run("review this", output_type=Review)
    assert result.output == Review(verdict="approve", score=7)


async def test_pydantic_ai_env_and_cwd(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("env", py_spec(tmp_path, "agents:echo", AID_TEST_VAR="py"))
            result = await session.run("env")
    assert result.output == f"py {tmp_path}"


async def test_pydantic_ai_cancel(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("slow", py_spec(tmp_path, "agents:echo"))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("slow"):
                events.append(event)
                if event == TextDelta(text="waiting"):
                    await session.cancel()
            after = await session.run("hi")
    assert events[-1] == Output(output=None, stop_reason="cancelled")
    assert after.output == "turn 1: echo hi"


async def test_pydantic_ai_history_survives_daemon_restart(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher).serve)
            async with aid.connect(paths) as client:
                session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
                assert (await session.run("a")).output == "turn 1: echo a"
                assert (await session.run("b")).output == "turn 2: echo b"
            tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher).serve)
            async with aid.connect(paths) as client:
                assert [s.name for s in await client.sessions()] == ["echo"]
                assert (await client.session("echo").run("c")).output == "turn 3: echo c"
            tg.cancel_scope.cancel()


async def test_delete_removes_state(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("gone", py_spec(tmp_path, "agents:echo"))
            await session.run("x")
            await session.delete()
            assert await client.sessions() == []
    assert not daemon.session_dir("gone").exists()


async def test_session_state_is_private(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("private", py_spec(tmp_path, "agents:echo", TOKEN="secret"))
    assert daemon.session_dir("private").stat().st_mode & 0o777 == 0o700
