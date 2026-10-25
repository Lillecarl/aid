from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
import zmq.asyncio
from pydantic import ValidationError

import aid
from aid.daemon import Daemon
from aid.protocol import (
    AidError,
    Cost,
    Failure,
    GetHistory,
    GetSpec,
    GetStatus,
    Lifecycle,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    Started,
    TextDelta,
    ToolMode,
    TurnError,
    Usage,
    decode_reply,
    encode,
)
from aid.spec import AgentSpecAdapter, PermissionMode
from tests.agents import Review
from tests.conftest import acp_spec, py_spec
from tests.fake_acp_agent import AGENT_NAME, AGENT_VERSION, CHUNKS, RESOLVED_MODEL

if TYPE_CHECKING:
    from pathlib import Path

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths
    from aid.protocol import HistoryItem

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
            history = await session.history()
    assert result.text == answer
    decision = next(e.item for e in history.entries if isinstance(e.item, PermissionDecision))
    assert (decision.option_id, decision.by) == (answer, PermissionDecider.POLICY)


@pytest.mark.parametrize("mode", [PermissionMode.ASK, PermissionMode.SUPERVISOR, PermissionMode.ASK_SUPERVISOR])
async def test_acp_permission_waits_for_a_person(daemon: Paths, tmp_path: Path, mode: PermissionMode) -> None:
    """Every waiting mode waits alike in the engine; without a supervisor, the person answers."""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("ask", acp_spec(tmp_path, mode))
            events: list[aid.SessionEvent] = []
            waiting: list[PermissionRequest] = []
            async for event in session.stream("permission"):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    waiting = (await session.status()).permissions
                    with pytest.raises(AidError, match="no option"):
                        await session.answer(event.request_id, "maybe")
                    await session.answer(event.request_id, "yes")
            after = (await session.status()).permissions
            with pytest.raises(AidError, match="waits on no permission request"):
                await session.answer(waiting[0].request_id, "no")
            recorded = [e.item for e in (await session.history()).entries]
    request = next(e for e in events if isinstance(e, PermissionRequest))
    assert (request.tool_call_id, request.title, [o.option_id for o in request.options], request.mode) == (
        "t1",
        "rm -rf /",
        ["yes", "no"],
        mode,
    )
    assert waiting == [request]
    assert after == []
    assert PermissionDecision(request_id=request.request_id, option_id="yes", by=PermissionDecider.PERSON) in events
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "yes"
    assert request in recorded


async def test_acp_permission_cancelled_with_the_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("ask", acp_spec(tmp_path, PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("permission"):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.cancel()
    decision = next(e for e in events if isinstance(e, PermissionDecision))
    assert (decision.option_id, decision.by) == (None, PermissionDecider.CANCEL)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "cancelled"


async def test_acp_permission_refused_when_nobody_answers(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("ask", acp_spec(tmp_path, PermissionMode.ASK, permission_timeout=0.2))
            events = [event async for event in session.stream("permission")]
    decision = next(e for e in events if isinstance(e, PermissionDecision))
    assert (decision.option_id, decision.by) == ("no", PermissionDecider.TIMEOUT)


async def test_acp_answer_carries_words_with_the_pick(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("ask", acp_spec(tmp_path, PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("permission"):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "yes", "only the one directory")
    assert (
        PermissionDecision(
            request_id=next(e for e in events if isinstance(e, PermissionRequest)).request_id,
            option_id="yes",
            by=PermissionDecider.PERSON,
            text="only the one directory",
        )
        in events
    )
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "yes"


async def test_acp_answer_refuses_words_without_a_pick(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("ask", acp_spec(tmp_path, PermissionMode.ASK))
            async for event in session.stream("permission"):
                if isinstance(event, PermissionRequest):
                    with pytest.raises(AidError, match="takes an option"):
                        await session.answer(event.request_id, None, "just wipe tmp")
                    await session.answer(event.request_id, "no")


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
    # The model the turn used, over the option's value at the start.
    assert (status.agent, status.model) == (f"{AGENT_NAME} {AGENT_VERSION}", RESOLVED_MODEL)
    assert (status.context_used, status.context_size) == (1234, 200_000)
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


async def test_a_request_of_several_frames_is_refused(daemon: Paths) -> None:
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(daemon.control)
    try:
        with anyio.fail_after(TIMEOUT):
            await sock.send_multipart([b"{}", b"{}"])  # pyright: ignore[reportUnknownMemberType] -- pyzmq types msg_parts as a bare Sequence
            refused = decode_reply(await sock.recv())
            async with aid.connect(daemon) as client:
                assert await client.sessions() == []  # The daemon still serves.
    finally:
        sock.close(linger=0)
        ctx.term()
    assert isinstance(refused, Failure)
    assert refused.code == "invalid_request"


async def test_an_undecodable_request_fails_its_own_call(daemon: Paths, tmp_path: Path) -> None:
    """A field this daemon does not know (a newer client's) fails the call that sent it, under its id: a Failure
    under no id never reaches the waiting call, which then hangs."""
    spec = json.loads(py_spec(tmp_path, "agents:echo").model_dump_json()) | {"from_a_newer_client": True}
    request = {"op": "create", "id": "newer-1", "name": "s", "spec": spec}
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(daemon.control)
    try:
        with anyio.fail_after(TIMEOUT):
            await sock.send(json.dumps(request).encode())
            refused = decode_reply(await sock.recv())
    finally:
        sock.close(linger=0)
        ctx.term()
    assert isinstance(refused, Failure)
    assert (refused.id, refused.code) == ("newer-1", "invalid_request")
    assert "from_a_newer_client" in refused.message


async def test_a_request_id_in_flight_is_refused(daemon: Paths, tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    first, second = ctx.socket(zmq.DEALER), ctx.socket(zmq.DEALER)
    first.connect(daemon.control)
    second.connect(daemon.control)
    try:
        with anyio.fail_after(TIMEOUT):
            async with aid.connect(daemon) as client:
                await client.create("echo", py_spec(tmp_path, "agents:echo"))
            # Waits for an entry that never comes, so its id stays in flight.
            waiting = GetHistory(id="same", session="echo", after=1000, wait=5)
            await first.send(encode(waiting))
            await anyio.sleep(0.2)
            await second.send(encode(GetStatus(id="same", session="echo")))
            refused = decode_reply(await second.recv())
    finally:
        first.close(linger=0)
        second.close(linger=0)
        ctx.term()
    assert isinstance(refused, Failure)
    assert refused.code == "duplicate_id"


async def test_native_typed_output(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("review", py_spec(tmp_path, "agents:reviewer"))
            result = await session.run("review this", output_type=Review)
    assert result.output == Review(verdict="approve", score=7)


async def test_native_env_and_cwd(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("env", py_spec(tmp_path, "agents:echo", AID_TEST_VAR="py"))
            result = await session.run("env")
    assert result.output == f"py {tmp_path}"


async def test_native_cancel(daemon: Paths, tmp_path: Path) -> None:
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
    # The interrupted first turn is retained for the model, not just the visible daemon transcript.
    assert after.output == "turn 2: echo hi"


async def test_compact_session(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("s", py_spec(tmp_path, "agents:summarizer"))
            await session.run("first")
            await session.run("second")
            assert await session.compact("the billing work") == "kept decisions"
            entries = [entry.item for entry in (await session.history(limit=100)).entries]
            lifecycle = next(item for item in entries if isinstance(item, Lifecycle))
            assert (lifecycle.event, lifecycle.detail, lifecycle.summary) == (
                "compacted",
                "manual",
                "kept decisions",
            )
            assert (await session.run("third")).output == "kept decisions"


async def test_a_failed_turn_is_over_when_its_failure_arrives(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("failing", py_spec(tmp_path, "agents:failer"))
            for _ in range(3):
                with pytest.raises(aid.AidError, match="agent_error"):
                    await session.run("go")


async def test_compact_refuses_a_running_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("slow", py_spec(tmp_path, "agents:echo"))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("slow"):
                events.append(event)
                if event == TextDelta(text="waiting"):
                    with pytest.raises(aid.AidError, match="busy"):
                        await client.session("slow").compact()
                    await session.cancel()
    assert events[-1] == Output(output=None, stop_reason="cancelled")


async def settled(session: aid.Session) -> list[HistoryItem]:
    """The history once the session is idle with an answer last."""
    while True:
        items = [entry.item for entry in (await session.history(limit=100)).entries]
        if isinstance(items[-1], Output) and not (await session.status()).busy:
            return items
        await anyio.sleep(0.05)


async def test_a_message_during_compaction_waits_for_its_end(daemon: Paths, tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    spec = py_spec(tmp_path, "agents:slow_summarizer").model_copy(update={"env": {"AID_COMPACT_GATE": str(gate)}})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client, anyio.create_task_group() as tg:
            session = await client.create("s", spec)
            await session.run("first")
            await session.run("second")
            tg.start_soon(session.compact)
            while not gate.with_suffix(".started").exists():  # noqa: ASYNC110 -- a file the worker drops
                await anyio.sleep(0.05)
            await client.send_message("s", "a request to answer", sender="peer")
            gate.touch()
            items = await settled(session)
            assert not [item for item in items if isinstance(item, TurnError)]
            compacted = next(
                i for i, item in enumerate(items) if isinstance(item, Lifecycle) and item.event == "compacted"
            )
            assert any(isinstance(item, Output) for item in items[compacted:])


async def test_native_history_survives_daemon_restart(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, plugin_dirs=[]).serve)
            async with aid.connect(paths) as client:
                session = await client.create("echo", py_spec(tmp_path, "agents:echo"))
                assert (await session.run("a")).output == "turn 1: echo a"
                assert (await session.run("b")).output == "turn 2: echo b"
            tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, plugin_dirs=[]).serve)
            async with aid.connect(paths) as client:
                assert [s.name for s in await client.sessions()] == ["echo"]
                assert (await client.session("echo").run("c")).output == "turn 3: echo c"
            tg.cancel_scope.cancel()


async def test_an_older_spec_file_loads_and_is_rewritten_as_the_model_reads_it(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    spec_file = paths.session_dir("old") / "spec.json"
    spec_file.parent.mkdir(parents=True)
    stored = json.loads(AgentSpecAdapter.dump_json(py_spec(tmp_path, "agents:echo", PermissionMode.ASK)))
    spec_file.write_text(json.dumps({**stored, "supervisor_permissions": ["ask_user", "shell"]}))
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, plugin_dirs=[]).serve)
            async with aid.connect(paths) as client:
                spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session="old")))
            tg.cancel_scope.cancel()
            modes = {"ask_user": PermissionMode.ASK_SUPERVISOR, "shell": PermissionMode.ASK_SUPERVISOR}
            assert spec.tool_permissions == modes
            assert AgentSpecAdapter.validate_json(spec_file.read_bytes()) == spec
            assert "supervisor_permissions" not in json.loads(spec_file.read_bytes())


async def test_a_stopped_session_lists_its_tools_after_a_daemon_restart(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    before: list[ToolMode] = []
    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, plugin_dirs=[]).serve)
            async with aid.connect(paths) as client:
                session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
                await session.start()
                await session.set_permissions({"shell": PermissionMode.ALLOW})
                before = (await session.status()).tools
            tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            await tg.start(Daemon(paths, launcher, plugin_dirs=[]).serve)
            async with aid.connect(paths) as client:
                after = await client.session("coder").status()
                assert not after.running
                assert after.tools == before
                assert [(t.mode, t.set) for t in after.tools if t.name == "shell"] == [(PermissionMode.ALLOW, True)]
            tg.cancel_scope.cancel()


async def test_delete_removes_state(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("gone", py_spec(tmp_path, "agents:echo"))
            await session.run("x")
            await session.delete()
            assert await client.sessions() == []
    assert not daemon.session_dir("gone").exists()


@pytest.mark.parametrize("target", ["agents:no_such_agent", "agents:echo"])
async def test_a_delete_during_the_first_start_waits_for_it(daemon: Paths, tmp_path: Path, target: str) -> None:
    """A delete that arrives while the session's first worker is starting waits for the start to end, then stops
    what started. Each path removes only the session it holds: no KeyError, no worker or directory left behind."""
    spec_file = daemon.session_dir("doomed") / "spec.json"
    deleted: list[str] = []

    async def create(client: aid.Client) -> None:
        try:
            await client.create("doomed", py_spec(tmp_path, target))
        except AidError as e:
            assert e.code != "internal", e.message

    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as creator, aid.connect(daemon) as deleter, anyio.create_task_group() as tg:
            tg.start_soon(create, creator)
            while not spec_file.exists():  # noqa: ASYNC110 -- the daemon writes it before starting the worker
                await anyio.sleep(0.005)
            try:
                await deleter.session("doomed").delete()
                deleted.append("deleted")
            except AidError as e:
                deleted.append(e.code)
        async with aid.connect(daemon) as client:
            assert await client.sessions() == []
    assert deleted in (["deleted"], ["not_found"]), deleted
    assert not daemon.session_dir("doomed").exists()
    assert not (daemon.runtime_dir / "hook-doomed.sock").exists()


async def test_tags_merge_into_the_saved_spec(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            spec = py_spec(tmp_path, "agents:echo").model_copy(update={"tags": {"team": "web", "ticket": "42"}})
            session = await client.create("tagged", spec)
            await session.set_tags({"ticket": None, "stage": "review"})
            listed = {info.name: info.tags for info in await client.sessions()}
            with pytest.raises(ValidationError):
                await session.set_tags({"two words": "x"})
    saved = AgentSpecAdapter.validate_json((daemon.session_dir("tagged") / "spec.json").read_bytes())
    assert listed == {"tagged": {"team": "web", "stage": "review"}}
    assert saved.tags == {"team": "web", "stage": "review"}


async def test_session_state_is_private(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("private", py_spec(tmp_path, "agents:echo", TOKEN="secret"))
    assert daemon.session_dir("private").stat().st_mode & 0o777 == 0o700
