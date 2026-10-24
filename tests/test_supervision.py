from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import pytest

import aid
from aid import events
from aid.daemon import Daemon, supervision_cycle
from aid.paths import ENV_STATE_DIR
from aid.protocol import (
    CallPlugin,
    Cancel,
    Goal,
    GoalState,
    HistoryEntry,
    HistoryPage,
    Lifecycle,
    MessageEntry,
    Output,
    PermissionChoice,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    QueryResult,
    SendMessage,
    TextDelta,
    ToolCall,
    Usage,
)
from aid.spec import AgentSpecAdapter, NativeSpec, PermissionMode
from plugins.supervision.policy import (
    SUPERVISION_ITEM_CHARS,
    SUPERVISION_LOOP_THRESHOLD,
    add_goal,
    loop_score,
    nudge_repeats,
    permission_review_text,
    set_goal,
    supervision_stuck,
    supervision_text,
)
from tests.conftest import py_spec
from tests.test_events import subscribed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths
    from aid.protocol import HistoryItem

pytestmark = pytest.mark.anyio

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))
PLUGINS = Path(__file__).parent.parent / "plugins"


@pytest.fixture
async def daemon(paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Paths]:
    """A daemon hosting the bundled plugins, supervision serving before the test starts."""
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[PLUGINS]).serve)
        async with aid.connect(paths) as client:
            with anyio.fail_after(TIMEOUT):
                while True:
                    try:
                        await client.call_plugin("supervision", "goals", session="")
                    except aid.AidError as error:
                        if error.code != "no_plugin":
                            break
                    await anyio.sleep(0.05)
        yield paths
        tg.cancel_scope.cancel()


async def sup(client: aid.Client, caller: str, method: str, **args: JsonValue) -> JsonValue:
    """A supervisor tool's call, as session `caller`."""
    return await client.call(CallPlugin(plugin="supervision", method=method, args=args, caller=caller))


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


def test_loop_score_ignores_cancelled_and_missing_answers() -> None:
    assert loop_score([Output(output=None, stop_reason="cancelled")] * 2) is None
    assert (
        loop_score([Output(output="same", stop_reason="end_turn"), Output(output="same", stop_reason="cancelled")])
        is None
    )
    assert loop_score([Output(output=" ", stop_reason="end_turn")] * 2) is None


def test_loop_score_releases_old_repetition_after_progress() -> None:
    outputs = [
        Output(output=text, stop_reason="end_turn") for text in ["same", "same", "new solution", "verified tests"]
    ]
    assert loop_score(outputs[:2]) == 1.0
    score = loop_score(outputs)
    assert score is not None and score < SUPERVISION_LOOP_THRESHOLD


def test_supervision_text_scores_answers_outside_the_review_tail() -> None:
    page = HistoryPage(
        entries=[_entry(0, ToolCall(tool_call_id="c", status="completed"))], has_older=True, has_newer=False, total=40
    )
    answers = [Output(output="same", stop_reason="end_turn")] * 2
    assert "Loop score: 1.00" in supervision_text("coder", "done", page, [], outputs=answers)


async def test_live_loop_wakes_supervisor_once_and_can_be_interrupted(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:steady"))
            await client.create("other", py_spec(tmp_path, "agents:steady"))
            worker = await client.create(
                "worker", py_spec(tmp_path, "agents:looping").model_copy(update={"supervisor": "boss"})
            )
            reviewed: str | None = None
            completed = 0
            final: Output | None = None
            async for event in worker.stream("loop"):
                if isinstance(event, ToolCall) and event.status == "completed":
                    completed += 1
                    if completed == 4:
                        messages = [
                            entry.item
                            for entry in (await boss.history(limit=1000)).entries
                            if isinstance(entry.item, MessageEntry)
                            and entry.item.text.startswith("Suspected tool loop")
                        ]
                        assert len(messages) == 1
                        match = re.search(r"live turn '([^']+)'", messages[0].text)
                        assert match is not None
                        reviewed = match[1]
                        assert reviewed is not None
                        # Notification did not cancel the worker; only the supervisor's explicit decision does.
                        assert (await worker.status()).busy
                        with pytest.raises(aid.AidError, match="not_supervised"):
                            await sup(client, "other", "interrupt", session="worker", turn=reviewed)
                        with pytest.raises(aid.AidError, match="stale_turn"):
                            await sup(client, "boss", "interrupt", session="worker", turn="old")
                        await sup(client, "boss", "interrupt", session="worker", turn=reviewed)
                elif isinstance(event, Output):
                    final = event
            assert reviewed is not None
            assert final is not None and final.stop_reason == "cancelled"
            assert (await worker.run("continue")).output == "continued"
            with pytest.raises(aid.AidError, match="stale_turn"):
                await sup(client, "boss", "interrupt", session="worker", turn=reviewed)
            messages = [
                entry.item
                for entry in (await boss.history(limit=1000)).entries
                if isinstance(entry.item, MessageEntry) and entry.item.text.startswith("Suspected tool loop")
            ]
            assert len(messages) == 1


async def test_worker_rejects_a_stale_cancel_even_after_daemon_routing(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            worker = await client.create("worker", py_spec(tmp_path, "agents:echo"))
            await worker.run("hi")
            old_turn = (await worker.history()).entries[-1].turn
            async for event in worker.stream("slow"):
                if isinstance(event, TextDelta):
                    with pytest.raises(aid.AidError, match="stale_turn"):
                        await client.call(Cancel(session="worker", turn=old_turn))
                    assert (await worker.status()).busy
                    await worker.cancel()
            assert (await worker.run("hi")).output == "turn 3: echo hi"


async def test_tool_heavy_turns_keep_previous_output_for_scoring(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:steady"))
            worker = await client.create(
                "worker", py_spec(tmp_path, "agents:coder").model_copy(update={"supervisor": "boss"})
            )
            calls: list[list[Any]] = [["ls", {}]] * 12
            await worker.run(json.dumps(calls))
            await worker.run(json.dumps(calls))
            # Replay then follow: no polling or race with the end-of-turn review.
            async for entry in boss.follow_history():
                if isinstance(entry.item, MessageEntry) and "Loop score: 1.00" in entry.item.text:
                    break


async def test_supervisor_agent_itself_decides_to_interrupt_a_live_loop(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:loop_judge"))
            worker = await client.create(
                "worker", py_spec(tmp_path, "agents:looping").model_copy(update={"supervisor": "boss"})
            )
            assert (await worker.run("loop")).stop_reason == "cancelled"
            calls: list[ToolCall] = []
            async for entry in boss.follow_history():
                if isinstance(entry.item, ToolCall):
                    calls.append(entry.item)
                elif isinstance(entry.item, Output):
                    break
            assert any(call.title == "interrupt" for call in calls)
            assert (await worker.run("continue")).output == "continued"


async def test_interruptions_do_not_renew_the_automatic_recovery_budget(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:loop_judge"))
            worker = await client.create(
                "worker",
                py_spec(tmp_path, "agents:looping").model_copy(
                    update={"supervisor": "boss", "supervisor_nudge_budget": 1}
                ),
            )
            assert (await worker.run("loop")).stop_reason == "cancelled"
            await client.call(SendMessage(to="worker", sender="boss", text="try another approach"))
            reviews = 0
            async for entry in boss.follow_history():
                if isinstance(entry.item, MessageEntry) and entry.item.text.startswith("Suspected tool loop"):
                    reviews += 1
                    if reviews == 2:
                        break
            async for status in worker.follow_status():
                if status.attention is not None and not status.busy:
                    break
            with pytest.raises(aid.AidError, match="nudge_budget"):
                await client.call(SendMessage(to="worker", sender="boss", text="try yet another approach"))
            # Explicit user instructions can start fresh; they are not an automatic recovery loop.
            assert (await worker.run("continue")).output == "continued"


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


async def test_a_supervisor_set_later_holds_to_the_links_rules(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:echo"))
            coder = await client.create("coder", py_spec(tmp_path, "agents:echo"))
            with pytest.raises(aid.AidError, match="unknown_supervisor"):
                await coder.set_supervisor("nobody")
            with pytest.raises(aid.AidError, match="self_supervised"):
                await coder.set_supervisor("coder")
            await coder.set_supervisor("boss")
            with pytest.raises(aid.AidError, match="cycle"):
                await boss.set_supervisor("coder")
            listed = {info.name: info.supervisor for info in await client.sessions()}
            supervised = (await coder.status()).supervisor
            await coder.set_supervisor(None)
            ended = (await coder.status()).supervisor
            events = [e.item for e in (await coder.history()).entries if isinstance(e.item, Lifecycle)]
    assert listed == {"boss": None, "coder": "boss"}
    assert (supervised, ended) == ("boss", None)
    assert [(e.event, e.detail) for e in events] == [("supervised", "boss"), ("supervised", "off")]


async def test_a_supervisor_set_later_answers_requests(daemon: Paths, tmp_path: Path) -> None:
    """The plugin follows a session that gains a supervisor while it runs, from that moment on."""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            spec = asking_spec(tmp_path, PermissionMode.SUPERVISOR, timeout=60, absent=60)
            asker = await client.create("coder", spec.model_copy(update={"supervisor": None}))
            await asker.start()
            await asker.set_supervisor("boss")
            with anyio.fail_after(20):
                result = await asker.run(
                    plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
                )
    assert 'The supervisor picked "postgres".' in str(result.output)


async def test_supervision_ops_refuse_strangers(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            with pytest.raises(aid.AidError, match="not_supervised"):
                await sup(client, "stranger", "history", session="coder")
            with pytest.raises(aid.AidError, match="not_supervised"):
                await sup(client, "coder", "goal_add", session="coder", text="self-dealing")
            with pytest.raises(aid.AidError, match="empty_goal"):
                await sup(client, "boss", "goal_add", session="coder", text="  ")
            with pytest.raises(aid.AidError, match="unknown_goal"):
                await sup(client, "boss", "goal_set", session="coder", goal_id="nope", state=GoalState.DONE)
            with pytest.raises(aid.AidError, match="not_supervised"):
                await sup(client, "stranger", "query", session="coder", sql="SELECT 1")
            with pytest.raises(aid.AidError, match="not_supervising"):
                await sup(client, "stranger", "query", sql="SELECT 1")
            with pytest.raises(aid.AidError, match="query_failed"):
                await sup(client, "boss", "query", session="coder", sql="DELETE FROM rulings")
            answer = await sup(client, "boss", "query", session="coder", sql="SELECT count(*) FROM rulings")
            assert QueryResult.model_validate(answer).rows == [[0]]


def read_goals(paths: Paths, name: str) -> list[dict[str, Any]]:
    return json.loads((paths.plugin_dir("supervision") / "goals" / f"{name}.json").read_bytes())


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


def test_permission_review_text_names_input_options_and_what_is_left() -> None:
    request = PermissionRequest(
        request_id="ab12",
        tool_call_id="t",
        tool_name="ask_user",
        title="Which database?",
        input={"argv": ["rm", "-rf", "build"]},
        options=[
            PermissionChoice(option_id="0", name="postgres", kind="ask_once"),
            PermissionChoice(option_id="1", name="sqlite", kind="ask_once", recommended=True),
        ],
    )
    text = permission_review_text("coder", request, final=False)
    assert "coder" in text
    assert "ab12" in text
    assert '"rm"' in text
    assert "- 0: postgres" in text
    assert "- 1: sqlite (recommended)" in text
    assert "answer_permission" in text
    assert "forward_permission" in text
    final = permission_review_text("coder", request, final=True, note="check the target")
    assert "cannot be forwarded" in final
    assert "forward_permission" not in final
    assert "Note: check the target" in final
    assert "query(" not in text
    removal = request.model_copy(update={"tool_name": "remove", "input": {"plan": 7, "files": 3}})
    assert "This removes plan 7. Query it" in permission_review_text("coder", removal, final=False)


def test_stored_supervisor_permissions_become_ask_supervisor_modes() -> None:
    stored = {
        "kind": "native",
        "cwd": "/",
        "target": "agents:echo",
        "permission": "ask",
        "tool_permissions": {"shell": "deny"},
        "supervisor_permissions": ["ask_user", "shell"],
    }
    spec = AgentSpecAdapter.validate_python(stored)
    assert isinstance(spec, NativeSpec)
    assert spec.tool_permissions == {"shell": PermissionMode.DENY, "ask_user": PermissionMode.ASK_SUPERVISOR}
    assert "supervisor_permissions" not in spec.model_dump()


def plan(*calls: tuple[str, dict[str, Any]]) -> str:
    return json.dumps([list(call) for call in calls])


def asking_spec(
    tmp_path: Path,
    mode: PermissionMode = PermissionMode.ASK_SUPERVISOR,
    permission_timeout: float = 30,
    timeout: float = 25,
    absent: float = 2,
) -> NativeSpec:
    """A supervised coder whose `ask_user` questions take `mode`, with short person windows."""
    return py_spec(tmp_path, "agents:coder", PermissionMode.ASK, permission_timeout=permission_timeout).model_copy(
        update={
            "supervisor": "boss",
            "tool_permissions": {"ask_user": mode},
            "supervisor_permission_timeout": timeout,
            "supervisor_permission_absent_timeout": absent,
        }
    )


def permission_wakes(items: list[HistoryItem]) -> list[MessageEntry]:
    """Backstop wakes among history items: firing-shaped, but about a waiting permission."""
    return [item for item in items if isinstance(item, MessageEntry) and "waits on permission" in item.text]


async def permission_wake(session: aid.Session, name: str) -> HistoryEntry:
    """The backstop wake the supervisor got about `name`, once it arrives."""
    while True:
        entries = (await session.history(limit=1000)).entries
        found = [
            e
            for e in entries
            if isinstance(e.item, MessageEntry) and e.item.sender == name and "waits on permission" in e.item.text
        ]
        if found:
            return found[0]
        await anyio.sleep(0.05)


async def rulings(client: aid.Client, sql: str, session: str | None = None) -> QueryResult:
    return QueryResult.model_validate(await client.call_plugin("supervision", "query", sql=sql, session=session))


async def settled_ruling(client: aid.Client) -> QueryResult:
    """The one ruling, once the plugin recorded its end: it follows the decision through history."""
    sql = "SELECT mode, end, allowed, decided_by, decider, option_id FROM rulings WHERE ended IS NOT NULL"
    while True:
        if (result := await rulings(client, sql)).rows:
            return result
        await anyio.sleep(0.05)


async def test_supervisor_answers_after_timeout(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            asker = await client.create("coder", asking_spec(tmp_path))
            result = await asker.run(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            )
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            # The asker's turn ends on the decision; the supervisor's wake turn lands after.
            wake = await permission_wake(boss, "coder")
            await output_after(boss, wake.seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
    assert 'The supervisor picked "postgres".' in str(result.output)
    decisions = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert len(decisions) == 1
    assert decisions[0].by is PermissionDecider.SUPERVISOR
    assert decisions[0].supervisor == "boss"
    assert decisions[0].option_id == "0"
    assert len(permission_wakes(boss_items)) == 1
    assert any("Answered" in out and "with 0" in out for out in completed_outputs(boss_items))


async def test_supervisor_mode_asks_the_supervisor_first(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            # Person windows longer than the test: only a supervisor asked first answers in time.
            asker = await client.create(
                "coder", asking_spec(tmp_path, PermissionMode.SUPERVISOR, timeout=60, absent=60)
            )
            with anyio.fail_after(20):
                result = await asker.run(
                    plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
                )
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            wake = await permission_wake(boss, "coder")
            await output_after(boss, wake.seq)
    assert 'The supervisor picked "postgres".' in str(result.output)
    [request] = [item for item in asker_items if isinstance(item, PermissionRequest)]
    assert request.mode is PermissionMode.SUPERVISOR
    [decision] = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert (decision.by, decision.supervisor) == (PermissionDecider.SUPERVISOR, "boss")


async def test_a_forwarded_request_comes_back_to_the_supervisor(daemon: Paths, tmp_path: Path) -> None:
    """supervisor mode: the unsure supervisor forwards, the person stays silent, the supervisor decides."""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_forwarder"))
            asker = await client.create("coder", asking_spec(tmp_path, PermissionMode.SUPERVISOR, timeout=1, absent=1))
            result = await asker.run(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            )
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            status = await asker.status()
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
            ruling = await settled_ruling(client)
            steps = await rulings(client, "SELECT step, holder, outcome, note FROM ruling_steps ORDER BY step")
    assert 'The supervisor picked "postgres".' in str(result.output)
    wakes = permission_wakes(boss_items)
    assert len(wakes) == 2
    assert "forward_permission hands it" in wakes[0].text
    assert "cannot be forwarded" in wakes[1].text
    assert any(out.startswith("Forwarded") for out in completed_outputs(boss_items))
    [decision] = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert decision.by is PermissionDecider.SUPERVISOR
    assert status.attention is None
    assert ruling.rows == [["supervisor", "decided", None, "supervisor", "boss", "0"]]
    assert steps.rows == [
        [0, "supervisor", "forwarded", "unsure: check the target"],
        [1, "person", "silent", None],
        [2, "supervisor", "decided", None],
    ]


async def test_rules_allow_a_plain_read_before_anyone_is_asked(daemon: Paths, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:coder", PermissionMode.ASK).model_copy(
        update={"supervisor": "boss", "tool_permissions": {"shell": PermissionMode.SUPERVISOR}}
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            coder = await client.create("coder", spec)
            await coder.run(plan(("shell", {"command": "ls -la | head -5"})))
            coder_items = [e.item for e in (await coder.history(limit=1000)).entries]
            ruling = await settled_ruling(client)
            steps = await rulings(client, "SELECT step, holder, outcome FROM ruling_steps")
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
    [decision] = [item for item in coder_items if isinstance(item, PermissionDecision)]
    assert (decision.by, decision.plugin, decision.option_id) == (PermissionDecider.PLUGIN, "supervision", "yes")
    assert ruling.rows == [["supervisor", "decided", True, "plugin", "supervision", "yes"]]
    assert steps.rows == [[0, "rules", "decided"]]
    assert permission_wakes(boss_items) == []


def classified_spec(tmp_path: Path, classifier: str) -> NativeSpec:
    return py_spec(tmp_path, "agents:coder", PermissionMode.ASK).model_copy(
        update={
            "supervisor": "boss",
            "tool_permissions": {"shell": PermissionMode.SUPERVISOR},
            "supervisor_classifier": classifier,
        }
    )


async def classified_run(
    client: aid.Client, tmp_path: Path, classifier: str
) -> tuple[aid.Session, list[HistoryItem], QueryResult]:
    """A supervised coder whose `touch made` the classifier judges first; its history and the ruling's steps."""
    boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
    coder = await client.create("coder", classified_spec(tmp_path, classifier))
    await coder.run(plan(("shell", {"command": "touch made"})))
    await settled_ruling(client)
    steps = await rulings(client, "SELECT step, holder, outcome, note FROM ruling_steps ORDER BY step")
    return boss, [e.item for e in (await coder.history(limit=1000)).entries], steps


async def test_the_classifier_allows_without_the_supervisor(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss, items, steps = await classified_run(client, tmp_path, "agents:classifier_allow")
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
            [classifier] = [s for s in await client.sessions() if s.name == "boss.classifier"]
    [decision] = [item for item in items if isinstance(item, PermissionDecision)]
    assert (decision.by, decision.plugin, decision.option_id) == (PermissionDecider.PLUGIN, "supervision", "yes")
    assert steps.rows == [[0, "classifier", "decided", None]]
    assert permission_wakes(boss_items) == []
    assert classifier.parent == "boss"
    assert (tmp_path / "made").exists()


async def test_a_remove_plan_skips_the_classifier(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "junk").write_text("x")
    spec = classified_spec(tmp_path, "agents:classifier_allow").model_copy(
        update={"tool_permissions": {"remove": PermissionMode.SUPERVISOR}}
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            coder = await client.create("coder", spec)
            await coder.run(plan(("remove", {"paths": ["junk"]})))
            await settled_ruling(client)
            steps = await rulings(client, "SELECT step, holder, outcome FROM ruling_steps ORDER BY step")
    assert steps.rows == [[0, "supervisor", "decided"]]


async def test_the_classifier_denies_with_its_reason(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            _, items, steps = await classified_run(client, tmp_path, "agents:classifier_deny")
    [decision] = [item for item in items if isinstance(item, PermissionDecision)]
    assert (decision.by, decision.option_id, decision.text) == (
        PermissionDecider.PLUGIN,
        "no",
        "write it with the edit tools instead",
    )
    assert steps.rows == [[0, "classifier", "decided", None]]
    outputs = [item for item in items if isinstance(item, Output)]
    assert "aid's supervision check said: write it with the edit tools instead" in str(outputs[-1].output)
    assert not (tmp_path / "made").exists()


async def test_the_classifier_escalates_to_the_supervisor(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss, items, steps = await classified_run(client, tmp_path, "agents:classifier_escalate")
            wake = await permission_wake(boss, "coder")
    [decision] = [item for item in items if isinstance(item, PermissionDecision)]
    assert (decision.by, decision.supervisor) == (PermissionDecider.SUPERVISOR, "boss")
    assert steps.rows == [
        [0, "classifier", "forwarded", "classifier: needs the task to judge"],
        [1, "supervisor", "decided", None],
    ]
    assert isinstance(wake.item, MessageEntry)
    assert "Note: classifier: needs the task to judge" in wake.item.text


async def test_ask_supervisor_hands_back_to_the_person_once(daemon: Paths, tmp_path: Path) -> None:
    """ask_supervisor: the person is silent, the supervisor forwards back, and nothing moves the request again."""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_forwarder"))
            asker = await client.create(
                "coder",
                asking_spec(tmp_path, PermissionMode.ASK_SUPERVISOR, permission_timeout=8, timeout=1, absent=1),
            )
            result = await asker.run(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            )
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
    assert "Nobody answered" in str(result.output)
    [wake] = permission_wakes(boss_items)
    assert "forward_permission hands it" in wake.text
    [decision] = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert decision.by is PermissionDecider.TIMEOUT


async def test_person_beats_supervisor(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            asker = await client.create("coder", asking_spec(tmp_path, timeout=8, absent=2))
            events: list[aid.SessionEvent] = []
            async for event in asker.stream(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            ):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await anyio.sleep(0.5)
                    await asker.answer(event.request_id, "1")
            assert isinstance(events[-1], Output)
            assert str(events[-1].output) == 'The person picked "sqlite".'
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            # Past the absent wait: a backstop the decision did not disarm would have woken the boss.
            await anyio.sleep(4)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
            # Drain the turn-end firing before leaving: its history read must finish, not die in teardown.
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
    decisions = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert len(decisions) == 1
    assert decisions[0].by is PermissionDecider.PERSON
    assert permission_wakes(boss_items) == []


async def test_unlisted_tool_stays_human(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            asker = await client.create("coder", asking_spec(tmp_path, PermissionMode.ASK, permission_timeout=6))
            events: list[aid.SessionEvent] = []
            async for event in asker.stream(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            ):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    with pytest.raises(aid.AidError, match="person_only"):
                        await sup(
                            client,
                            "boss",
                            "answer_permission",
                            session="coder",
                            request_id=event.request_id,
                            option_id="0",
                        )
            assert isinstance(events[-1], Output)
            assert "Nobody answered; proceed with your best judgment." in str(events[-1].output)
            asker_items = [e.item for e in (await asker.history(limit=1000)).entries]
            await anyio.sleep(2)
            boss_items = [e.item for e in (await boss.history(limit=1000)).entries]
            # Drain the turn-end firing before leaving: its history read must finish, not die in teardown.
            firings = await supervised_firings(boss, "coder", 1)
            await output_after(boss, firings[0].seq)
    decisions = [item for item in asker_items if isinstance(item, PermissionDecision)]
    assert len(decisions) == 1
    assert decisions[0].by is PermissionDecider.TIMEOUT
    assert permission_wakes(boss_items) == []


async def test_recent_answer_keeps_full_timeout(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:supervisor_answerer"))
            asker = await client.create("coder", asking_spec(tmp_path, timeout=12, absent=2))
            # Round one: the person answers, leaving a recent person-authored entry.
            async for event in asker.stream(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            ):
                if isinstance(event, PermissionRequest):
                    await asker.answer(event.request_id, "1")
            # Round two: with a person around, the full timeout applies.
            events: list[aid.SessionEvent] = []
            async for event in asker.stream(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            ):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    # Past the absent wait, inside the full one: a person around keeps the long wait.
                    await anyio.sleep(4)
                    during = [e.item for e in (await boss.history(limit=1000)).entries]
                    assert permission_wakes(during) == []
                    await asker.answer(event.request_id, "1")
            # Drain both turn-end firings before leaving: their history reads must finish, not die in teardown.
            firings = await supervised_firings(boss, "coder", 2)
            await output_after(boss, firings[1].seq)
    assert isinstance(events[-1], Output)
    assert str(events[-1].output) == 'The person picked "sqlite".'


async def test_supervised_answer_refuses_strangers_and_gone(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            with pytest.raises(aid.AidError, match="not_supervised"):
                await sup(client, "stranger", "answer_permission", session="coder", request_id="x", option_id=None)
            with pytest.raises(aid.AidError, match="no_pending"):
                await sup(client, "boss", "answer_permission", session="coder", request_id="x", option_id=None)


def test_nudge_repeats() -> None:
    assert not nudge_repeats(None, "anything goes")
    assert nudge_repeats("keep digging", "keep digging")
    assert nudge_repeats("Keep   digging", "keep digging!")
    assert not nudge_repeats("keep digging", "try another angle")


def test_supervision_stuck_counts_resets_and_escalates() -> None:
    assert supervision_stuck("coder", "boss", 0, 0, 3, None) == (0, 0, None)
    assert supervision_stuck("coder", "boss", 4, 1, 3, None) == (4, 1, None)
    assert supervision_stuck("coder", "boss", 4, 1, 3, 0.5) == (0, 0, None)
    assert supervision_stuck("coder", "boss", 0, 0, 3, 1.0) == (1, 0, None)
    assert supervision_stuck("coder", "boss", 2, 2, 3, 1.0) == (3, 2, None)
    assert supervision_stuck("coder", "boss", 2, 3, 3, 1.0) == (
        3,
        3,
        "coder is stuck for 3 turns after 3 nudges from boss; needs a person",
    )
    assert supervision_stuck("coder", "boss", 0, 0, 0, 1.0)[2] is not None


async def all_idle(client: aid.Client, *names: str) -> None:
    """Every named session out of its turn: teardown must not cancel a history read mid-flight."""
    while True:
        for name in names:
            if (await client.session(name).status()).busy:
                break
        else:
            return
        await anyio.sleep(0.05)


async def test_nudge_gates(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            await client.create(
                "tight",
                py_spec(tmp_path, "agents:echo").model_copy(
                    update={"supervisor": "boss", "supervisor_nudge_budget": 1}
                ),
            )
            await client.create("other", py_spec(tmp_path, "agents:echo"))
            await client.call(SendMessage(to="coder", sender="boss", text="keep digging"))
            with pytest.raises(aid.AidError, match="repeated_nudge"):
                await client.call(SendMessage(to="coder", sender="boss", text="keep  digging!"))
            await client.call(SendMessage(to="coder", sender="boss", text="try another angle"))
            await client.call(SendMessage(to="coder", sender=None, text="keep digging"))
            await client.call(SendMessage(to="coder", sender=None, text="keep digging"))
            await client.call(SendMessage(to="other", sender="boss", text="same"))
            await client.call(SendMessage(to="other", sender="boss", text="same"))
            await client.call(SendMessage(to="tight", sender="boss", text="one"))
            with pytest.raises(aid.AidError, match="nudge_budget"):
                await client.call(SendMessage(to="tight", sender="boss", text="two"))
            coder_texts = [
                e.item.text
                for e in (await client.session("coder").history(limit=1000)).entries
                if isinstance(e.item, MessageEntry) and e.item.sender == "boss"
            ]
            await all_idle(client, "boss", "coder", "tight", "other")
            await anyio.sleep(0.5)
    assert coder_texts == ["keep digging", "try another angle"]


async def test_stuck_episode_escalates_to_attention(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with subscribed(daemon, events.supervision_topic("coder")) as sub, aid.connect(daemon) as client:
            boss = await client.create(
                "boss", py_spec(tmp_path, "agents:nagger", AID_SUPERVISED="coder", AID_NUDGES="2")
            )
            coder = await client.create(
                "coder",
                py_spec(tmp_path, "agents:looper").model_copy(
                    update={"supervisor": "boss", "supervisor_nudge_budget": 2}
                ),
            )
            await coder.run("go")
            topic = events.supervision_topic("coder")
            first = await sub.until(topic, lambda p: p.get("firing") and p["firing"]["stuck_turns"] == 0)
            assert first["firing"]["reason"] == "done"
            assert first["firing"]["score"] is None
            first_nudge = await sub.until(topic, lambda p: p.get("nudge") and p["nudge"]["nudges"] == 1)
            assert first_nudge["nudge"]["text"] == "keep digging"
            second = await sub.until(topic, lambda p: p.get("firing") and p["firing"]["stuck_turns"] == 1)
            assert second["firing"]["score"] == 1.0
            second_nudge = await sub.until(topic, lambda p: p.get("nudge") and p["nudge"]["nudges"] == 2)
            assert second_nudge["nudge"]["text"] == "try another angle"
            third = await sub.until(topic, lambda p: p.get("firing") and p["firing"]["stuck_turns"] == 2)
            assert third["firing"]["score"] == 1.0
            status = await coder.status()
            assert status.attention is not None
            assert "coder" in status.attention and "needs a person" in status.attention
            nudges = [
                e.item.text
                for e in (await coder.history(limit=1000)).entries
                if isinstance(e.item, MessageEntry) and e.item.sender == "boss"
            ]
            firings = await supervised_firings(boss, "coder", 3)
            await output_after(boss, firings[2].seq)
    assert status.attention == "coder is stuck for 2 turns after 2 nudges from boss; needs a person"
    assert nudges == ["keep digging", "try another angle"]


async def test_supervision_goals_are_published(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with subscribed(daemon, events.supervision_topic("coder")) as sub, aid.connect(daemon) as client:
            await client.create("boss", py_spec(tmp_path, "agents:echo"))
            await client.create("coder", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            topic = events.supervision_topic("coder")
            goal = await sup(client, "boss", "goal_add", session="coder", text="ship it")
            assert isinstance(goal, dict)
            goal_id = str(goal["id"])
            added = await sub.until(topic, lambda p: p.get("goals") and p["goals"]["action"] == "added")
            assert added["goals"]["goal"]["id"] == goal_id
            await sup(client, "boss", "goal_set", session="coder", goal_id=goal_id, state=GoalState.DONE)
            done = await sub.until(topic, lambda p: p.get("goals") and p["goals"]["action"] == "done")
            assert done["goals"]["goal"]["state"] == "done"


def call_plan(**request: JsonValue) -> str:
    return json.dumps([["control_call", {"request": request}]])


async def test_a_session_calls_supervision_only_as_itself(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            boss = await client.create("boss", py_spec(tmp_path, "agents:control_probe"))
            await client.create("worker", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            await client.create("other", py_spec(tmp_path, "agents:echo"))
            intruder = await client.create("intruder", py_spec(tmp_path, "agents:control_probe"))
            add: dict[str, JsonValue] = {"op": "call_plugin", "plugin": "supervision", "method": "goal_add"}
            authorized = await boss.run(call_plan(**add, args={"session": "worker", "text": "authorized"}))
            unsupervised = await boss.run(call_plan(**add, args={"session": "other", "text": "unauthorized"}))
            spoofed = await intruder.run(call_plan(**add, caller="boss", args={"session": "worker", "text": "spoofed"}))
            goals = await sup(client, "boss", "goals", session="worker")
    assert json.loads(str(authorized.output))["text"] == "authorized"
    assert json.loads(str(unsupervised.output)) == {"error": "not_supervised"}
    assert json.loads(str(spoofed.output)) == {"error": "not_supervised"}
    assert isinstance(goals, list)
    assert [goal["text"] for goal in goals if isinstance(goal, dict)] == ["authorized"]
