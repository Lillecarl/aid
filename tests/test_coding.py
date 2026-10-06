from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

import aid
from aid.backends.permissions import PermissionWaits
from aid.backends.pydantic_ai import PydanticAIBackend
from aid.coding import CODING, SPILL_LIMIT, Coding, compact, spill
from aid.protocol import Output, PermissionDecider, PermissionDecision, PermissionRequest, Started
from aid.spec import PermissionMode
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 60


def plan(*calls: tuple[str, dict[str, Any]]) -> str:
    return json.dumps([list(call) for call in calls])


async def test_edits_stage_until_applied(daemon: Paths, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "app.py").write_text('def greet():\n    print("Hi")\n')
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(work, "agents:coder", PermissionMode.ALLOW))
            staged = await session.run(
                plan(
                    ("read", {"path": "app.py"}),
                    ("edit", {"path": "app.py", "old": 'print("Hi")', "new": 'print("Hello")'}),
                    ("read", {"path": "app.py"}),
                    ("edit", {"path": "app.py", "old": "absent", "new": "x"}),
                    ("read", {"path": "../outside.txt"}),
                    ("outline", {"path": "app.py"}),
                )
            )
            on_disk_before = (work / "app.py").read_text()
            applied = await session.run(plan(("show_edits", {}), ("apply_edits", {})))
    before, edited, after, missing, outside, outline = str(staged.output).split("\n=====\n")
    assert '     2\t    print("Hi")' in before
    assert '+    print("Hello")' in edited
    assert "--- a/app.py" in edited
    assert '     2\t    print("Hello")' in after
    assert "'absent' is not in app.py" in missing
    assert "outside the session's directory" in outside
    assert "function_definition\tgreet" in outline
    assert on_disk_before == 'def greet():\n    print("Hi")\n'
    diff, wrote = str(applied.output).split("\n=====\n")
    assert '+    print("Hello")' in diff
    assert wrote == "wrote app.py"
    assert (work / "app.py").read_text() == 'def greet():\n    print("Hello")\n'


async def test_ls_lists_directories_first(daemon: Paths, tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "pkg").mkdir(parents=True)
    (work / "pkg" / "b.py").write_text("B = 1\n")
    (work / "a.py").write_text("A = 1\n")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(work, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    ("ls", {}),
                    ("ls", {"path": "missing"}),
                    ("ls", {"path": "pkg"}),
                    ("ls", {"path": "a.py"}),
                )
            )
    root, no_such, pkg, not_dir = str(result.output).split("\n=====\n")
    assert root == "pkg/\na.py"
    assert "missing does not exist" in no_such
    assert pkg == "b.py"
    assert "a.py is a file; read it with read" in not_dir


async def test_spill_writes_big_results_to_stable_files(tmp_path: Path) -> None:
    coding = Coding(
        cwd=tmp_path,
        store=tmp_path / "runs",
        outputs=tmp_path / "outputs",
        mode=PermissionMode.ALLOW,
        timeout=60,
        waits=PermissionWaits(),
    )
    token = CODING.set(coding)
    try:
        big = "x" * (SPILL_LIMIT + 100)
        digest = hashlib.sha256(big.encode()).hexdigest()[:16]
        expected = tmp_path / "outputs" / f"{digest}.txt"
        first = await spill(big)
        assert first.startswith(big[:SPILL_LIMIT])
        assert len(first) <= SPILL_LIMIT + 500
        assert str(expected) in first
        assert expected.read_text() == big
        assert await spill(big) == first
        assert await spill("small") == "small"
        assert coding.resolve(str(expected)) == expected
    finally:
        CODING.reset(token)


async def test_compact_replaces_past_with_digest(tmp_path: Path) -> None:
    agent = Agent(TestModel(custom_output_text="billing decisions kept"))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart(content="did one")]),
        ModelRequest(parts=[UserPromptPart(content="second")]),
        ModelResponse(parts=[TextPart(content="did two")]),
    ]
    coding = Coding(
        cwd=tmp_path,
        store=tmp_path / "runs",
        outputs=tmp_path / "outputs",
        mode=PermissionMode.ALLOW,
        timeout=60,
        waits=PermissionWaits(),
    )
    PydanticAIBackend(  # constructed for its side effect: wiring compact into coding
        agent, anyio.Path(tmp_path / "history.json"), history, [], Started(pid=1, agent="t", model="m"), coding
    )
    token = CODING.set(coding)
    try:
        assert await compact("the billing work") == "billing decisions kept"
    finally:
        CODING.reset(token)
    stored = json.loads((tmp_path / "history.json").read_bytes())
    first, *tail = stored
    assert first["parts"][0]["content"].startswith("[Summary of earlier work]\nbilling decisions kept")
    assert len(tail) == 2
    assert tail[0]["parts"][0]["content"] == "second"


async def test_compact_leaves_short_history(tmp_path: Path) -> None:
    agent = Agent(TestModel(custom_output_text="unused"))
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="only")])]
    coding = Coding(
        cwd=tmp_path,
        store=tmp_path / "runs",
        outputs=tmp_path / "outputs",
        mode=PermissionMode.ALLOW,
        timeout=60,
        waits=PermissionWaits(),
    )
    PydanticAIBackend(  # constructed for its side effect: wiring compact into coding
        agent, anyio.Path(tmp_path / "history.json"), history, [], Started(pid=1, agent="t", model="m"), coding
    )
    token = CODING.set(coding)
    try:
        assert await compact("anything") == "history is short; nothing compacted"
    finally:
        CODING.reset(token)
    assert not (tmp_path / "history.json").exists()


async def test_outline_names_a_directory(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("outline", {"path": "pkg"})))
    assert "pkg is a directory; outline reads a single file" in str(result.output)


async def test_apply_writes_only_what_changed(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")
    read_only = (tmp_path / "a.py").stat().st_mtime_ns
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    ("read", {"path": "a.py"}),
                    ("show_edits", {}),
                    ("edit", {"path": "b.py", "old": "B = 1", "new": "B = 2"}),
                    ("apply_edits", {}),
                )
            )
    _read, shown, _edited, applied = str(result.output).split("\n=====\n")
    assert shown == "nothing is staged"
    assert applied == "wrote b.py"
    assert (tmp_path / "a.py").stat().st_mtime_ns == read_only
    assert (tmp_path / "b.py").read_text() == "B = 2\n"


async def test_python_asks_for_each_program(daemon: Paths, tmp_path: Path) -> None:
    script = 'a = await run("echo", "one")\nb = await run("echo", "two")\nprint(a.text + b.text, end="")\n'
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(plan(("python", {"script": script}))):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "always")
    [request] = [e for e in events if isinstance(e, PermissionRequest)]
    assert (request.tool_name, request.title, request.input) == (
        "python",
        "echo one",
        {"argv": ["echo", "one"], "cwd": "."},
    )
    assert [o.option_id for o in request.options] == ["yes", "always", "no"]
    assert PermissionDecision(request_id=request.request_id, option_id="always", by=PermissionDecider.PERSON) in events
    last = events[-1]
    assert isinstance(last, Output)
    report = str(last.output)
    assert "$ echo one  (exit 0, " in report
    assert "$ echo two  (exit 0, " in report
    assert "printed:\n  one\n  two" in report


async def test_deny_mode_refuses_commands_and_writes(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            result = await session.run(
                plan(
                    ("python", {"script": 'await run("touch", "made")\n'}),
                    ("write", {"path": "new.txt", "content": "text\n"}),
                    ("apply_edits", {}),
                )
            )
    ran, _written, applied = str(result.output).split("\n=====\n")
    assert "Denied: not allowed to run touch made" in ran
    assert applied == "not allowed: the edits stay staged"
    assert not (tmp_path / "made").exists()
    assert not (tmp_path / "new.txt").exists()


async def test_a_reports_records_are_readable(daemon: Paths, tmp_path: Path) -> None:
    script = 'await run("seq", "100")\n'
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            first = await session.run(plan(("python", {"script": script})))
            record = str(first.output).split("read ", 1)[1].split()[0]
            second = await session.run(plan(("read", {"path": record, "offset": 50, "limit": 2})))
    assert record.endswith(".1/stdout")
    assert str(second.output).startswith("    50\t50\n    51\t51\n… 49 more lines")


async def test_ask_user_returns_the_persons_pick_and_words(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(
                plan(("ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}))
            ):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "1", "it must embed")
            history = [e.item for e in (await session.history()).entries]
    [request] = [e for e in events if isinstance(e, PermissionRequest)]
    assert (request.tool_name, request.title, [o.name for o in request.options]) == (
        "ask_user",
        "Which database?",
        ["postgres", "sqlite"],
    )
    assert (
        PermissionDecision(
            request_id=request.request_id, option_id="1", by=PermissionDecider.PERSON, text="it must embed"
        )
        in events
    )
    assert [request, next(e for e in events if isinstance(e, PermissionDecision))] == [
        e for e in history if isinstance(e, PermissionRequest | PermissionDecision)
    ]
    last = events[-1]
    assert isinstance(last, Output)
    assert str(last.output) == 'The person picked "sqlite" and added: it must embed'


async def test_ask_user_takes_words_alone(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(plan(("ask_user", {"question": "What is the password?", "options": []}))):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    assert event.options == []
                    await session.answer(event.request_id, None, "hunter2")
    last = events[-1]
    assert isinstance(last, Output)
    assert str(last.output) == "The person answered: hunter2"


async def test_ask_user_without_a_watcher_proceeds(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("ask_user", {"question": "Which database?", "options": ["postgres"]})))
            history = [e.item for e in (await session.history()).entries]
    assert str(result.output) == "No one is watching; proceed with your best judgment."
    assert not [e for e in history if isinstance(e, PermissionRequest | PermissionDecision)]


async def test_ask_user_declined_in_deny_mode(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            result = await session.run(plan(("ask_user", {"question": "Which database?", "options": ["postgres"]})))
    assert "declines questions" in str(result.output)


async def test_ask_user_unanswered_in_time(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create(
                "coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK, permission_timeout=0.2)
            )
            result = await session.run(plan(("ask_user", {"question": "Which database?", "options": ["postgres"]})))
            history = [e.item for e in (await session.history()).entries]
    assert str(result.output) == "Nobody answered; proceed with your best judgment."
    decision = next(e for e in history if isinstance(e, PermissionDecision))
    assert (decision.option_id, decision.by) == (None, PermissionDecider.TIMEOUT)


async def test_ask_user_autoselects_recommended(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create(
                "coder",
                py_spec(tmp_path, "agents:coder", PermissionMode.ASK, ask_autoselect_after=0.2),
            )
            events: list[aid.SessionEvent] = []
            async for event in session.stream(
                plan(
                    (
                        "ask_user",
                        {
                            "question": "Which database?",
                            "options": ["postgres", "sqlite"],
                            "recommended": "sqlite",
                        },
                    )
                )
            ):
                events.append(event)
    [request] = [e for e in events if isinstance(e, PermissionRequest)]
    assert [o.recommended for o in request.options] == [False, True]
    assert PermissionDecision(request_id=request.request_id, option_id="1", by=PermissionDecider.AUTO) in events
    last = events[-1]
    assert isinstance(last, Output)
    assert str(last.output) == 'Nobody answered in time; proceeding with the recommended option "sqlite".'


async def test_ask_user_person_beats_autoselect(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(
                plan(
                    (
                        "ask_user",
                        {
                            "question": "Which database?",
                            "options": ["postgres", "sqlite"],
                            "recommended": "sqlite",
                        },
                    )
                )
            ):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "0")
    last = events[-1]
    assert isinstance(last, Output)
    assert str(last.output) == 'The person picked "postgres".'


async def test_ask_user_rejects_unknown_recommended(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    (
                        "ask_user",
                        {"question": "Which database?", "options": ["postgres"], "recommended": "mysql"},
                    )
                )
            )
    assert "is not one of the options" in str(result.output)


async def test_background_outlives_its_turn(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            started = await session.run(plan(("background", {"argv": ["sleep", "30"]})))
            assert str(started.output).startswith("started bg1: sleep 30 (pid ")
            # A new turn: the task is still there, running.
            running = await session.run(plan(("task_output", {"task_id": "bg1"})))
            assert "bg1 is running" in str(running.output)
            stopped = await session.run(plan(("task_stop", {"task_id": "bg1"})))
            assert "bg1 stopped" in str(stopped.output)
            listed = await session.run(plan(("tasks", {})))
    assert "bg1: sleep 30 — stopped" in str(listed.output)


async def test_background_output_returns_like_a_tool_call(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    ("background", {"argv": ["echo", "hello"]}),
                    ("task_output", {"task_id": "bg1", "wait": 5}),
                )
            )
    begun, report = str(result.output).split("\n=====\n")
    assert begun.startswith("started bg1: echo hello (pid ")
    assert report.startswith("bg1 finished: exit 0 after ")
    assert "\nhello" in report


async def test_background_timeout_kills(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    ("background", {"argv": ["sleep", "30"], "time_limit": 0.2}),
                    ("task_output", {"task_id": "bg1", "wait": 5}),
                )
            )
    _begun, report = str(result.output).split("\n=====\n")
    assert report.startswith("bg1 timed out after 0.2s and was killed: sleep 30")


async def test_background_asks_first(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(plan(("background", {"argv": ["echo", "hello"]}))):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "yes")
            await session.run(plan(("task_stop", {"task_id": "bg1"})))
    [request] = [e for e in events if isinstance(e, PermissionRequest)]
    assert (request.tool_name, request.title, [o.option_id for o in request.options]) == (
        "background",
        "echo hello",
        ["yes", "always", "no"],
    )


async def test_background_deny_refuses(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            result = await session.run(plan(("background", {"argv": ["echo", "hello"]})))
    assert str(result.output) == "not allowed: the task stays unstarted"


async def test_background_unknown_task(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("task_output", {"task_id": "bg9"})))
    assert "no background task 'bg9'" in str(result.output)


async def test_background_dies_with_the_worker(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            started = await session.run(plan(("background", {"argv": ["sleep", "120"]})))
            pid = int(str(started.output).rsplit("(pid ", 1)[1].rstrip(")"))
            await client.session("coder").delete()
            # A reaped child leaves /proc; an orphaned one stays. Poll for gone.
            while await anyio.Path(f"/proc/{pid}").exists():  # noqa: ASYNC110 -- no event fires when a foreign pid dies
                await anyio.sleep(0.2)


async def test_python_in_loop_runs_in_the_worker(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("python", {"script": 'print("hi")', "mode": "in-loop"})))
    assert str(result.output).startswith("in-loop finished after ")
    assert str(result.output).endswith("hi\n")


async def test_python_in_loop_sees_aid_itself(daemon: Paths, tmp_path: Path) -> None:
    # The trust model in one assertion: in-loop code shares the worker's imports, so no gate can contain
    # it. Whole-script permit is review, not containment.
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(("python", {"script": "import aid.coding\nprint(aid.coding.TASK_POLL)", "mode": "in-loop"}))
            )
    assert "\n0.2\n" in str(result.output)


async def test_python_in_loop_timeout(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(
                    (
                        "python",
                        {
                            "script": "import asyncio\nawait asyncio.sleep(30)",
                            "mode": "in-loop",
                            "time_limit": 0.2,
                        },
                    )
                )
            )
    assert str(result.output).startswith("in-loop timed out after 0.2s")


async def test_python_in_loop_failure_reports(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("python", {"script": "1/0", "mode": "in-loop"})))
    output = str(result.output)
    assert output.startswith("in-loop failed after ")
    assert "ZeroDivisionError" in output


async def test_python_in_loop_asks_for_the_whole_script(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            events: list[aid.SessionEvent] = []
            async for event in session.stream(plan(("python", {"script": 'print("hi")', "mode": "in-loop"}))):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    await session.answer(event.request_id, "yes")
    [request] = [e for e in events if isinstance(e, PermissionRequest)]
    assert (request.tool_name, request.title, request.input) == (
        "python",
        'in-loop: print("hi")',
        {"mode": "in-loop", "script": 'print("hi")'},
    )
    last = events[-1]
    assert isinstance(last, Output)
    assert str(last.output).endswith("hi\n")


async def test_python_in_loop_deny_refuses(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            result = await session.run(plan(("python", {"script": 'print("hi")', "mode": "in-loop"})))
    assert str(result.output) == "not allowed: the script stays unrun"


async def test_python_sync_runs_isolated(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("python", {"script": 'print("hi")', "mode": "sync"})))
    output = str(result.output)
    assert output.startswith("sync finished: exit 0 after ")
    assert output.endswith("hi\n")


async def test_python_sync_nonzero_reports(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(plan(("python", {"script": "raise SystemExit(3)", "mode": "sync"})))
    assert str(result.output).startswith("sync finished: exit 3 after ")


async def test_python_xonsh_pipes_and_pythons(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            result = await session.run(
                plan(("python", {"script": "echo hello | grep hello\necho @(40+2)", "mode": "xonsh"}))
            )
    output = str(result.output)
    assert output.startswith("xonsh finished: exit 0 after ")
    assert output.endswith("42\n")


async def test_shell_persists_state_across_turns_and_restarts(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            made = await session.run(plan(("shell", {"command": "cd sub && pwd && export AID_E2E=1"})))
            kept = await session.run(plan(("shell", {"command": 'echo "in $(basename $(pwd)):$AID_E2E"'})))
            restarted = await session.run(plan(("shell_restart", {}), ("shell", {"command": "pwd"})))
    assert "shell finished: exit 0 after " in str(made.output)
    assert str(tmp_path / "sub") in str(made.output)
    assert str(kept.output).endswith("in sub:1\n")
    reset, back = str(restarted.output).split("\n=====\n")
    assert reset == "shell restarted; cd and env reset"
    assert back.endswith(f"{tmp_path}\n")


async def test_shell_deny_refuses(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            result = await session.run(plan(("shell", {"command": "touch made"})))
    assert str(result.output) == "not allowed: the command stays unrun"
    assert not (tmp_path / "made").exists()
