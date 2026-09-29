from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest

import aid
from aid.protocol import Output, PermissionDecider, PermissionDecision, PermissionRequest
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
