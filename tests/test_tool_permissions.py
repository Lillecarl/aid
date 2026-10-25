from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.protocol import Output, PermissionRequest, Started, ToolMode
from aid.spec import ClaudeSdkSpec, NativeSpec, PermissionMode
from tests.conftest import acp_spec, py_spec
from tests.test_claude_sdk import FAKE_CLAUDE
from tests.test_coding import plan

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def by_name(tools: list[ToolMode]) -> dict[str, ToolMode]:
    return {tool.name: tool for tool in tools}


async def test_a_started_session_lists_its_tools_with_their_modes(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            await session.start()
            running = by_name((await session.status()).tools)
            await session.stop()
            stopped = by_name((await session.status()).tools)
    assert running == stopped
    assert (running["ls"].group, running["ls"].asks, running["ls"].mode, running["ls"].set) == (
        "coding",
        False,
        PermissionMode.ALLOW,
        False,
    )
    assert (running["python"].asks, running["python"].mode) == (True, PermissionMode.ASK)
    assert running["send_message"].group == "aid"


async def test_a_change_reaches_the_running_worker_and_the_spec(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n")
    spec_file = daemon.state_dir / "sessions" / "coder" / "spec.json"
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW))
            listed = await session.run(plan(("ls", {})))
            await session.set_permissions({"ls": PermissionMode.DENY})
            refused = await session.run(plan(("ls", {})))
            status = await session.status()
            saved = json.loads(spec_file.read_text())["tool_permissions"]
            await session.set_permissions({"ls": None})
            restored = by_name((await session.status()).tools)["ls"]
    assert "a.txt" in str(listed.output)
    assert str(refused.output) == "not allowed: ls stays unrun"
    assert (by_name(status.tools)["ls"].mode, by_name(status.tools)["ls"].set) == (PermissionMode.DENY, True)
    assert saved == {"ls": "deny"}
    assert json.loads(spec_file.read_text())["tool_permissions"] == {}
    assert (restored.mode, restored.set) == (PermissionMode.ALLOW, False)


async def test_ask_gates_a_tool_once_and_never_twice(daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("contents\n")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            await session.set_permissions({"read": PermissionMode.ASK, "python": PermissionMode.ALLOW})
            requests: list[PermissionRequest] = []
            output = None
            steps = plan(("read", {"path": "a.txt"}), ("python", {"script": 'print((await run("echo", "hi")).text)'}))
            async for event in session.stream(steps):
                if isinstance(event, PermissionRequest):
                    requests.append(event)
                    await session.answer(event.request_id, "yes")
                elif isinstance(event, Output):
                    output = str(event.output)
    [request] = requests
    assert (request.tool_name, request.input) == ("read", {"path": "a.txt"})
    assert request.title == f"read {json.dumps(request.input)}"
    assert output is not None and "contents" in output and "hi" in output


async def test_the_default_applies_to_the_tools_that_ask(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder"))
            await session.start()
            await session.set_permissions(default=PermissionMode.ALLOW)
            ran = await session.run(plan(("python", {"script": 'await run("touch", "made")'})))
            status = await session.status()
    assert (tmp_path / "made").exists(), ran.output
    assert status.permission is PermissionMode.ALLOW
    assert by_name(status.tools)["python"].mode is PermissionMode.ALLOW


async def test_acp_learns_tools_from_requests_and_answers_each_by_its_mode(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            before = (await session.status()).tools
            # The fake asks about Bash, and about aid's own `add`, as claude-agent-acp names them.
            refused = await session.run("spoofed-title")
            ours = await session.run("aid-tool")
            learned = by_name((await session.status()).tools)
            await session.set_permissions({"Bash": PermissionMode.ALLOW})
            allowed = await session.run("spoofed-title")
            await session.set_permissions({"mcp__aid__add": PermissionMode.DENY})
            denied = await session.run("aid-tool")
    assert before == []
    assert (refused.output, ours.output, allowed.output, denied.output) == ("no", "yes", "yes", "no")
    assert (learned["Bash"].group, learned["Bash"].mode, learned["Bash"].set) == ("agent", PermissionMode.DENY, False)
    assert (learned["mcp__aid__add"].group, learned["mcp__aid__add"].mode) == ("aid", PermissionMode.ALLOW)


async def test_claude_sdk_reports_its_tools_after_a_prompt(daemon: Paths, tmp_path: Path) -> None:
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    logs = {"FAKE_CLAUDE_ARGV": str(tmp_path / "argv.log"), "FAKE_CLAUDE_STDIN": str(tmp_path / "stdin.log")}
    spec = ClaudeSdkSpec(cwd=str(tmp_path), model="opus", command=[str(fake)], env=logs)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("sdk", spec)
            await session.start()
            before = (await session.status()).tools
            await session.run("hello")
            await session.set_permissions({"Bash": PermissionMode.DENY})
            after = by_name((await session.status()).tools)
    assert before == []
    assert (after["Bash"].group, after["Bash"].mode, after["Bash"].set) == ("claude", PermissionMode.DENY, True)
    assert (after["Read"].mode, after["Read"].set) == (None, False)
    assert after["mcp__aid__send_message"].group == "aid"


async def test_a_python_script_asks_before_touching_files_outside_its_directory(daemon: Paths, tmp_path: Path) -> None:
    work, outside = tmp_path / "work", tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    script = f"""
Path("here.txt").write_text("quiet")
try:
    Path({str(outside / "there.txt")!r}).write_text("loud")
except PermissionError as error:
    print("refused")
"""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(work, "agents:coder", PermissionMode.ASK))
            requests: list[PermissionRequest] = []
            output = ""
            async for event in session.stream(plan(("python", {"script": script}))):
                if isinstance(event, PermissionRequest):
                    requests.append(event)
                    await session.answer(event.request_id, "no")
                elif isinstance(event, Output):
                    output = str(event.output)
            tools = by_name((await session.status()).tools)
    [request] = requests
    assert (request.tool_name, request.title) == ("python:write", f"write {outside / 'there.txt'}")
    assert (work / "here.txt").read_text() == "quiet"
    assert not (outside / "there.txt").exists()
    assert "refused" in output
    assert (tools["python:read"].mode, tools["python:write"].group) == (PermissionMode.ASK, "coding")


async def test_a_question_left_open_ends_with_the_scripts_time_limit(daemon: Paths, tmp_path: Path) -> None:
    script = f"Path({str(tmp_path.parent / 'elsewhere.txt')!r}).write_text('x')\n"
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("coder", py_spec(tmp_path, "agents:coder", PermissionMode.ASK))
            result = await session.run(plan(("python", {"script": script, "time_limit": 1})))
            status = await session.status()
    assert "time limit" in str(result.output)
    assert status.permissions == []


async def test_a_tool_ceiling_offers_only_the_tools_it_matches(daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#50: a tool whose whole name no pattern matches is never offered, aid's own tools included,
    and a call to one runs nothing, even under `allow`."""
    (tmp_path / "a.txt").write_text("a\n")
    open_spec = py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW)
    spec = NativeSpec.model_validate({**open_spec.model_dump(), "tools": ["ls", "rea[d]"]})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("reader", spec)
            result = await session.run(plan(("shell", {"command": "touch made"}), ("ls", {})))
            offered = sorted(by_name((await session.status()).tools))
    assert offered == ["ls", "read"]
    assert not (tmp_path / "made").exists()
    assert "a.txt" in str(result.output)


SCRIPTED_PROVIDER = '''\
import aid
import agents


class Scripted(aid.ModelProvider):
    """The test coder's planned model under a provider name."""

    async def transport(self, model: str) -> "_Planned":
        return _Planned()


class _Planned:
    model_name = "scripted/planned"

    def stream(self, messages, info):
        return agents.coder.model.stream(messages, info)

    async def request(self, messages, info):
        return await agents.coder.model.request(messages, info)

    async def aclose(self) -> None:
        pass
'''


async def test_a_spec_with_instructions_is_the_agent(daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#50 phase 2: no agent definition; the spec's instructions on its model, offered every named
    tool its ceiling matches and nothing else."""
    plugins = tmp_path / "plugins"
    (plugins / "scripted").mkdir(parents=True)
    (plugins / "scripted" / "__init__.py").write_text(SCRIPTED_PROVIDER)
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("a\n")
    base = py_spec(work, "agents:coder", PermissionMode.ALLOW, AID_PLUGINS_PATH=str(plugins))
    spec = NativeSpec.model_validate(
        {
            **base.model_dump(exclude={"target"}),
            "instructions": "You only list files.",
            "model": "scripted/planned",
            "tools": ["ls", "send_message", "goals"],
        }
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("lister", spec)
            result = await session.run(plan(("shell", {"command": "touch made"}), ("ls", {})))
            status = await session.status()
            history = [entry.item for entry in (await session.history()).entries]
    assert sorted(by_name(status.tools)) == ["goals", "ls", "send_message"]
    assert status.runs == "instructions"
    assert not (work / "made").exists()
    assert "a.txt" in str(result.output)
    started = next(item for item in history if isinstance(item, Started))
    assert (started.agent, started.model) == ("instructions", "scripted/planned")


def test_a_spec_with_instructions_names_its_model_and_tools() -> None:
    NativeSpec(cwd="/", instructions="x", model="a/b", tools=[])
    for partial in ({"model": "a/b"}, {"tools": ["ls"]}):
        with pytest.raises(ValidationError, match="sets model and tools"):
            NativeSpec.model_validate({"cwd": "/", "instructions": "x", **partial})
    with pytest.raises(ValidationError, match="exactly one of agent, target and instructions"):
        NativeSpec(cwd="/", agent="coder", instructions="x", model="a/b", tools=[])


def test_a_tool_pattern_matches_the_whole_name() -> None:
    spec = NativeSpec(cwd="/", target="m:a", tools=["read", "task_.*"])
    assert [spec.offers(name) for name in ("read", "read_new", "task_stop", "shell")] == [True, False, True, False]
    assert NativeSpec(cwd="/", target="m:a").offers("shell")
    with pytest.raises(ValidationError, match="invalid tool pattern"):
        NativeSpec(cwd="/", target="m:a", tools=["("])
