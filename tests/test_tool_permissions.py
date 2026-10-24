from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.protocol import Output, PermissionRequest, ToolMode
from aid.spec import ClaudeSdkSpec, PermissionMode
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
