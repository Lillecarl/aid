from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.backends.claude_tty import Screen, classify, claude_argv, launcher_script
from aid.mcp import claude_config
from aid.protocol import AidError, Output, TextDelta, ToolCall
from aid.spec import ClaudeTtySpec, McpHttp
from tests.fake_claude import COUNT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aid.paths import Paths

TESTS = Path(__file__).parent
TIMEOUT = 30

needs_pymux = pytest.mark.skipif(shutil.which("pymux") is None, reason="pymux is not on PATH")


@pytest.mark.parametrize(
    ("capture", "screen"),
    [
        ("", Screen.STARTING),
        (" Do you trust this folder?\n ❯ No, exit\n   Yes, I trust this folder\n", Screen.TRUST),
        ('header\n────\n❯ Try "refactor <filepath>"\n────\n', Screen.READY),
        ("header\n────\n❯\n────\n", Screen.READY),
        ("loading ❯ not at the start\n", Screen.STARTING),
    ],
)
def test_classify(capture: str, screen: Screen) -> None:
    assert classify(capture) is screen


def test_launcher_script_quotes_everything() -> None:
    script = launcher_script({"A": "x y", "B": "it's"}, "/w d", ["claude", "--session-id", "id;rm"])
    assert script.splitlines() == [
        "#!/bin/sh",
        "cd '/w d' || exit 1",
        "exec env -i 'A=x y' 'B=it'\"'\"'s' claude --session-id 'id;rm'",
    ]


def test_variadic_options_go_before_other_arguments() -> None:
    spec = ClaudeTtySpec(cwd="/", args=["first prompt"])
    assert claude_argv(spec, "id", resume=False, mcp_config="/s/mcp.json") == [
        "claude",
        "--mcp-config",
        "/s/mcp.json",
        "--allowedTools",
        "mcp__aid",
        "--session-id",
        "id",
        "first prompt",
    ]
    without = spec.model_copy(update={"aid_tools": False})
    assert claude_argv(without, "id", resume=True, mcp_config=None) == ["claude", "--resume", "id", "first prompt"]


@pytest.fixture
async def pymux_socket() -> AsyncIterator[str]:
    directory = Path(tempfile.mkdtemp(prefix="aid-pymux-"))
    socket = str(directory / "pymux.sock")
    yield socket
    await anyio.run_process(["pymux", "-S", socket, "kill-server"], check=False)
    shutil.rmtree(directory, ignore_errors=True)


def fake_spec(cwd: Path, socket: str, env: dict[str, str] | None = None, *, trust_cwd: bool = False) -> ClaudeTtySpec:
    return ClaudeTtySpec(
        cwd=str(cwd),
        command=[sys.executable, str(TESTS / "fake_claude.py")],
        env={"CLAUDE_CONFIG_DIR": str(cwd / "claude-config"), **(env or {})},
        pymux_socket=socket,
        trust_cwd=trust_cwd,
    )


pytestmark = pytest.mark.anyio


@needs_pymux
async def test_prompt_round_trip(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            plain = await session.run("hello")
            multi = await session.run("one\ntwo; three")
            counted = await session.run("count")
            events = [e async for e in session.stream("tool")]
    assert plain.output == "echo: hello"
    assert plain.stop_reason == "end_turn"
    assert multi.output == "echo: one\ntwo; three"
    assert counted.text == "".join(f"{i} " for i in range(COUNT))
    assert events == [
        ToolCall(tool_call_id="toolu_fake", title="Bash", status="in_progress"),
        ToolCall(tool_call_id="toolu_fake", status="completed"),
        TextDelta(text="ran it"),
        Output(output="ran it", stop_reason="end_turn"),
    ]


@needs_pymux
async def test_cancel_interrupts_the_turn(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("slow"):
                events.append(event)
                if event == TextDelta(text="waiting"):
                    await session.cancel()
            after = await session.run("again")
    assert events[-1] == Output(output="waiting", stop_reason="cancelled")
    assert after.output == "echo: again"


@needs_pymux
async def test_stop_resumes_the_same_transcript(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            await session.run("first")
            await session.stop()
            second = await session.run("second")
    transcripts = list((tmp_path / "claude-config" / "projects").glob("*/*.jsonl"))
    assert len(transcripts) == 1
    assert second.output == "echo: second"


@needs_pymux
async def test_trust_dialog(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    env = {"FAKE_CLAUDE_TRUST": "1"}
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(AidError):
                await client.create("untrusted", fake_spec(tmp_path, pymux_socket, env=env))
            session = await client.create("trusted", fake_spec(tmp_path, pymux_socket, env=env, trust_cwd=True))
            result = await session.run("hi")
    assert result.output == "echo: hi"


@needs_pymux
async def test_mcp_servers_reach_claude(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    servers = [McpHttp(name="web", url="http://127.0.0.1:1/mcp", headers={"Authorization": "Bearer t"})]
    spec = fake_spec(tmp_path, pymux_socket).model_copy(update={"mcp_servers": servers})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", spec)
            result = await session.run("mcp")
    config = next(daemon.state_dir.rglob("mcp.json"))
    got = json.loads(result.text)["mcpServers"]
    builtin = got.pop("aid")
    assert got == claude_config(servers)["mcpServers"]
    assert (builtin["command"], builtin["args"]) == (sys.executable, ["-m", "aid.mcp_server"])
    assert builtin["env"]["AID_SESSION"] == "tty"
    assert config.stat().st_mode & 0o777 == 0o600
