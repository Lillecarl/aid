from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

# The parsing rules under test are private to the CLI.
from aid.cli import (
    _parser,  # pyright: ignore[reportPrivateUsage]
    _preload,  # pyright: ignore[reportPrivateUsage]
    _spec,  # pyright: ignore[reportPrivateUsage]
    _split_agent_command,  # pyright: ignore[reportPrivateUsage]
    _tool_line,  # pyright: ignore[reportPrivateUsage]
)
from aid.launcher import PRELOAD
from aid.protocol import ToolCall
from aid.spec import ClaudeSdkSpec, ClaudeTtySpec, LiteLMSpec, McpHttp, McpStdio, PermissionMode

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("argv", "head", "tail"),
    [
        (
            ["new-acp", "x", "--cwd", "/tmp", "--", "agent", "--flag", "--"],
            ["new-acp", "x", "--cwd", "/tmp"],
            ["agent", "--flag", "--"],
        ),
        (["new-acp", "x"], ["new-acp", "x"], []),
        (["prompt", "x", "--", "-text"], ["prompt", "x", "--", "-text"], []),
    ],
)
def test_split_agent_command(argv: list[str], head: list[str], tail: list[str]) -> None:
    assert _split_agent_command(argv) == (head, tail)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["daemon"], [*PRELOAD]),
        (["daemon", "--preload-module", "litelm"], [*PRELOAD, "litelm"]),
        (["daemon", "--no-preload"], []),
        (["daemon", "--no-preload", "--preload-module", "zmq.asyncio"], ["zmq.asyncio"]),
    ],
)
def test_daemon_preload(argv: list[str], expected: list[str]) -> None:
    assert _preload(_parser().parse_args(argv)) == expected


def test_compact_command() -> None:
    args = _parser().parse_args(["compact", "s", "the billing work"])
    assert (args.command, args.name, args.focus) == ("compact", "s", "the billing work")
    assert _parser().parse_args(["compact", "s"]).focus is None


def test_daemon_allow_worker_command() -> None:
    assert _parser().parse_args(["daemon"]).allow_worker_command is False
    assert _parser().parse_args(["daemon", "--allow-worker-command"]).allow_worker_command is True


def test_new_py_max_requests() -> None:
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--max-requests", "150"])
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, LiteLMSpec)
    assert spec.max_requests == 150


def test_new_sdk_builds_spec() -> None:
    args = _parser().parse_args(["new-sdk", "s", "--model", "opus", "--permission", "allow"])
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, ClaudeSdkSpec)
    assert (spec.command, spec.model, spec.permission) == (["claude"], "opus", PermissionMode.ALLOW)


def test_new_py_worker_command(tmp_path: Path) -> None:
    argv = [
        "new-py",
        "s",
        "agents:echo",
        "--worker-command",
        "aid worker",
        "--worker-endpoint",
        "ipc://x",
        "--worker-ca",
        "/ca.pem",
        "--max-context",
        "500000",
    ]
    args = _parser().parse_args(argv)
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, LiteLMSpec)
    assert spec.worker_command == ["aid", "worker"]
    assert spec.worker_endpoint == "ipc://x"
    assert spec.worker_ca == "/ca.pem"
    assert spec.max_context == 500000


def test_new_py_worker_endpoint_needs_command() -> None:
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--worker-endpoint", "ipc://x"])
    args.agent_command = []
    with pytest.raises(SystemExit, match="need --worker-command"):
        _spec(args)


def test_new_py_worker_command_names_a_program() -> None:
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--worker-command", "   "])
    args.agent_command = []
    with pytest.raises(SystemExit, match="names no program"):
        _spec(args)


def test_mcp_config_files(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"mcpServers": {"files": {"command": "mcp-files", "args": ["/"]}}}))
    (tmp_path / "b.json").write_text(json.dumps({"mcpServers": {"web": {"type": "http", "url": "http://x/mcp"}}}))
    argv = ["new-claude", "c", "--mcp-config", str(tmp_path / "a.json"), "--mcp-config", str(tmp_path / "b.json")]
    args = _parser().parse_args(argv)
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, ClaudeTtySpec)
    assert spec.mcp_servers == [
        McpStdio(name="files", command=["mcp-files", "/"]),
        McpHttp(name="web", url="http://x/mcp"),
    ]


def test_tool_line_names_updates_by_their_start() -> None:
    """An update carries only the call id; the line still names the tool, and a failure shows why."""
    titles: dict[str, str] = {}
    start = ToolCall(tool_call_id="c1", title="shell", status="in_progress", input={"command": "ruff  check\nsrc"})
    assert _tool_line(start, titles) == '[tool] shell in_progress {"command": "ruff check\\nsrc"}'
    assert _tool_line(ToolCall(tool_call_id="c1", status="completed", output="ok"), titles) == "[tool] shell completed"
    failed = ToolCall(tool_call_id="c1", status="failed", output="exit 1:\n" + "x" * 200)
    line = _tool_line(failed, titles)
    assert line.startswith("[tool] shell failed exit 1: xxx") and line.endswith("…")
    assert _tool_line(ToolCall(tool_call_id="other", status="completed"), titles) == "[tool] other completed"
