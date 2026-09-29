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
)
from aid.launcher import PRELOAD
from aid.spec import ClaudeTtySpec, McpHttp, McpStdio, PydanticAISpec

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
        (["daemon", "--preload-module", "pydantic_ai"], [*PRELOAD, "pydantic_ai"]),
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
    assert isinstance(spec, PydanticAISpec)
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
