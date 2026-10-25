from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

# The parsing rules under test are private to the CLI.
from aid.cli import (
    _history_line,  # pyright: ignore[reportPrivateUsage]
    _parser,  # pyright: ignore[reportPrivateUsage]
    _preload,  # pyright: ignore[reportPrivateUsage]
    _query_table,  # pyright: ignore[reportPrivateUsage]
    _session_line,  # pyright: ignore[reportPrivateUsage]
    _spec,  # pyright: ignore[reportPrivateUsage]
    _split_agent_command,  # pyright: ignore[reportPrivateUsage]
    _tagged,  # pyright: ignore[reportPrivateUsage]
    _tags,  # pyright: ignore[reportPrivateUsage]
    _tool_line,  # pyright: ignore[reportPrivateUsage]
)
from aid.launcher import PRELOAD
from aid.protocol import Cost, HistoryEntry, QueryResult, SessionInfo, ToolCall, Usage
from aid.spec import AgentKind, ClaudeSdkSpec, ClaudeTtySpec, McpHttp, McpStdio, NativeSpec, PermissionMode

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
        (["daemon", "--preload-module", "openai"], [*PRELOAD, "openai"]),
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
    assert isinstance(spec, NativeSpec)
    assert spec.max_requests == 150


def test_new_py_takes_a_tool_ceiling() -> None:
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--tool", "ls", "--tool", "read|outline"])
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, NativeSpec)
    assert spec.tools == ["ls", "read|outline"]
    args = _parser().parse_args(["new-py", "s", "agents:echo"])
    args.agent_command = []
    assert isinstance(spec := _spec(args), NativeSpec) and spec.tools is None


def test_new_py_makes_an_agent_from_instructions(tmp_path: Path) -> None:
    args = _parser().parse_args(["new-py", "s", "--instructions", "Read.", "--model", "a/b", "--tool", "read"])
    args.agent_command = []
    spec = _spec(args)
    assert isinstance(spec, NativeSpec)
    assert (spec.agent, spec.instructions, spec.model, spec.tools, spec.runs) == (
        None,
        "Read.",
        "a/b",
        ["read"],
        "instructions",
    )
    (file := tmp_path / "instructions.md").write_text("From a file.")
    args = _parser().parse_args(["new-py", "s", "--instructions-file", str(file), "--model", "a/b", "--tool", "ls"])
    args.agent_command = []
    assert isinstance(spec := _spec(args), NativeSpec) and spec.instructions == "From a file."
    args = _parser().parse_args(["new-py", "s", "--instructions", "Read.", "--model", "a/b"])
    args.agent_command = []
    with pytest.raises(ValidationError, match="sets model and tools"):
        _spec(args)
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--instructions", "Read."])
    args.agent_command = []
    with pytest.raises(SystemExit, match="not both"):
        _spec(args)


def test_new_takes_tags() -> None:
    args = _parser().parse_args(["new-py", "s", "agents:echo", "--tag", "team=web", "--tag", "x=1=2"])
    args.agent_command = []
    assert _spec(args).tags == {"team": "web", "x": "1=2"}
    with pytest.raises(SystemExit):
        _tags(["novalue"])
    assert _tags(["gone="]) == {"gone": None}


@pytest.mark.parametrize(
    ("wanted", "shown"), [([], True), (["team"], True), (["team=web"], True), (["team=api"], False), (["x"], False)]
)
def test_list_filters_by_tag(wanted: list[str], shown: bool) -> None:
    info = SessionInfo(name="a", kind=AgentKind.NATIVE, running=False, tags={"team": "web"})
    assert _tagged(info, wanted) is shown


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
    assert isinstance(spec, NativeSpec)
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


@pytest.mark.parametrize(
    ("info", "line"),
    [
        (SessionInfo(name="a", kind=AgentKind.NATIVE, running=False), "a\tnative\tstopped"),
        (SessionInfo(name="a", kind=AgentKind.NATIVE, running=True), "a\tnative\trunning"),
        (
            SessionInfo(name="a", kind=AgentKind.NATIVE, running=True, working=True, permissions=1),
            "a\tnative\tworking\t1 permission",
        ),
        (
            SessionInfo(name="a", kind=AgentKind.NATIVE, running=True, permissions=2, attention="look\there"),
            "a\tnative\trunning\t2 permissions\tlook\there",
        ),
        (
            SessionInfo(name="a", kind=AgentKind.NATIVE, running=False, tags={"z": "1", "a": "b c"}),
            "a\tnative\tstopped\ta=b c z=1",
        ),
    ],
)
def test_session_line(info: SessionInfo, line: str) -> None:
    assert _session_line(info) == line


def test_list_json() -> None:
    assert _parser().parse_args(["list", "--json"]).json is True


def test_query_table_keeps_a_row_on_one_line() -> None:
    """Columns first, then a row per line; tabs and newlines in a cell are escaped, and None is empty."""
    result = QueryResult(
        columns=["session", "input", "allowed"],
        rows=[["a", '{"command": "x\ty"}', None], ["b", "two\nlines", 1]],
        truncated=False,
    )
    assert _query_table(result) == 'session\tinput\tallowed\na\t{"command": "x\\ty"}\t\nb\ttwo\\nlines\t1'
    empty = QueryResult(columns=["n"], rows=[], truncated=False)
    assert _query_table(empty) == "n"


@pytest.mark.parametrize(("amount", "shown"), [(0.0011324, "0.0011 USD"), (0.25, "0.25 USD"), (12.345, "12.35 USD")])
def test_a_usage_line_shows_its_cost(amount: float, shown: str) -> None:
    usage = Usage(
        input_tokens=6, output_tokens=42, models=["claude-haiku-5-5"], cost=Cost(amount=amount, currency="USD")
    )
    line = _history_line(HistoryEntry(seq=5, at=0, turn="t", item=usage), {})
    assert line == f"     5  [usage] 6 in, 42 out claude-haiku-5-5 {shown}"
