from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

# The parsing rules under test are private to the CLI.
from aid.cli import (
    _parser,  # pyright: ignore[reportPrivateUsage]
    _spec,  # pyright: ignore[reportPrivateUsage]
    _split_agent_command,  # pyright: ignore[reportPrivateUsage]
)
from aid.spec import ClaudeTtySpec, McpHttp, McpStdio

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
