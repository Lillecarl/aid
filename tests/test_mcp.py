from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.mcp import claude_config, from_claude_config
from aid.spec import AcpSpec, McpHttp, McpSse, McpStdio
from tests.conftest import acp_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

TIMEOUT = 30

SERVERS = [
    McpStdio(name="files", command=["/bin/mcp-files", "--root", "/"], env={"TOKEN": "t"}),
    McpHttp(name="web", url="http://127.0.0.1:1/mcp", headers={"Authorization": "Bearer t"}),
]
CLAUDE_CONFIG = {
    "mcpServers": {
        "files": {"type": "stdio", "command": "/bin/mcp-files", "args": ["--root", "/"], "env": {"TOKEN": "t"}},
        "web": {"type": "http", "url": "http://127.0.0.1:1/mcp", "headers": {"Authorization": "Bearer t"}},
    }
}


def test_claude_config_round_trip() -> None:
    assert claude_config(SERVERS) == CLAUDE_CONFIG
    assert from_claude_config(CLAUDE_CONFIG) == SERVERS


def test_untyped_claude_entry_is_stdio() -> None:
    assert from_claude_config({"mcpServers": {"x": {"command": "mcp-x"}}}) == [McpStdio(name="x", command=["mcp-x"])]


def test_names_are_unique_and_plain() -> None:
    with pytest.raises(ValidationError):
        AcpSpec(cwd="/", command=["a"], mcp_servers=[SERVERS[0], SERVERS[0]])
    with pytest.raises(ValidationError):
        McpHttp(name="has space", url="http://x")


@pytest.mark.anyio
async def test_acp_agent_gets_the_servers_on_new_and_load(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path).model_copy(update={"mcp_servers": SERVERS})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("mcp", spec)
            fresh = await session.run("mcp")
            await session.stop()
            resumed = await session.run("mcp")
    expected = [
        {
            "name": "files",
            "command": "/bin/mcp-files",
            "args": ["--root", "/"],
            "env": [{"name": "TOKEN", "value": "t"}],
        },
        {
            "type": "http",
            "name": "web",
            "url": "http://127.0.0.1:1/mcp",
            "headers": [{"name": "Authorization", "value": "Bearer t"}],
        },
    ]
    assert json.loads(fresh.text) == {"via": "new", "servers": expected}
    assert json.loads(resumed.text) == {"via": "load", "servers": expected}


@pytest.mark.anyio
async def test_acp_agent_without_the_transport_fails_to_start(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path).model_copy(update={"mcp_servers": [McpSse(name="old", url="http://x")]})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(aid.AidError, match="does not take sse MCP servers"):
                await client.create("sse", spec)
