from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.agents import ENV_AGENTS_PATH
from aid.mcp import claude_config, from_claude_config
from aid.paths import ENV_SESSION
from aid.spec import AcpSpec, McpHttp, McpSse, McpStdio, PermissionMode
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
    spec = acp_spec(tmp_path, AID_AGENTS_PATH="/agents").model_copy(update={"mcp_servers": SERVERS})
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
    fresh_json, resumed_json = json.loads(fresh.text), json.loads(resumed.text)
    builtin = fresh_json["servers"].pop()
    env = {e["name"]: e["value"] for e in builtin["env"]}
    assert fresh_json == {"via": "new", "servers": expected}
    assert resumed_json == {"via": "load", "servers": [*expected, builtin]}
    assert (builtin["name"], builtin["command"], builtin["args"]) == ("aid", sys.executable, ["-m", "aid.mcp_server"])
    assert env[ENV_SESSION] == "mcp"
    assert env[ENV_AGENTS_PATH] == "/agents"


@pytest.mark.anyio
async def test_aid_tools_can_be_left_out(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path).model_copy(update={"aid_tools": False})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            result = await (await client.create("plain", spec)).run("mcp")
    assert json.loads(result.text)["servers"] == []


def test_the_name_aid_is_reserved() -> None:
    with pytest.raises(ValidationError, match="aid's own tools"):
        AcpSpec(cwd="/", command=["a"], mcp_servers=[McpHttp(name="aid", url="http://x")])


@pytest.mark.anyio
async def test_acp_agent_without_the_transport_fails_to_start(daemon: Paths, tmp_path: Path) -> None:
    spec = acp_spec(tmp_path).model_copy(update={"mcp_servers": [McpSse(name="old", url="http://x")]})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(aid.AidError, match="does not take sse MCP servers"):
                await client.create("sse", spec)


@pytest.mark.anyio
async def test_acp_approves_aid_tools_by_their_name(daemon: Paths, tmp_path: Path) -> None:
    denying = acp_spec(tmp_path, PermissionMode.DENY)
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("perm", denying)
            aid_tool = await session.run("aid-tool")
            spoofed = await session.run("spoofed-title")
            other = await session.run("permission")
            off = await client.create("off", denying.model_copy(update={"aid_tools": False}))
            aid_tool_off = await off.run("aid-tool")
    assert (aid_tool.text, spoofed.text, other.text, aid_tool_off.text) == ("yes", "no", "no", "no")
