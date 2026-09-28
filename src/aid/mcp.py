"""aid's MCP server list, in the shapes ACP and Claude Code's `--mcp-config` take."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Final

from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import TypeAdapter

from aid.agents import ENV_AGENTS_PATH, agents_path
from aid.paths import ENV_CHANNEL, ENV_RUNTIME_DIR, ENV_SESSION, ENV_STATE_DIR, default_paths
from aid.spec import BUILTIN_MCP_SERVER, AgentKind, McpHttp, McpServer, McpSse, McpStdio

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aid.spec import AcpSpec, ClaudeTtySpec

_servers: TypeAdapter[list[McpServer]] = TypeAdapter(list[McpServer])


# How Claude Code names the built-in server's tools; also a permission rule that allows all of them.
AID_TOOLS_RULE: Final = f"mcp__{BUILTIN_MCP_SERVER}"
AID_TOOL_PREFIX: Final = f"{AID_TOOLS_RULE}__"


def builtin_server(session: str, *, channel: bool) -> McpStdio:
    """`aid.mcp_server` for one session. The agent starts it with its own environment, which may lack ours.

    `channel` makes it the session's channel too; only interactive Claude listens to one.
    """
    paths = default_paths()
    env = {
        ENV_AGENTS_PATH: ":".join(str(p) for p in agents_path()),
        ENV_SESSION: session,
        ENV_RUNTIME_DIR: str(paths.runtime_dir),
        ENV_STATE_DIR: str(paths.state_dir),
    }
    if channel:
        env[ENV_CHANNEL] = "1"
    # The dev shell runs aid from src/ through PYTHONPATH; without it the server imports the venv's copy.
    if python_path := os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = python_path
    return McpStdio(name=BUILTIN_MCP_SERVER, command=[sys.executable, "-m", "aid.mcp_server"], env=env)


def session_servers(spec: AcpSpec | ClaudeTtySpec, session: str) -> list[McpServer]:
    """What a session's agent gets. One function for new and resumed sessions, so both see the same list."""
    if not spec.aid_tools:
        return list(spec.mcp_servers)
    return [*spec.mcp_servers, builtin_server(session, channel=spec.kind is AgentKind.CLAUDE_TTY)]


def to_acp(server: McpServer) -> McpServerStdio | HttpMcpServer | SseMcpServer:
    match server:
        case McpStdio():
            env = [EnvVariable(name=k, value=v) for k, v in server.env.items()]
            return McpServerStdio(name=server.name, command=server.command[0], args=server.command[1:], env=env)
        case McpHttp():
            headers = [HttpHeader(name=k, value=v) for k, v in server.headers.items()]
            return HttpMcpServer(type="http", name=server.name, url=server.url, headers=headers)
        case McpSse():
            headers = [HttpHeader(name=k, value=v) for k, v in server.headers.items()]
            return SseMcpServer(type="sse", name=server.name, url=server.url, headers=headers)


def claude_config(servers: Iterable[McpServer]) -> dict[str, Any]:
    """The `{"mcpServers": {...}}` document `claude --mcp-config` reads."""
    entries: dict[str, Any] = {}
    for server in servers:
        match server:
            case McpStdio():
                entry = {"type": "stdio", "command": server.command[0], "args": server.command[1:], "env": server.env}
            case McpHttp() | McpSse():
                entry = {"type": server.type, "url": server.url, "headers": server.headers}
        entries[server.name] = entry
    return {"mcpServers": entries}


def from_claude_config(document: dict[str, Any]) -> list[McpServer]:
    """Read a Claude Code `.mcp.json`-style document. Entries without `type` are stdio, as in Claude Code."""
    servers: list[dict[str, Any]] = []
    for name, entry in dict(document.get("mcpServers", {})).items():
        entry = dict(entry)
        kind = entry.pop("type", "stdio")
        if kind == "stdio" and "command" in entry:
            command = [entry.pop("command"), *entry.pop("args", [])]
            servers.append({"type": kind, "name": name, "command": command, **entry})
        else:
            servers.append({"type": kind, "name": name, **entry})
    return _servers.validate_python(servers)
