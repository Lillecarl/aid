"""aid's MCP server list, in the shapes ACP and Claude Code's `--mcp-config` take."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import TypeAdapter

from aid.spec import McpHttp, McpServer, McpSse, McpStdio

if TYPE_CHECKING:
    from collections.abc import Iterable

_servers: TypeAdapter[list[McpServer]] = TypeAdapter(list[McpServer])


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
