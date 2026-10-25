"""aid's MCP server list, in the shapes ACP and Claude Code's `--mcp-config` take."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import JsonValue, TypeAdapter

from aid.control import ENV_AGENT_CONNECTION
from aid.paths import ENV_CHANNEL, ENV_RUNTIME_DIR, ENV_SESSION, ENV_STATE_DIR, default_paths
from aid.plugins import ENV_PLUGINS_PATH, plugins_path
from aid.spec import BUILTIN_MCP_SERVER, AgentKind, McpHttp, McpServer, McpSse, McpStdio, PermissionMode

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import anyio

    from aid.spec import AcpSpec, ClaudeSdkSpec, ClaudeTtySpec

_servers: TypeAdapter[list[McpServer]] = TypeAdapter(list[McpServer])


class ClaudeStdioEntry(TypedDict):
    type: Literal["stdio"]
    command: str
    args: list[str]
    env: dict[str, str]


class ClaudeRemoteEntry(TypedDict):
    type: Literal["http", "sse"]
    url: str
    headers: dict[str, str]


class ClaudeMcpConfig(TypedDict):
    """What `claude --mcp-config` reads."""

    mcpServers: dict[str, ClaudeStdioEntry | ClaudeRemoteEntry]


# How Claude Code names the built-in server's tools; also a permission rule that allows all of them.
AID_TOOLS_RULE: Final = f"mcp__{BUILTIN_MCP_SERVER}"
AID_TOOL_PREFIX: Final = f"{AID_TOOLS_RULE}__"


def claude_rules(modes: Mapping[str, PermissionMode], *, aid_tools: bool) -> tuple[list[str], list[str], list[str]]:
    """Tool modes as Claude Code's permission rules: allowed (aid's own tools first, with `aid_tools`), denied,
    asked."""
    allowed = [*([AID_TOOLS_RULE] if aid_tools else []), *(t for t, m in modes.items() if m is PermissionMode.ALLOW)]
    denied = [t for t, m in modes.items() if m is PermissionMode.DENY]
    # Claude must ask for every waiting mode: aid hears only what Claude asks, and routes it from there.
    asked = [t for t, m in modes.items() if m.waits]
    return allowed, denied, asked


def builtin_server(session: str, *, channel: bool) -> McpStdio:
    """`aid.mcp_server` for one session. The agent starts it with its own environment, which may lack ours.

    `channel` makes it the session's channel too; only interactive Claude listens to one.
    """
    paths = default_paths()
    env = {
        ENV_PLUGINS_PATH: ":".join(str(p) for p in plugins_path()),
        ENV_SESSION: session,
        ENV_RUNTIME_DIR: str(paths.runtime_dir),
        ENV_STATE_DIR: str(paths.state_dir),
    }
    if channel:
        env[ENV_CHANNEL] = "1"
    if connection := os.environ.get(ENV_AGENT_CONNECTION):
        env[ENV_AGENT_CONNECTION] = connection
    # The dev shell runs aid from src/ through PYTHONPATH; without it the server imports the venv's copy.
    if python_path := os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = python_path
    return McpStdio(name=BUILTIN_MCP_SERVER, command=[sys.executable, "-m", "aid.mcp_server"], env=env)


MCP_CONFIG_FILE: Final = "mcp.json"
"""Where a Claude Code session's `--mcp-config` document lives, in its state directory."""


async def write_private(path: anyio.Path, text: str, mode: int) -> None:
    """Environments and MCP headers hold credentials; the file gets its mode before it gets them."""
    await path.touch(mode=mode)
    await path.chmod(mode)
    await path.write_text(text)


def session_servers(spec: AcpSpec | ClaudeTtySpec | ClaudeSdkSpec, session: str) -> list[McpServer]:
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


def claude_config(servers: Iterable[McpServer]) -> ClaudeMcpConfig:
    entries: dict[str, ClaudeStdioEntry | ClaudeRemoteEntry] = {}
    for server in servers:
        match server:
            case McpStdio():
                entries[server.name] = ClaudeStdioEntry(
                    type="stdio", command=server.command[0], args=server.command[1:], env=server.env
                )
            case McpHttp() | McpSse():
                entries[server.name] = ClaudeRemoteEntry(type=server.type, url=server.url, headers=server.headers)
    return {"mcpServers": entries}


def from_claude_config(document: JsonValue) -> list[McpServer]:
    """Read a Claude Code `.mcp.json`-style document. Entries without `type` are stdio, as in Claude Code."""
    entries = document.get("mcpServers", {}) if isinstance(document, dict) else None
    if not isinstance(entries, dict):
        raise ValueError("not a JSON object with an mcpServers object")
    servers: list[JsonValue] = []
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"mcpServers.{name} is not an object")
        entry = dict(entry)
        kind = entry.pop("type", "stdio")
        if kind == "stdio" and "command" in entry:
            args = entry.pop("args", [])
            if not isinstance(args, list):
                raise ValueError(f"mcpServers.{name}.args is not a list")
            servers.append({"type": kind, "name": name, "command": [entry.pop("command"), *args], **entry})
        else:
            servers.append({"type": kind, "name": name, **entry})
    return _servers.validate_python(servers)
