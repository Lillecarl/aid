"""Agent specifications: what a session runs, independent of how it is hosted."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class AgentKind(StrEnum):
    ACP = "acp"
    PYDANTIC_AI = "pydantic-ai"
    CLAUDE_TTY = "claude-tty"


class PermissionMode(StrEnum):
    """Answer the daemon gives an ACP agent's permission requests. Nobody is watching to ask."""

    ALLOW = "allow"
    DENY = "deny"


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: str
    env: dict[str, str] = Field(default_factory=dict[str, str])
    aid_tools: bool = True
    """Offer the `@aid.mcptool` functions on the agents path: as the MCP server `aid`, or in-process to a
    pydantic-ai agent."""


BUILTIN_MCP_SERVER: Final = "aid"


class _McpServer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Claude Code builds tool names as mcp__<name>__<tool>.
    name: Annotated[str, Field(pattern=r"^[\w-]+$")]


class McpStdio(_McpServer):
    type: Literal["stdio"] = "stdio"
    command: Annotated[list[str], Field(min_length=1)]
    env: dict[str, str] = Field(default_factory=dict[str, str])


class McpHttp(_McpServer):
    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict[str, str])


class McpSse(_McpServer):
    type: Literal["sse"] = "sse"
    url: str
    headers: dict[str, str] = Field(default_factory=dict[str, str])


type McpServer = Annotated[McpStdio | McpHttp | McpSse, Field(discriminator="type")]


def _unique_names(servers: list[McpServer]) -> list[McpServer]:
    names = [server.name for server in servers]
    if len(set(names)) != len(names):
        raise ValueError(f"MCP server names repeat: {names}")
    if BUILTIN_MCP_SERVER in names:
        raise ValueError(f"the MCP server name {BUILTIN_MCP_SERVER!r} belongs to aid's own tools")
    return servers


type McpServers = Annotated[list[McpServer], AfterValidator(_unique_names)]


class AcpSpec(_Spec):
    """An external ACP agent, such as `claude-agent-acp` or `opencode acp`."""

    kind: Literal[AgentKind.ACP] = AgentKind.ACP
    command: Annotated[list[str], Field(min_length=1)]
    inherit_env: bool = True
    permission: PermissionMode = PermissionMode.DENY
    mcp_servers: McpServers = Field(default_factory=list[McpServer])


class PydanticAISpec(_Spec):
    """A pydantic-ai agent: `agent`, the name of an `aid.PydanticAgent` on the agents path, or `target`, a
    `package.module:attribute` that is an agent itself. Exactly one of them."""

    kind: Literal[AgentKind.PYDANTIC_AI] = AgentKind.PYDANTIC_AI
    agent: Annotated[str, Field(pattern=r"^[\w.-]+$")] | None = None
    target: Annotated[str, Field(pattern=r"^[\w.]+:[\w.]+$")] | None = None
    python_path: list[str] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.agent is None) == (self.target is None):
            raise ValueError("set exactly one of agent and target")
        return self


class ClaudeTtySpec(_Spec):
    """Interactive Claude Code in a pymux window: prompts are pasted in, events come from its transcript.

    A person can attach to the window, and Remote Control works, which the Agent SDK behind ACP does not offer.
    """

    kind: Literal[AgentKind.CLAUDE_TTY] = AgentKind.CLAUDE_TTY
    command: Annotated[list[str], Field(min_length=1)] = Field(default_factory=lambda: ["claude"])
    args: list[str] = Field(default_factory=list[str])
    inherit_env: bool = True
    trust_cwd: bool = False
    """Answer Claude Code's "do you trust this folder" dialog with yes. Without it, an untrusted cwd fails."""
    pymux_socket: str | None = None
    """The pymux server to run in. None uses one owned by aid, under its runtime directory."""
    pymux_command: Annotated[list[str], Field(min_length=1)] = Field(default_factory=lambda: ["pymux"])
    mcp_servers: McpServers = Field(default_factory=list[McpServer])
    """Added to the MCP servers Claude Code finds in its own configuration."""


type AgentSpec = Annotated[AcpSpec | PydanticAISpec | ClaudeTtySpec, Field(discriminator="kind")]

AgentSpecAdapter: TypeAdapter[AgentSpec] = TypeAdapter(AgentSpec)
