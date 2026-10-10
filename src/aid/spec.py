"""Agent specifications: what a session runs, independent of how it is hosted."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class AgentKind(StrEnum):
    ACP = "acp"
    PYDANTIC_AI = "pydantic-ai"
    CLAUDE_TTY = "claude-tty"
    CLAUDE_SDK = "claude-sdk"


class PermissionMode(StrEnum):
    """How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    """A person answers, from the web UI or `aid answer`; the request is refused after `permission_timeout`."""


def _compile_pattern(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid hook match regex {pattern!r}: {e}") from e
    return pattern


type HookMatch = Annotated[str, AfterValidator(_compile_pattern)]


class Hook(BaseModel):
    """A pre-tool gate: when `tool` runs with rendered text matching `match`, refuse it with `message`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    """A coding tool name (`python`, `shell`, `apply_edits`), or `*` for all tools."""
    match: HookMatch
    message: str
    """Fed back to the model on block."""


def match_hook(hooks: list[Hook], tool: str | None, text: str | None) -> Hook | None:
    """First hook whose tool fits and whose match finds `text`, else None. Never raises."""
    if not isinstance(tool, str) or not isinstance(text, str):
        return None
    try:
        candidates = list(hooks)
    except TypeError:
        return None
    for hook in candidates:
        try:
            if hook.tool != "*" and hook.tool != tool:
                continue
            if re.search(hook.match, text):
                return hook
        except re.error, TypeError, AttributeError:
            continue
    return None


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: str
    env: dict[str, str] = Field(default_factory=dict[str, str])
    aid_tools: bool = True
    """Offer the `@aid.mcptool` functions on the agents path: as the MCP server `aid`, or in-process to a
    pydantic-ai agent."""
    worker_command: Annotated[list[str], Field(min_length=1)] | None = None
    """Start this session's worker by running this command instead of forking it. It must run `aid worker`,
    which reads its arguments as JSON from stdin. A command runs with the daemon's environment and working
    directory; point it at another checkout or closure to try new worker code while other sessions keep theirs.
    The daemon refuses it unless started with --allow-worker-command: a command runs as the daemon with the
    session's worker credentials."""
    worker_endpoint: str | None = None
    """The workers socket the command reaches. None means the daemon's own; set it for a command on another
    host, with the daemon listening there (`aid daemon --workers-listen`)."""
    worker_ca: str | None = None
    """PEM CAs file for a `wss://` worker_endpoint with a private certificate, like `aid daemon --worker-ca`."""
    supervisor: str | None = None
    """The session watching this one: the daemon hands it the clipped tail of every turn end. The supervisor
    must already exist, must not be this session, and must not close a cycle; the link rides the persisted
    spec file, so it survives restarts."""
    supervisor_permissions: list[str] = Field(default_factory=list[str])
    """Tool names the supervisor may answer permission requests for when the person does not: allow-listed
    tools only, and only in `ask` mode. Empty means no backstop — the supervisor never answers permissions."""
    supervisor_permission_timeout: Annotated[float, Field(gt=0)] = 300
    """Seconds a permission request waits for a person before the daemon wakes the supervisor. Also the
    presence window: a person-authored entry (message, permission answer) this recent keeps the full wait."""
    supervisor_permission_absent_timeout: Annotated[float, Field(gt=0)] = 60
    """The wait above, shortened when nobody is around: no person-authored entry in the presence window."""
    supervisor_nudge_budget: Annotated[int, Field(ge=0)] = 3
    """Nudges the supervisor may send per stuck episode — the run of supervised turns since the last
    unstuck one. Past it the daemon refuses nudges; K stuck turns after K nudges raises `attention` to
    the person instead. Zero never nudges, and escalates the first stuck turn."""
    hooks: list[Hook] = Field(default_factory=list[Hook])
    """Pre-tool gates: the first hook whose tool fits and whose match finds the call's text refuses it."""

    @model_validator(mode="after")
    def _worker_launch_is_sane(self) -> Self:
        if self.worker_command is not None and not self.worker_command[0]:
            raise ValueError("worker_command names no program")
        if self.worker_command is None and (self.worker_endpoint is not None or self.worker_ca is not None):
            raise ValueError("worker_endpoint and worker_ca need worker_command")
        return self


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
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    """Seconds an `ask` request waits for a person before it is refused."""
    mcp_servers: McpServers = Field(default_factory=list[McpServer])


class PydanticAISpec(_Spec):
    """A pydantic-ai agent: `agent`, the name of an `aid.PydanticAgent` on the agents path, or `target`, a
    `package.module:attribute` that is an agent itself. Exactly one of them."""

    kind: Literal[AgentKind.PYDANTIC_AI] = AgentKind.PYDANTIC_AI
    agent: Annotated[str, Field(pattern=r"^[\w.-]+$")] | None = None
    target: Annotated[str, Field(pattern=r"^[\w.]+:[\w.]+$")] | None = None
    python_path: list[str] = Field(default_factory=list[str])
    permission: PermissionMode = PermissionMode.DENY
    """How aid answers what `aid.coding` tools ask: each command a `python` script starts, each edit applied."""
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    ask_autoselect_after: Annotated[float, Field(gt=0)] = 240.0
    """Seconds an `ask_user` question with a recommended option waits for a person before it picks it."""
    max_context: Annotated[int, Field(gt=0)] | None = None
    """Tokens the context gauge calls 100%: rule of thumb half the model's window. None takes a conservative
    backend default."""
    max_requests: Annotated[int, Field(gt=0)] | None = None
    """Model requests one turn may spend before it ends capped. None keeps pydantic-ai's default; coding
    sessions doing real tasks want more."""
    instruction_providers: list[str] | None = None
    """Providers that load this session's instructions, in order. None runs every provider the agents path offers,
    in catalog order."""
    skip_instructions: list[str] = Field(default_factory=list[str])
    """Instruction names to drop after loading: `skill:<name>` opts out of one skill."""

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


class ClaudeSdkSpec(_Spec):
    """Claude Code driven directly over NDJSON (`claude -p`), so sessions get Remote Control.

    Skeleton slice A: one process per prompt, resumed by the persisted session id. Permissions, Remote Control,
    and CLI wiring arrive in later slices.
    """

    kind: Literal[AgentKind.CLAUDE_SDK] = AgentKind.CLAUDE_SDK
    command: Annotated[list[str], Field(min_length=1)] = Field(default_factory=lambda: ["claude"])
    """The `claude` program and its fixed arguments. Tests point it at a fake."""
    model: str
    """Passed as `--model`: the model the session runs on."""
    inherit_env: bool = True
    permission: PermissionMode = PermissionMode.DENY
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    """Seconds an `ask` request waits for a person before it is refused."""


type AgentSpec = Annotated[AcpSpec | PydanticAISpec | ClaudeTtySpec | ClaudeSdkSpec, Field(discriminator="kind")]

AgentSpecAdapter: TypeAdapter[AgentSpec] = TypeAdapter(AgentSpec)
