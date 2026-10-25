"""Agent specifications: what a session runs, independent of how it is hosted."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator


class AgentKind(StrEnum):
    ACP = "acp"
    NATIVE = "native"
    CLAUDE_TTY = "claude-tty"
    CLAUDE_SDK = "claude-sdk"


class PermissionMode(StrEnum):
    """How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    """A person answers, from the web UI or `aid answer`; the request is refused after `permission_timeout`."""
    SUPERVISOR = "supervisor"
    """The session's supervisor answers first. Unsure, it hands the request to a person, and it comes back to the
    supervisor if nobody answers. Without a supervisor it is `ask`."""
    ASK_SUPERVISOR = "ask_supervisor"
    """A person answers first; past `supervisor_permission_timeout` the supervisor does, and it may hand the request
    back to the person once. Without a supervisor it is `ask`."""

    @property
    def waits(self) -> bool:
        """The engine asks and waits for an answer, refused after `permission_timeout`. Who is asked, and in what
        order, is the supervision plugin's routing: engines treat every waiting mode alike."""
        return self not in (PermissionMode.ALLOW, PermissionMode.DENY)


def _compile_pattern(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid hook match regex {pattern!r}: {e}") from e
    return pattern


type HookMatch = Annotated[str, AfterValidator(_compile_pattern)]


def _compile_tool_pattern(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid tool pattern {pattern!r}: {e}") from e
    return pattern


type ToolPattern = Annotated[str, AfterValidator(_compile_tool_pattern)]
"""A regex matched against a whole tool name."""


def _absolute(paths: list[str]) -> list[str]:
    if relative := [p for p in paths if not p.startswith("/")]:
        raise ValueError(f"sandbox paths must be absolute: {relative}")
    return paths


class Sandbox(BaseModel):
    """bubblewrap around what a native session's tools start (`aid.sandbox`): the filesystem read-only but the
    session's directory and `writable`, a private /tmp. A session that asks for one never runs without it: no bwrap,
    or no user namespaces, and the worker refuses to start; `python`'s `in-loop` mode, which no sandbox can hold, is
    refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network: bool = True
    writable: Annotated[list[str], AfterValidator(_absolute)] = Field(default_factory=list[str])
    """Absolute paths the shell may write besides the session's directory: caches, a build output."""


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


TagKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:/-]{1,64}$")]
TagValue = Annotated[str, Field(max_length=256)]

_FIELDS: Final = TypeAdapter(dict[str, object])
_TOOLS: Final = TypeAdapter(list[str])
_MODES: Final = TypeAdapter(dict[str, PermissionMode])


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: str
    env: dict[str, str] = Field(default_factory=dict[str, str])
    aid_tools: bool = True
    """Offer the `@aid.mcptool` functions on the plugins path: as the MCP server `aid`, or in-process to a
    native agent."""
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
    """The session watching this one: the daemon hands it the clipped tail of every turn end, and it answers
    the permission requests of tools in `supervisor` and `ask_supervisor` mode. The supervisor must already
    exist, must not be this session, and must not close a cycle; the link rides the persisted spec file, so it
    survives restarts."""
    supervisor_permission_timeout: Annotated[float, Field(gt=0)] = 300
    """Seconds a person holds a request in a supervisor mode before it moves on. Also the presence window: a
    person-authored entry (message, permission answer) this recent keeps the full wait."""
    supervisor_permission_absent_timeout: Annotated[float, Field(gt=0)] = 60
    """The wait above, shortened when nobody is around: no person-authored entry in the presence window."""
    supervisor_nudge_budget: Annotated[int, Field(ge=0)] = 3
    """Nudges the supervisor may send per stuck episode — the run of supervised turns since the last
    unstuck one. Past it the daemon refuses nudges; K stuck turns after K nudges raises `attention` to
    the person instead. Zero never nudges, and escalates the first stuck turn."""
    supervisor_classifier: str | None = "classifier"
    """What judges a routed permission request the read-only rules leave, before the supervisor's route: an agent
    name, or a `module:attr` target. It runs in a session of its own beside the supervisor, one fresh turn per
    request, and allows, denies with a reason, or escalates. None sends such requests straight to the route."""
    hooks: list[Hook] = Field(default_factory=list[Hook])
    """Pre-tool gates: the first hook whose tool fits and whose match finds the call's text refuses it."""
    parent: str | None = None
    """The session this one works for, such as a fork's parent: its results reach the parent as messages.
    Chat plugins leave such a session out; a person follows it through the parent."""
    tags: dict[TagKey, TagValue] = Field(default_factory=dict[str, str])
    """Free-form labels for people and plugins to sort sessions by; aid itself acts on none."""

    @model_validator(mode="before")
    @classmethod
    def _supervisor_permissions(cls, data: object) -> object:
        """A stored spec's `supervisor_permissions`: tools the supervisor answered after the person, in `ask`
        mode only. Each such tool is in `ask_supervisor` mode now; the daemon writes the spec back without it."""
        if not isinstance(data, dict):
            return data
        fields = _FIELDS.validate_python(data)
        if "supervisor_permissions" not in fields:
            return fields
        listed = _TOOLS.validate_python(fields.pop("supervisor_permissions") or [])
        modes = _MODES.validate_python(fields.get("tool_permissions") or {})
        default = fields.get("permission")
        for tool in listed:
            if modes.get(tool, default) == PermissionMode.ASK:
                modes[tool] = PermissionMode.ASK_SUPERVISOR
        if modes:
            fields["tool_permissions"] = modes
        return fields

    @model_validator(mode="after")
    def _worker_launch_is_sane(self) -> Self:
        if self.worker_command is not None and not self.worker_command[0]:
            raise ValueError("worker_command names no program")
        if self.worker_command is None and (self.worker_endpoint is not None or self.worker_ca is not None):
            raise ValueError("worker_endpoint and worker_ca need worker_command")
        return self


BUILTIN_MCP_SERVER: Final = "aid"


def tool_group(name: str, own: str) -> str:
    """The server's name for an MCP tool as Claude Code names it (`mcp__<server>__<tool>`), else `own`: the agent's
    own tools."""
    if name.startswith("mcp__") and (server := name.removeprefix("mcp__").split("__", 1)[0]):
        return server
    return own


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
    """How aid answers a request about a tool `tool_permissions` does not name; aid's own tools are allowed."""
    tool_permissions: dict[str, PermissionMode] = Field(default_factory=dict[str, PermissionMode])
    """Modes by the tool name a request carries (claude-agent-acp's `toolCall.name`). aid answers only what the
    agent asks: a tool the agent runs without asking runs whatever its mode says."""
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    """Seconds an `ask` request waits for a person before it is refused."""
    mcp_servers: McpServers = Field(default_factory=list[McpServer])

    def mode_for(self, tool_name: str | None) -> PermissionMode:
        """The tool's own mode, else aid's own tools are approved: whoever put their plugins on the path chose them."""
        if tool_name is not None and (mode := self.tool_permissions.get(tool_name)) is not None:
            return mode
        if self.aid_tools and tool_name is not None and tool_name.startswith(f"mcp__{BUILTIN_MCP_SERVER}__"):
            return PermissionMode.ALLOW
        return self.permission


class NativeSpec(_Spec):
    """A native agent: `agent`, the name of an `aid.AgentDefinition` on the plugins path, `target`, a
    `package.module:attribute` that is an agent itself, or `instructions`, which makes the spec the agent. Exactly
    one of them."""

    kind: Literal[AgentKind.NATIVE] = AgentKind.NATIVE
    agent: Annotated[str, Field(pattern=r"^[\w.-]+$")] | None = None
    target: Annotated[str, Field(pattern=r"^[\w.]+:[\w.]+$")] | None = None
    instructions: str | None = None
    """The agent's own instructions, for an agent made from this spec alone: it needs `model` and `tools`, and its
    tools are every named tool (`aid.coding_tools`, aid's and plugins'), so `tools` alone picks what it gets."""
    model: Annotated[str, Field(pattern=r"^\S+$")] | None = None
    """`provider/model` the session runs on (`aid models` lists them), over the agent's own. The agent's
    `model_settings` follow only to a model of the same provider: they may hold keys another provider refuses."""
    model_settings: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    """Over the agent's `model_settings`: request settings (`max_tokens`, `reasoning_effort`, ...) and client ones
    (`api_key`, `api_base`, `max_retries`)."""
    python_path: list[str] = Field(default_factory=list[str])
    permission: PermissionMode = PermissionMode.DENY
    """The mode of tools that ask for themselves (`Tool.asks`: each command a `python` script starts, each edit
    applied) when `tool_permissions` names no mode of their own."""
    tool_permissions: dict[str, PermissionMode] = Field(default_factory=dict[str, PermissionMode])
    """Modes by tool name. A tool it does not name follows `permission` if it asks for itself, else is allowed."""
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    ask_autoselect_after: Annotated[float, Field(gt=0)] = 240.0
    """Seconds an `ask_user` question with a recommended option waits for a person before it picks it."""
    sandbox: Sandbox | None = None
    """Run the shell, `background` tasks and `python` scripts under bubblewrap. None runs them as the user."""
    max_context: Annotated[int, Field(gt=0)] | None = None
    """Tokens the context gauge calls 100%: rule of thumb half the model's window. None takes a conservative
    backend default."""
    max_requests: Annotated[int, Field(gt=0)] | None = None
    """Optional model-request budget per turn. None means unlimited, never the engine's implicit default.
    Resource budgets are not loop detection: a supervisor judges suspected non-progress separately."""
    instruction_providers: list[str] | None = None
    """Providers that load this session's instructions, in order. None runs every provider the plugins path offers,
    in catalog order."""
    skip_instructions: list[str] = Field(default_factory=list[str])
    """Instruction names to drop after loading: `skill:<name>` opts out of one skill."""
    seed_history: str | None = None
    """A history file, absolute, the session starts from when it has none of its own: a fork's copy of its
    parent's conversation. Taken as is, instructions included, so the prompt-cache prefix stays the parent's."""
    tools: list[ToolPattern] | None = None
    """The tool ceiling (Lillecarl/aid#50): a tool whose whole name no pattern matches is never offered to the
    model, whatever its mode; aid's and plugins' tools included. None offers every tool. Agents never set it."""

    def offers(self, tool_name: str) -> bool:
        return self.tools is None or any(re.fullmatch(pattern, tool_name) for pattern in self.tools)

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if sum(source is not None for source in (self.agent, self.target, self.instructions)) != 1:
            raise ValueError("set exactly one of agent, target and instructions")
        if self.instructions is not None and (self.model is None or self.tools is None):
            raise ValueError("a spec with instructions sets model and tools")
        return self

    @property
    def runs(self) -> str:
        """What the session runs, for people: the agent's name, the target, or "instructions"."""
        return self.agent or self.target or "instructions"


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
    tool_permissions: dict[str, PermissionMode] = Field(default_factory=dict[str, PermissionMode])
    """Modes by Claude Code tool name, as Claude's own rules at launch, as `ClaudeSdkSpec.tool_permissions`: Claude
    asks an ask tool in its pane, and aid's people through the PermissionRequest hook. A change restarts Claude."""


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
    mcp_servers: McpServers = Field(default_factory=list[McpServer])
    """Beside aid's own (`aid.mcp.session_servers`), in the `--mcp-config` file each prompt's process reads."""
    inherit_env: bool = True
    permission: PermissionMode = PermissionMode.DENY
    """How aid answers what Claude Code asks (`can_use_tool`) about a tool `tool_permissions` does not name."""
    tool_permissions: dict[str, PermissionMode] = Field(default_factory=dict[str, PermissionMode])
    """Modes by Claude Code tool name, as Claude's own rules: allow and deny as `--allowedTools` and
    `--disallowedTools` (a denied tool leaves the model's list), ask as a `permissions.ask` setting, which Claude
    asks aid and aid asks a person. A tool it does not name follows Claude's own defaults."""
    permission_timeout: Annotated[float, Field(gt=0)] = 1800
    """Seconds an `ask` request waits for a person before it is refused."""


type AgentSpec = Annotated[AcpSpec | NativeSpec | ClaudeTtySpec | ClaudeSdkSpec, Field(discriminator="kind")]

AgentSpecAdapter: TypeAdapter[AgentSpec] = TypeAdapter(AgentSpec)
