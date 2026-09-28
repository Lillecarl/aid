"""Agent specifications: what a session runs, independent of how it is hosted."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


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


class AcpSpec(_Spec):
    """An external ACP agent, such as `claude-agent-acp` or `opencode acp`."""

    kind: Literal[AgentKind.ACP] = AgentKind.ACP
    command: Annotated[list[str], Field(min_length=1)]
    inherit_env: bool = True
    permission: PermissionMode = PermissionMode.DENY


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


type AgentSpec = Annotated[AcpSpec | PydanticAISpec | ClaudeTtySpec, Field(discriminator="kind")]

AgentSpecAdapter: TypeAdapter[AgentSpec] = TypeAdapter(AgentSpec)
