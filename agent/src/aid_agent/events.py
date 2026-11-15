"""What a run tells its caller as it goes: text, thinking, tool calls, usage and the final output. aid's protocol
carries the same classes on its sockets."""

from __future__ import annotations

from typing import Final, Literal, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue
from pydantic_core import to_jsonable_python

PATH_KEYS: Final = ("file_path", "path", "notebook_path")
"""Argument names that carry a file a call touches, across Claude Code's tools, common MCP servers and aid.coding: aid's
web UI's gist() in web/src/lib/tools.ts reads the same names."""
TOOL_TEXT_LIMIT: Final = 20_000
"""Characters of a tool's output, or of a side of a diff, that an event carries; history keeps every event."""


class Event(BaseModel):
    # The last: a field with a default is always sent, so the page's generated types (aid.schema) require it.
    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)


def clip(text: str) -> str:
    if len(text) <= TOOL_TEXT_LIMIT:
        return text
    return f"{text[:TOOL_TEXT_LIMIT]}\n… {len(text) - TOOL_TEXT_LIMIT} more characters"


def to_json(value: object) -> JsonValue:
    """A tool's arguments as JSON; what JSON cannot hold becomes its `str`."""
    return cast(JsonValue, to_jsonable_python(value, fallback=str))


def tool_paths(input: JsonValue) -> list[str]:
    """Files a call touches, from its arguments."""
    if not isinstance(input, dict):
        return []
    return [p for key in PATH_KEYS if isinstance(p := input.get(key), str)]


class TextDelta(Event):
    type: Literal["text"] = "text"
    text: str


class ThoughtDelta(Event):
    type: Literal["thought"] = "thought"
    text: str


class ToolDiff(Event):
    path: str
    old: str | None = None
    """None for a new file."""
    new: str


class AttachmentRef(Event):
    """An attachment a prompt carried or a tool returned, its bytes in an `AttachmentStore` by `sha256`."""

    media_type: str
    sha256: str


class ToolCall(Event):
    """A tool call starting, or an update to it: fields left empty keep what an earlier event of the call said."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool: str | None = None
    """The tool's name, as permission rules name it; None where the agent names none (ACP)."""
    title: str | None = None
    kind: str | None = None
    status: str | None = None
    input: JsonValue = None
    output: str | None = None
    diffs: list[ToolDiff] = Field(default_factory=list[ToolDiff])
    paths: list[str] = Field(default_factory=list[str])
    """Files the call touches, `path` or `path:line`."""
    attachments: list[AttachmentRef] = Field(
        default_factory=list[AttachmentRef], validation_alias=AliasChoices("attachments", "images")
    )
    """What the call returned to the model beside its text (native: `aid_agent.attachments.ToolAttachment`); `images` in
    transcripts before #72."""


class Cost(Event):
    amount: float
    currency: str


class Usage(Event):
    """Tokens a turn used, as its agent reports them; sent just before the turn's Output. None where the agent
    says nothing.

    What "the turn" covers differs by agent (measured 2026-09):
    - claude-agent-acp 0.75.1: every model call of the turn, subagents included.
    - opencode 1.x: the turn's last model response only.
    - Native: the run, every request. `input_tokens` follows the provider: Chat Completions counts cache reads
      in it, Anthropic's Messages API does not.
    - claude-tty: the main conversation's messages, each counted once; subagents write transcripts of their own,
      which aid does not read.

    `context_used`/`context_size` come from the agent's own report for ACP and claude-tty. The native backend sets them
    itself: `context_size` is the session's context gauge (`max_context`, half the model window by default) and
    `context_used` is aid's rough estimate of the send view, since the engine reports no window of its own.
    """

    type: Literal["usage"] = "usage"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    thought_tokens: int | None = None
    """Included in output_tokens where the agent bills them together (Claude)."""
    requests: int | None = None
    """Model calls in the turn."""
    models: list[str] = Field(default_factory=list[str])
    context_used: int | None = None
    """Tokens in the context window at the turn's end."""
    context_size: int | None = None
    agent: str | None = None
    """Set for a subagent's tokens (interactive Claude): its type, or `subagent` where Claude names none. They
    are the subagent's own, recorded when it stops, and not in the turn's Usage. A fork's turn, recorded in its
    parent after the fork's answer, is `fork <name>`."""
    session_cost: Cost | None = None
    """The agent's own running total for its session, not this turn's share: ACP's `usage_update` cost is
    cumulative. It restarts with the agent's session."""
    cost: Cost | None = None
    """This turn's own cost, where aid prices the tokens itself (native, from models.dev: `aid_models.prices`).
    Turns add up; None is unknown, not free."""


class Output(Event):
    """Final event of a prompt. `output` is text for ACP and the agent's typed output for native."""

    type: Literal["output"] = "output"
    output: JsonValue
    stop_reason: str
