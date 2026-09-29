"""Messages on the daemon's zmq sockets.

Clients talk to the daemon, and the daemon talks to workers, with the same
request models. A worker answers with the same reply models the daemon
forwards to the client, matched by request `id`. Every zmq message carries
one JSON-encoded model as its last frame.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

from aid.spec import AgentKind, AgentSpec

PROTOCOL_VERSION = 1


class AidError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _request_id() -> str:
    return uuid.uuid4().hex


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Request(_Message):
    id: str = Field(default_factory=_request_id)


class CreateSession(_Request):
    op: Literal["create"] = "create"
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    spec: AgentSpec


class ListSessions(_Request):
    op: Literal["list"] = "list"


class Prompt(_Request):
    op: Literal["prompt"] = "prompt"
    session: str
    text: str


class Cancel(_Request):
    op: Literal["cancel"] = "cancel"
    session: str


class StartSession(_Request):
    """Start a stopped session's worker without a turn; for interactive Claude, its pane. A running one is left be."""

    op: Literal["start"] = "start"
    session: str


class StopSession(_Request):
    """Stop the worker and keep the session's state, so the next prompt starts it again."""

    op: Literal["stop"] = "stop"
    session: str


class DeleteSession(_Request):
    op: Literal["delete"] = "delete"
    session: str


class ListAgents(_Request):
    """The `aid.PydanticAgent`s on the daemon's agents path."""

    op: Literal["agents"] = "agents"


class GetHistory(_Request):
    """A page of a session's history: the newest entries before `before`, the oldest after `after`, or the
    newest of all when neither is given."""

    op: Literal["history"] = "history"
    session: str
    before: Annotated[int, Field(ge=0)] | None = None
    after: Annotated[int, Field(ge=-1)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100


class SendMessage(_Request):
    """A message to another session. The daemon wakes it: a turn of its own for ACP and pydantic-ai, a channel
    event for interactive Claude. `sender` is a session name, or None for a person."""

    op: Literal["send"] = "send"
    to: str
    text: str
    sender: str | None = None


class ReceiveMessages(_Request):
    """The messages waiting for an interactive Claude session, taken out of its mailbox. Waits up to `wait`
    seconds for one to arrive; an empty list means none did. Only the session's channel server asks."""

    op: Literal["receive"] = "receive"
    session: str
    wait: Annotated[float, Field(ge=0, le=60)] = 25


class GetStatus(_Request):
    op: Literal["status"] = "status"
    session: str


class GetScreen(_Request):
    """What interactive Claude's pane shows now, as HTML. The daemon forwards it to the session's worker."""

    op: Literal["screen"] = "screen"
    session: str
    stylesheet: bool = False
    """Also return the stylesheet the HTML is written against, with the pane's own colours."""
    since: int | None = None
    """A revision already drawn: answer once the pane has left it, or after `wait` seconds with it unchanged."""
    wait: Annotated[float, Field(gt=0, le=60)] = 25


class GetPane(_Request):
    """Where interactive Claude's pane lives: the pymux socket and pane id, for a relay that streams it."""

    op: Literal["pane"] = "pane"
    session: str


type Request = Annotated[
    GetStatus
    | GetScreen
    | GetPane
    | CreateSession
    | ListSessions
    | Prompt
    | Cancel
    | StartSession
    | StopSession
    | DeleteSession
    | ListAgents
    | GetHistory
    | GetSummary
    | SendMessage
    | ReceiveMessages,
    Field(discriminator="op"),
]


class AgentInfo(_Message):
    name: str
    description: str
    module: str


class ToolInfo(_Message):
    name: str
    description: str
    module: str


class AgentCatalog(_Message):
    agents: list[AgentInfo]
    tools: list[ToolInfo] = Field(default_factory=list[ToolInfo])
    problems: list[str]


class TextDelta(_Message):
    type: Literal["text"] = "text"
    text: str


class ThoughtDelta(_Message):
    type: Literal["thought"] = "thought"
    text: str


TOOL_TEXT_LIMIT: Final = 20_000
"""Characters of a tool's output, or of a side of a diff, that an event carries; history keeps every event."""


def clip(text: str) -> str:
    if len(text) <= TOOL_TEXT_LIMIT:
        return text
    return f"{text[:TOOL_TEXT_LIMIT]}\n… {len(text) - TOOL_TEXT_LIMIT} more characters"


def to_json(value: object) -> JsonValue:
    """A tool's arguments as JSON; what JSON cannot hold becomes its `str`."""
    return cast("JsonValue", to_jsonable_python(value, fallback=str))


def tool_text(value: object) -> str | None:
    """A tool's result as text: a string as it is, anything else as indented JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(to_json(value), indent=2)


class ToolDiff(_Message):
    path: str
    old: str | None = None
    """None for a new file."""
    new: str


class ToolCall(_Message):
    """A tool call starting, or an update to it: fields left empty keep what an earlier event of the call said."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    title: str | None = None
    kind: str | None = None
    status: str | None = None
    input: JsonValue = None
    output: str | None = None
    diffs: list[ToolDiff] = Field(default_factory=list[ToolDiff])
    paths: list[str] = Field(default_factory=list[str])
    """Files the call touches, `path` or `path:line`."""


class Cost(_Message):
    amount: float
    currency: str


class Usage(_Message):
    """Tokens a turn used, as its agent reports them; sent just before the turn's Output. None where the agent
    says nothing.

    What "the turn" covers differs by agent (measured 2026-09):
    - claude-agent-acp 0.75.1: every model call of the turn, subagents included.
    - opencode 1.x: the turn's last model response only.
    - pydantic-ai: the run, every request.
    - claude-tty: the main conversation's messages, each counted once; subagents write transcripts of their own,
      which aid does not read.
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
    session_cost: Cost | None = None
    """The agent's own running total for its session, not this turn's share: ACP's `usage_update` cost is
    cumulative. It restarts with the agent's session."""


class Output(_Message):
    """Final event of a prompt. `output` is text for ACP and the agent's typed output for pydantic-ai."""

    type: Literal["output"] = "output"
    output: JsonValue
    stop_reason: str


type SessionEvent = Annotated[TextDelta | ThoughtDelta | ToolCall | Usage | Output, Field(discriminator="type")]


class Started(_Message):
    """A worker started, and what its backend said about itself. The daemon records one per start."""

    type: Literal["started"] = "started"
    pid: int
    agent_session: str | None = None
    """The agent's own session id: ACP's, or Claude Code's (its transcript's name). None for pydantic-ai."""
    resumed: bool = False
    """It carried on an earlier agent session rather than starting one."""
    agent: str | None = None
    """The agent's name and version where it reports them, such as `claude-agent-acp 0.75.1`."""
    model: str | None = None
    """The model when the session started, where the agent reports one. A turn's Usage names what it used."""


class PromptEntry(_Message):
    type: Literal["prompt"] = "prompt"
    text: str


class TurnError(_Message):
    """A prompt that ended in a Failure rather than an Output."""

    type: Literal["error"] = "error"
    code: str
    message: str


class MessageEntry(_Message):
    """A message another session, or a person, sent to this one."""

    type: Literal["message"] = "message"
    sender: str | None
    text: str


type HistoryItem = Annotated[
    PromptEntry | MessageEntry | Started | TextDelta | ThoughtDelta | ToolCall | Usage | Output | TurnError,
    Field(discriminator="type"),
]


class HistoryEntry(_Message):
    seq: int
    """Position in the session's history, from 0. Pages are asked for by it."""
    at: float
    """Unix time the entry was written."""
    turn: str
    """The id of the prompt request this entry belongs to."""
    item: HistoryItem


class HistoryPage(_Message):
    entries: list[HistoryEntry]
    has_older: bool
    has_newer: bool
    total: int


class FileActivity(_Message):
    path: str
    """As the agent named it: absolute, or relative to the session's directory."""
    reads: int = 0
    writes: int = 0
    last_seq: int
    """The history entry of its latest tool call."""


class SessionSummary(_Message):
    """A session's whole history, totalled. Token counts sum each turn's Usage, so they cover what the agents
    reported and nothing more (see Usage)."""

    turns: int = 0
    starts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thought_tokens: int = 0
    requests: int = 0
    models: dict[str, int] = Field(default_factory=dict[str, int])
    """Turns per model."""
    cost: dict[str, float] = Field(default_factory=dict[str, float])
    """Per currency: the last cost each agent session reported, summed over the agent sessions."""
    agent_sessions: list[str] = Field(default_factory=list[str])
    """Oldest first."""
    files: list[FileActivity] = Field(default_factory=list[FileActivity])
    """Most recently touched first."""


class GetSummary(_Request):
    op: Literal["summary"] = "summary"
    session: str


class SessionInfo(_Message):
    name: str
    kind: AgentKind
    running: bool


class SessionStatus(_Message):
    """What a session is doing now, and what it runs. Never env values or MCP headers: those hold credentials."""

    name: str
    kind: AgentKind
    running: bool
    busy: bool
    """A turn is running, a client's or a wake."""
    pending: int
    """Messages waiting: for the running turn to end, or for interactive Claude's channel to take them."""
    pid: int | None
    cwd: str
    runs: str
    """The agent command, or the pydantic-ai agent's name or target."""
    mcp_servers: list[str]
    aid_tools: bool
    agent_session: str | None = None
    """From the last start (`Started`); None before the first."""
    agent: str | None = None
    model: str | None = None
    """What the latest turn used, since this daemon started; else the model at the last start."""


class PaneAddress(_Message):
    socket: str
    """The pymux server's unix socket. Whoever can open it can type into every pane on it."""
    pane: str
    """The pane id, such as `%1002`."""


class PaneView(_Message):
    revision: int
    """pymux's revision of the pane, read before `html` was drawn: `html` is at least this new. Compare it, never
    order it; -1 from a pymux too old to count."""
    html: str
    """A `<pre class="pyte-screen">` of the visible rows, drawn by pymux. No cursor yet."""
    stylesheet: str | None = None
    overlay: str | None = None
    """A pymux mode drawn over the pane, such as copy mode, which `html` does not show."""


class Event(_Message):
    reply: Literal["event"] = "event"
    id: str
    event: SessionEvent


class Done(_Message):
    reply: Literal["done"] = "done"
    id: str
    data: JsonValue = None


class Failure(_Message):
    reply: Literal["failure"] = "failure"
    id: str
    code: str
    message: str


class Hello(_Message):
    """First message from a worker, once its backend is ready for prompts."""

    reply: Literal["hello"] = "hello"
    started: Started


class StartFailed(_Message):
    """Instead of Hello, from a worker whose backend did not start. It exits right after."""

    reply: Literal["start_failed"] = "start_failed"
    message: str


class Observed(_Message):
    """From a worker, unasked: an entry of a turn nobody sent through aid, such as a prompt typed into interactive
    Claude's pane. The worker names the turn; the daemon records the entry under it."""

    reply: Literal["observed"] = "observed"
    turn: str
    item: HistoryItem


type Reply = Annotated[Event | Done | Failure | Hello | StartFailed | Observed, Field(discriminator="reply")]

RequestAdapter: TypeAdapter[Request] = TypeAdapter(Request)
ReplyAdapter: TypeAdapter[Reply] = TypeAdapter(Reply)


def encode(message: _Message) -> bytes:
    return message.model_dump_json().encode()


def decode_request(data: bytes) -> Request:
    return RequestAdapter.validate_json(data)


def decode_reply(data: bytes) -> Reply:
    return ReplyAdapter.validate_json(data)
