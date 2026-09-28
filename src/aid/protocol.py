"""Messages on the daemon's zmq sockets.

Clients talk to the daemon, and the daemon talks to workers, with the same
request models. A worker answers with the same reply models the daemon
forwards to the client, matched by request `id`. Every zmq message carries
one JSON-encoded model as its last frame.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

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


type Request = Annotated[
    CreateSession
    | ListSessions
    | Prompt
    | Cancel
    | StopSession
    | DeleteSession
    | ListAgents
    | GetHistory
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


class ToolCall(_Message):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    title: str | None = None
    kind: str | None = None
    status: str | None = None


class Output(_Message):
    """Final event of a prompt. `output` is text for ACP and the agent's typed output for pydantic-ai."""

    type: Literal["output"] = "output"
    output: JsonValue
    stop_reason: str


type SessionEvent = Annotated[TextDelta | ThoughtDelta | ToolCall | Output, Field(discriminator="type")]


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
    PromptEntry | MessageEntry | TextDelta | ThoughtDelta | ToolCall | Output | TurnError, Field(discriminator="type")
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


class SessionInfo(_Message):
    name: str
    kind: AgentKind
    running: bool


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
    pid: int


class StartFailed(_Message):
    """Instead of Hello, from a worker whose backend did not start. It exits right after."""

    reply: Literal["start_failed"] = "start_failed"
    message: str


type Reply = Annotated[Event | Done | Failure | Hello | StartFailed, Field(discriminator="reply")]

RequestAdapter: TypeAdapter[Request] = TypeAdapter(Request)
ReplyAdapter: TypeAdapter[Reply] = TypeAdapter(Reply)


def encode(message: _Message) -> bytes:
    return message.model_dump_json().encode()


def decode_request(data: bytes) -> Request:
    return RequestAdapter.validate_json(data)


def decode_reply(data: bytes) -> Reply:
    return ReplyAdapter.validate_json(data)
