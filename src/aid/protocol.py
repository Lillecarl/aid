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


type Request = Annotated[
    CreateSession | ListSessions | Prompt | Cancel | StopSession | DeleteSession,
    Field(discriminator="op"),
]


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


type Reply = Annotated[Event | Done | Failure | Hello, Field(discriminator="reply")]

RequestAdapter: TypeAdapter[Request] = TypeAdapter(Request)
ReplyAdapter: TypeAdapter[Reply] = TypeAdapter(Reply)


def encode(message: _Message) -> bytes:
    return message.model_dump_json().encode()


def decode_request(data: bytes) -> Request:
    return RequestAdapter.validate_json(data)


def decode_reply(data: bytes) -> Reply:
    return ReplyAdapter.validate_json(data)
