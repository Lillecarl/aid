"""The Anthropic Messages API's streaming wire, typed to the fields aid sends and reads (#65).

The request's core is typed; a session's other settings (`thinking`, `output_config`, `metadata`, …) pass through
as body fields for the server to check, as the SDK's keyword arguments did. `extra_headers` (betas) and
`extra_body` keep the SDK's meaning.
"""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

DEFAULT_BASE: Final = "https://api.anthropic.com"
VERSION: Final = "2023-06-01"
KEY_HEADER: Final = "x-api-key"
OVERLOADED: Final = frozenset({"overloaded_error", "api_error", "rate_limit_error"})
"""An `error` event's types that mean "send again later", as their HTTP statuses (529, 500, 429) do."""


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TextBlock(_Wire):
    type: Literal["text"] = "text"
    text: str


class ImageSource(_Wire):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class FileSource(_Wire):
    """An image uploaded through the Files API (`files_url`): sent by id, not resent as bytes."""

    type: Literal["file"] = "file"
    file_id: str


class ImageBlock(_Wire):
    type: Literal["image"] = "image"
    source: ImageSource | FileSource


class ThinkingBlock(_Wire):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class RedactedThinkingBlock(_Wire):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class ToolUseBlock(_Wire):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, JsonValue] = {}


class ToolResultBlock(_Wire):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool | None = None


type Block = Annotated[
    TextBlock | ImageBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(_Wire):
    role: Literal["user", "assistant"]
    content: list[Block]


class ToolDefinition(_Wire):
    name: str
    description: str = ""
    input_schema: JsonValue


class Request(BaseModel):
    """The body. Extra fields are the session's settings, passed through."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    max_tokens: int
    stream: Literal[True] = True
    system: list[TextBlock] | None = None
    tools: list[ToolDefinition] | None = None

    def body(self) -> bytes:
        return self.model_dump_json(exclude_none=True).encode()


class Usage(_Wire):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class StartedMessage(_Wire):
    model: str = ""
    usage: Usage = Usage()


class MessageStart(_Wire):
    type: Literal["message_start"]
    message: StartedMessage


class MessageDelta(_Wire):
    stop_reason: str | None = None


class MessageDeltaEvent(_Wire):
    type: Literal["message_delta"]
    delta: MessageDelta = MessageDelta()
    usage: Usage = Usage()


class UnknownBlock(_Wire):
    type: str


class BlockStart(_Wire):
    type: Literal["content_block_start"]
    index: int
    content_block: TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock | UnknownBlock


class TextDelta(_Wire):
    type: Literal["text_delta"]
    text: str


class ThinkingDelta(_Wire):
    type: Literal["thinking_delta"]
    thinking: str


class SignatureDelta(_Wire):
    type: Literal["signature_delta"]
    signature: str


class InputJsonDelta(_Wire):
    type: Literal["input_json_delta"]
    partial_json: str = ""


class UnknownDelta(_Wire):
    type: str


class BlockDelta(_Wire):
    type: Literal["content_block_delta"]
    index: int
    delta: TextDelta | ThinkingDelta | SignatureDelta | InputJsonDelta | UnknownDelta


class BlockStop(_Wire):
    type: Literal["content_block_stop"]
    index: int


class ErrorDetail(_Wire):
    type: str = ""
    message: str = ""


class ErrorEvent(_Wire):
    """Sent mid-stream (or as a refusal's body): `overloaded_error` and the like."""

    type: Literal["error"]
    error: ErrorDetail = ErrorDetail()


class Other(_Wire):
    """`ping`, `message_stop` and event types aid does not know yet."""

    type: str


type Event = MessageStart | MessageDeltaEvent | BlockStart | BlockDelta | BlockStop | ErrorEvent | Other

_EVENT: Final = TypeAdapter[Event](
    MessageStart | MessageDeltaEvent | BlockStart | BlockDelta | BlockStop | ErrorEvent | Other
)


def event(data: str) -> Event:
    """One SSE data payload. The union tries the known types first: `Other` takes what none of them matches."""
    return _EVENT.validate_json(data)


def error_of(body: str) -> ErrorDetail | None:
    try:
        return ErrorEvent.model_validate_json(body).error
    except ValidationError:
        return None


def messages_url(base: str | None) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/v1/messages"


def files_url(base: str | None) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/v1/files"


class StoredFile(_Wire):
    """The Files API's answer to an upload. Measured 2026-11-13: no beta header needed; `expires_at` null."""

    id: str


MISSING_FILE: Final = re.compile(r"File `([^`]+)` not found")
"""A request naming a deleted or unknown file id: 404 `not_found_error`, "File `file_…` not found." (measured)."""
