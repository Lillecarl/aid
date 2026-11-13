"""OpenAI's Responses API (`/v1/responses`), typed to the fields aid sends and reads (#70).

Requests are TypedDicts aid builds; a session's other settings pass through as body fields. aid sends stateless
requests (`store: false`): its own history is the conversation, and reasoning goes back as `encrypted_content`.
Measured 2026-11-13 on gpt-5-nano, gpt-4o-mini and gpt-4.1-mini:
- a function tool without `strict` is strict (the echo says `strict: true`), so aid sends `strict: false`;
- a reasoning item replays with or without its `id`, and a call without its reasoning item is accepted;
- `include: ["reasoning.encrypted_content"]` and `reasoning.summary` are accepted by models that do not reason;
- `function_call_output.output` takes `input_image` parts, by data URL or by file id;
- `input_tokens` counts cached tokens inside it (`input_tokens_details.cached_tokens`).
"""

from __future__ import annotations

import re
from typing import Final, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from aid.chat_api import DEFAULT_BASE

INCLUDE_REASONING: Final = "reasoning.encrypted_content"
TRANSIENT: Final = frozenset({"server_error", "rate_limit_exceeded", "server_is_overloaded", "slow_down"})
"""Error codes in a stream (`error`, `response.failed`) that mean "send again later"."""


class InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class InputImage(TypedDict):
    type: Literal["input_image"]
    detail: Literal["auto", "low", "high"]
    image_url: NotRequired[str]
    """A `data:` URL or a link."""
    file_id: NotRequired[str]
    """An upload through the Files API (`files_url`)."""


type InputContent = InputText | InputImage


class Message(TypedDict):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[InputContent]


class FunctionCall(TypedDict):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionCallOutput(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str | list[InputContent]


class SummaryText(TypedDict):
    type: Literal["summary_text"]
    text: str


class Reasoning(TypedDict):
    type: Literal["reasoning"]
    summary: list[SummaryText]
    encrypted_content: str


type Item = Message | FunctionCall | FunctionCallOutput | Reasoning


class FunctionTool(TypedDict):
    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, JsonValue]
    strict: bool


class _Read(BaseModel):
    model_config = ConfigDict(extra="allow")


class InputDetails(_Read):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None


class OutputDetails(_Read):
    reasoning_tokens: int | None = None


class Usage(_Read):
    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_details: InputDetails | None = None
    output_tokens_details: OutputDetails | None = None


class Incomplete(_Read):
    reason: str | None = None


class Error(_Read):
    code: str | None = None
    message: str = ""


class Response(_Read):
    model: str | None = None
    status: str | None = None
    usage: Usage | None = None
    incomplete_details: Incomplete | None = None
    error: Error | None = None


class OutputItem(_Read):
    type: str
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    encrypted_content: str | None = None


class ResponseEvent(_Read):
    """`response.created`, `.completed`, `.incomplete`, `.failed`: the response as it stands."""

    type: Literal["response.created", "response.completed", "response.incomplete", "response.failed"]
    response: Response


class ItemEvent(_Read):
    type: Literal["response.output_item.added", "response.output_item.done"]
    output_index: int
    item: OutputItem


class TextDelta(_Read):
    type: Literal["response.output_text.delta"]
    output_index: int
    delta: str


class ArgumentsDelta(_Read):
    type: Literal["response.function_call_arguments.delta"]
    output_index: int
    delta: str


class SummaryDelta(_Read):
    type: Literal["response.reasoning_summary_text.delta"]
    output_index: int
    summary_index: int = 0
    delta: str


class SummaryPartAdded(_Read):
    type: Literal["response.reasoning_summary_part.added"]
    output_index: int
    summary_index: int = 0


class ErrorEvent(_Read):
    type: Literal["error"]
    code: str | None = None
    message: str = ""


class Other(_Read):
    """Events aid reads nothing from (`response.in_progress`, `*.done`, …) and types it does not know yet."""

    type: str


type Event = (
    ResponseEvent | ItemEvent | TextDelta | ArgumentsDelta | SummaryDelta | SummaryPartAdded | ErrorEvent | Other
)

_EVENT: Final = TypeAdapter[Event](
    ResponseEvent | ItemEvent | TextDelta | ArgumentsDelta | SummaryDelta | SummaryPartAdded | ErrorEvent | Other
)


def event(data: str) -> Event:
    """One SSE data payload. The union tries the known types first: `Other` takes what none of them matches."""
    return _EVENT.validate_json(data)


def responses_url(base: str | None) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/responses"


def files_url(base: str | None) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/files"


FILE_PURPOSE: Final = "vision"
"""Measured 2026-11-13: `vision` and `user_data` uploads both work as `input_image.file_id`."""


class StoredFile(_Read):
    id: str


MISSING_FILES: Final = re.compile(r"Files \[([^\]]*)\] were not found")
"""A request naming a deleted or unknown file id: 404, "Files [file-…] were not found" (measured)."""


def missing_files(body: str) -> set[str]:
    return {name.strip() for found in MISSING_FILES.findall(body) for name in found.split(",") if name.strip()}
