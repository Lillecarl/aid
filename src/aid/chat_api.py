"""OpenAI's Chat Completions and embeddings wire, typed to the fields aid sends and reads (#65).

Many servers speak it (OpenAI, DeepSeek, OpenRouter, OpenCode, vLLM, …) and each adds fields: requests are
TypedDicts aid builds, and the session's other settings pass through as body fields; responses keep unknown fields
(`model_extra`), where providers put reasoning (`reasoning_content`, `reasoning`) and counts
(`prompt_cache_hit_tokens`, `cost`).
"""

from __future__ import annotations

from typing import Final, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, JsonValue

DEFAULT_BASE: Final = "https://api.openai.com/v1"
DONE: Final = "[DONE]"
"""The data of the stream's last event: no JSON."""


class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, JsonValue]


class FunctionTool(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class CalledFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: CalledFunction


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: str
    """A `data:` URL or a link."""


class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class FileData(TypedDict):
    filename: str
    file_data: str
    """A `data:` URL."""


class FileContent(TypedDict):
    """A PDF (#72). OpenAI and OpenRouter take it (measured); other servers' models.dev entries decide."""

    type: Literal["file"]
    file: FileData


type UserContent = TextContent | ImageContent | FileContent


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[UserContent]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]
    """Reasoning text beside the content, as DeepSeek and others take it back."""


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


type Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class JsonSchemaFormat(TypedDict):
    name: str
    schema: dict[str, JsonValue]


class ResponseFormatJsonSchema(TypedDict):
    type: Literal["json_schema"]
    json_schema: JsonSchemaFormat


class ResponseFormatJsonObject(TypedDict):
    type: Literal["json_object"]


type ResponseFormat = ResponseFormatJsonSchema | ResponseFormatJsonObject


class _Read(BaseModel):
    model_config = ConfigDict(extra="allow")


class PromptDetails(_Read):
    cached_tokens: int | None = None


class CompletionDetails(_Read):
    reasoning_tokens: int | None = None


class Usage(_Read):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tokens_details: PromptDetails | None = None
    completion_tokens_details: CompletionDetails | None = None


class FunctionDelta(_Read):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(_Read):
    index: int = 0
    id: str | None = None
    function: FunctionDelta | None = None


class Delta(_Read):
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class Choice(_Read):
    index: int = 0
    delta: Delta = Delta()
    finish_reason: str | None = None


class ChunkError(_Read):
    """An error some servers send as an event mid-stream (OpenRouter), after a 200."""

    message: str = ""
    code: int | str | None = None
    type: str | None = None


class Chunk(_Read):
    model: str | None = None
    choices: list[Choice] = []
    usage: Usage | None = None
    error: ChunkError | None = None


class ErrorBody(_Read):
    error: ChunkError | None = None


class Embedding(_Read):
    index: int
    embedding: list[float]


class EmbeddingUsage(_Read):
    prompt_tokens: int = 0


class Embeddings(_Read):
    data: list[Embedding] = []
    usage: EmbeddingUsage | None = None


def url(base: str | None, path: str) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/{path}"
