"""The Gemini API's `streamGenerateContent` wire, typed to the fields aid sends and reads (#65).

JSON names are lowerCamel; settings accept either spelling, as the API does. A thought signature is bytes in the
API and base64 in its JSON: aid keeps that string as is.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel

DEFAULT_BASE: Final = "https://generativelanguage.googleapis.com"
API_VERSION: Final = "v1beta"
KEY_HEADER: Final = "x-goog-api-key"
KEY_REASONS: Final = frozenset({"API_KEY_INVALID", "API_KEY_EXPIRED"})
"""`ErrorInfo.reason` of a refused key: Gemini answers 400 INVALID_ARGUMENT, not 401 (measured 2026-11-13)."""


class _Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, serialize_by_alias=True)


class _Settings(_Wire):
    model_config = ConfigDict(extra="forbid")


class FunctionCall(_Wire):
    name: str = ""
    args: dict[str, JsonValue] = {}
    id: str | None = None


class Blob(_Wire):
    mime_type: str
    data: str
    """Base64."""


class FunctionResponsePart(_Wire):
    inline_data: Blob


class FunctionResponse(_Wire):
    name: str
    response: dict[str, JsonValue]
    id: str | None = None
    parts: list[FunctionResponsePart] | None = None
    """The tool's images. Measured 2026-11-13 on gemini-3.1-flash-lite: an image beside the function response, as a
    user part, gets an answer led by junk text ("plot्यूडthought", " Malthirteen"), 5 of 5; nested here, clean."""


class Part(_Wire):
    text: str | None = None
    inline_data: Blob | None = None
    thought: bool | None = None
    thought_signature: str | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None


class Content(_Wire):
    role: str | None = None
    parts: list[Part] = []


class FunctionDeclaration(_Wire):
    name: str
    description: str | None = None
    parameters_json_schema: JsonValue = None


class Tool(_Wire):
    function_declarations: list[FunctionDeclaration]


class ThinkingConfig(_Settings):
    include_thoughts: bool | None = None
    thinking_budget: int | None = None
    thinking_level: str | None = None


class GenerationConfig(_Settings):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    response_mime_type: str | None = None
    response_json_schema: JsonValue = None
    response_logprobs: bool | None = None
    logprobs: int | None = None
    thinking_config: ThinkingConfig | None = None
    media_resolution: str | None = None
    response_modalities: list[str] | None = None


class Settings(GenerationConfig):
    """A session's `model_settings` for Gemini: the generation config's fields, plus these that sit beside it."""

    safety_settings: list[dict[str, JsonValue]] | None = None
    tool_config: dict[str, JsonValue] | None = None
    cached_content: str | None = None

    def generation(self) -> GenerationConfig | None:
        fields = self.model_dump(include=set(GenerationConfig.model_fields), exclude_none=True, by_alias=False)
        return GenerationConfig.model_validate(fields) if fields else None


class Request(_Wire):
    contents: list[Content]
    system_instruction: Content | None = None
    tools: list[Tool] | None = None
    generation_config: GenerationConfig | None = None
    safety_settings: list[dict[str, JsonValue]] | None = None
    tool_config: dict[str, JsonValue] | None = None
    cached_content: str | None = None

    def body(self) -> bytes:
        return self.model_dump_json(exclude_none=True).encode()


class EmbedContentRequest(_Wire):
    model: str
    """`models/<model>`."""
    content: Content
    output_dimensionality: int | None = None


class BatchEmbedRequest(_Wire):
    requests: list[EmbedContentRequest]

    def body(self) -> bytes:
        return self.model_dump_json(exclude_none=True).encode()


class UsageMetadata(_Wire):
    prompt_token_count: int = 0
    cached_content_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int | None = None
    tool_use_prompt_token_count: int = 0


class Candidate(_Wire):
    content: Content | None = None
    finish_reason: str | None = None


class Response(_Wire):
    candidates: list[Candidate] = []
    usage_metadata: UsageMetadata | None = None
    model_version: str | None = None


class ContentEmbedding(_Wire):
    values: list[float]


class BatchEmbedResponse(_Wire):
    embeddings: list[ContentEmbedding] = []
    usage_metadata: UsageMetadata | None = None
    """Measured 2026-11-13: `gemini-embedding-2` sends it, `gemini-embedding-001` does not."""


class ErrorInfo(_Wire):
    reason: str | None = None


class Error(_Wire):
    code: int | None = None
    message: str = ""
    status: str | None = None
    details: list[ErrorInfo] = []


class ErrorBody(_Wire):
    error: Error | None = None


_ERROR_LIST: Final = TypeAdapter(list[ErrorBody])


def error_of(body: str) -> Error | None:
    """The API's error from a refusal's body; it is sometimes a list of one."""
    try:
        parsed = ErrorBody.model_validate_json(body)
    except ValidationError:
        try:
            [parsed] = _ERROR_LIST.validate_json(body)
        except ValueError:
            return None
    return parsed.error


def refused_key(body: str) -> bool:
    error = error_of(body)
    return error is not None and any(item.reason in KEY_REASONS for item in error.details)


def embed_url(base: str | None, model: str) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/{API_VERSION}/models/{model}:batchEmbedContents"


def stream_url(base: str | None, model: str) -> str:
    return f"{(base or DEFAULT_BASE).rstrip('/')}/{API_VERSION}/models/{model}:streamGenerateContent?alt=sse"
