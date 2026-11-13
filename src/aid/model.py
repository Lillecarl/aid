"""Model transports, and streamed response assembly. No agent execution lives here. Every transport is aid's
own client (`aid.wire`, #65): no model SDK."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, cast, runtime_checkable

import anyio
import httpx2
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from aid import anthropic_api, chat_api, gemini, wire
from aid.images import NO_IMAGES, ImageStore, encoded
from aid.messages import (
    ImagePart,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence

    from pydantic.json_schema import JsonSchemaValue

    from aid.agent import Tool
    from aid.images import Encoded


@dataclass
class ModelInfo:
    tools: Sequence[Tool] = ()
    output_type: Any = str
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    images: ImageStore | None = None
    """The bytes of the history's `ImagePart`s. None: a request carrying one fails `image_missing`."""

    async def encoded(self, messages: Sequence[ModelMessage]) -> Encoded:
        return NO_IMAGES if self.images is None else await self.images.encoded(messages)


@dataclass
class RequestUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None
    """None: the provider counts no thinking apart (Anthropic bills it inside `output_tokens`)."""


@dataclass
class StreamDelta:
    index: int = 0
    part: TextPart | ThinkingPart | ToolCallPart | None = None
    usage: RequestUsage | None = None
    model_name: str | None = None
    truncated: bool = False
    """The provider stopped the response at its output token limit."""


class StreamBroken(Exception):
    """The connection failed after the response began. No tool call of it ran, so the request may go again."""


class ModelUnavailable(StreamBroken):
    """The server refused the request or was out of reach before it answered: nothing of it ran."""


TRANSIENT_STATUS: Final = frozenset({408, 409, 429})
"""Statuses below 500 a request may go again after: the ones OpenAI's and Anthropic's SDKs retry."""


def unavailable(error: BaseException) -> bool:
    """A failure to send again later: no connection, a timeout, a 5xx, or a rate limit. Measured 2026-11: a vLLM
    gateway answered 503 "no available server" for hours, past two quick retries."""
    match error:
        case wire.StatusError():
            return error.status >= 500 or error.status in TRANSIENT_STATUS
        case _:
            return False


def rejected_key(error: BaseException) -> bool:
    """The server refused the key: a person must replace it, so sending again cannot help. Measured 2026-11-13 with a
    bogus key: OpenAI, Anthropic, DeepSeek, OpenCode and OpenRouter answer 401; an expired OpenCode key too. Gemini
    answers 400 with a reason (`gemini.refused_key`)."""
    match error:
        case wire.StatusError():
            return error.status == 401
        case _:
            return False


def refused_key(name: str, error: BaseException) -> AidError:
    return AidError("bad_credentials", f"{name}: the server refused the key: {error}")


BROKEN: Final = (httpx2.TransportError,)
"""What a dropped stream raises: every client is httpx2, and a response body raises it bare mid-stream (openai 2.x, measured 2026-10-28: vLLM dropped 7 streams in 5 hours, 61 to 545 s in)."""


class Model(Protocol):
    model_name: str

    def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]: ...
    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse: ...


class Transport(Model, Protocol):
    """A model aid built from a name: it owns its clients, and whoever resolved it closes them."""

    cache_reads_in_prompt: bool
    """Its usage counts cache reads inside the prompt tokens (Chat Completions, Gemini), not apart (Anthropic's
    Messages API): pricing subtracts them once."""

    async def aclose(self) -> None: ...


@runtime_checkable
class Hosted(Protocol):
    """A model that says where its requests go: what a private session (`NativeSpec.private`) needs to run on it."""

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        """The host name of the base URL this model's client uses with `settings`."""
        ...


def check_host(model: object, settings: Mapping[str, JsonValue], hosts: Sequence[str]) -> None:
    """Refuse a model whose requests go to a host not in `hosts`, or that cannot say where they go."""
    if not isinstance(model, Hosted):
        raise AidError("private_host", f"{model_label(model)} cannot say which host it sends to")
    if (host := model.host(settings)) not in hosts:
        raise AidError("private_host", f"{model_label(model)} sends to {host!r}, which is not one of {list(hosts)}")


def model_label(model: object) -> str:
    return str(getattr(model, "model_name", type(model).__name__))


def wire_messages(
    messages: Sequence[ModelMessage], *, reasoning: bool = True, images: Encoded = NO_IMAGES
) -> list[chat_api.Message]:
    """Stored parts as Chat Completions messages, keeping tool identities and, with `reasoning`, reasoning text."""
    result: list[chat_api.Message] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            thoughts = [p for p in message.parts if isinstance(p, ThinkingPart)]
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if not text and not calls:
                # Providers refuse an assistant message with neither (DeepSeek: 400). A failed request could
                # store one, and that stored history must still send.
                continue
            item: chat_api.AssistantMessage = {"role": "assistant", "content": text or None}
            if calls:
                item["tool_calls"] = [
                    {
                        "id": p.tool_call_id,
                        "type": "function",
                        "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
                    }
                    for p in calls
                ]
            if thoughts and reasoning:
                item["reasoning_content"] = "".join(p.content for p in thoughts)
            result.append(item)
        else:
            # A tool message takes text only: a call's image goes in a user message after the request's last tool
            # message, since a user message between them would part the answers from their calls.
            held: list[chat_api.TextContent | chat_api.ImageContent] = []
            prompt: chat_api.UserMessage | None = None
            for part in message.parts:
                if held and not isinstance(part, ToolReturnPart | RetryPromptPart | ImagePart):
                    result.append({"role": "user", "content": list(held)})
                    held.clear()
                if not isinstance(part, ImagePart):
                    prompt = None
                match part:
                    case ImagePart():
                        url = f"data:{part.media_type};base64,{encoded(images, part)}"
                        block: chat_api.ImageContent = {"type": "image_url", "image_url": {"url": url}}
                        if part.tool_call_id is not None:
                            held.append(block)
                        elif prompt is not None:
                            content = prompt["content"]
                            if isinstance(content, str):
                                blocks: list[chat_api.TextContent | chat_api.ImageContent] = [
                                    {"type": "text", "text": content}
                                ]
                                prompt["content"] = content = blocks
                            content.append(block)
                        else:
                            result.append(prompt := {"role": "user", "content": [block]})
                    case SystemPromptPart():
                        result.append({"role": "system", "content": part.content})
                    case UserPromptPart():
                        result.append(prompt := {"role": "user", "content": str(part.content)})
                    case ToolReturnPart():
                        result.append(
                            {"role": "tool", "tool_call_id": part.tool_call_id, "content": part.model_response_str()}
                        )
                    case RetryPromptPart():
                        if part.tool_name:
                            result.append(
                                {"role": "tool", "tool_call_id": part.tool_call_id, "content": part.model_response()}
                            )
                        else:
                            result.append({"role": "user", "content": part.model_response()})
                    case _:
                        raise ValueError("response part in a model request")
            if held:
                tail: chat_api.UserMessage = {"role": "user", "content": held}
                result.append(tail)
    return result


def accumulate(response: ModelResponse, indexes: dict[int, int], delta: StreamDelta) -> None:
    part = delta.part
    if part is None:
        return
    if delta.index not in indexes:
        indexes[delta.index] = len(response.parts)
        response.parts.append(part.model_copy(deep=True))
        return
    previous = response.parts[indexes[delta.index]]
    if isinstance(previous, TextPart | ThinkingPart) and isinstance(part, type(previous)):
        previous.content += part.content
        if isinstance(previous, ThinkingPart) and isinstance(part, ThinkingPart) and part.signature:
            previous.signature = (previous.signature or "") + part.signature
        for key, value in (part.model_extra or {}).items():
            setattr(previous, key, value)
    elif isinstance(previous, ToolCallPart) and isinstance(part, ToolCallPart):
        previous.tool_name += part.tool_name
        previous.tool_call_id = part.tool_call_id or previous.tool_call_id
        previous.args = previous.args_as_json_str() + part.args_as_json_str()
    else:
        raise ValueError("stream changes the type of an existing response part")


@dataclass(frozen=True)
class Provider:
    base_url: str | None
    """None: `chat_api.DEFAULT_BASE` (api.openai.com)."""
    key_env: str
    local: bool = False
    """An unauthenticated local server: a missing key is not an error."""
    defaults: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Request settings the session's `model_settings` override."""
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    """Sent with every request of the client."""
    stream_usage: bool = True
    """The server takes `stream_options.include_usage` (Gemini's refuses it)."""


OPENAI_COMPATIBLE: Final[dict[str, Provider]] = {
    "openai": Provider(None, "OPENAI_API_KEY"),
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", stream_usage=False
    ),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": Provider("https://api.x.ai/v1", "XAI_API_KEY"),
    "together": Provider("https://api.together.xyz/v1", "TOGETHERAI_API_KEY"),
    "fireworks": Provider("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "ollama": Provider("http://localhost:11434/v1", "OLLAMA_API_KEY", local=True),
    "vllm": Provider("http://localhost:8000/v1", "VLLM_API_KEY", local=True),
    "lmstudio": Provider("http://localhost:1234/v1", "LM_STUDIO_API_KEY", local=True),
}
"""`provider/model` prefixes served over Chat Completions, for servers models.dev names no URL for and local ones.
Others resolve through `aid.providers`."""


class ClientSettings(BaseModel):
    """The `model_settings` keys that configure the client; every other key goes into each request."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    api_key: str | None = None
    base_url: str | None = Field(None, validation_alias=AliasChoices("api_base", "base_url"))
    max_retries: int = Field(0, validation_alias=AliasChoices("max_retries", "num_retries"))


def _setting_names(model: type[BaseModel]) -> frozenset[str]:
    names: set[str] = set()
    for name, info in model.model_fields.items():
        alias = info.validation_alias
        if isinstance(alias, AliasChoices):
            names.update(c for c in alias.choices if isinstance(c, str))
        else:
            names.add(name)
    return frozenset(names)


CLIENT_SETTINGS: Final = _setting_names(ClientSettings)
DEFAULT_TIMEOUT: Final = 600
_TIMEOUT: Final = TypeAdapter(float)


def split_name(name: str) -> tuple[str, str]:
    """`provider/model` → (provider, model); a bare name is OpenAI's."""
    provider, _, model = name.partition("/")
    return (provider, model) if model else ("openai", provider)


@dataclass(frozen=True)
class ClientKey:
    base_url: str | None
    api_key: str
    max_retries: int


def client_settings(name: str, settings: Mapping[str, JsonValue]) -> ClientSettings:
    try:
        return ClientSettings.model_validate(settings)
    except ValidationError as e:
        raise AidError("bad_model_settings", f"{name}: {e}") from e


def client_key(
    name: str, settings: Mapping[str, JsonValue], key_env: str | None, base_url: str | None, *, local: bool = False
) -> ClientKey:
    """The client's settings over the provider's: `key_env` and `base_url` apply where the settings name none."""
    client = client_settings(name, settings)
    api_key = client.api_key or (os.environ.get(key_env) if key_env else None)
    if not api_key:
        if not local:
            raise AidError("missing_credentials", f"{name}: set {key_env or 'api_key in model_settings'}")
        api_key = "local"
    return ClientKey(client.base_url or base_url, api_key, client.max_retries)


def _request_settings(settings: Mapping[str, JsonValue], provider: Provider | None = None) -> dict[str, JsonValue]:
    kwargs: dict[str, JsonValue] = {**(provider.defaults if provider else {})}
    kwargs.update((key, value) for key, value in settings.items() if key not in CLIENT_SETTINGS)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return kwargs


_HEADERS: Final = TypeAdapter(dict[str, str])
_BODY: Final = TypeAdapter(dict[str, JsonValue])
RETRY_BACKOFF: Final = 0.5
"""Seconds before a client retry (`max_retries`), doubling. The backend replays past them (`STREAM_RETRIES`)."""


def _extra(name: str, settings: dict[str, JsonValue]) -> tuple[dict[str, str], dict[str, JsonValue]]:
    """Pop the SDK-style `extra_headers` and `extra_body` from a request's settings."""
    try:
        return (
            _HEADERS.validate_python(settings.pop("extra_headers", None) or {}),
            _BODY.validate_python(settings.pop("extra_body", None) or {}),
        )
    except ValidationError as error:
        raise AidError("bad_model_settings", f"{name}: extra_headers/extra_body: {error}") from error


async def open_model_stream(
    name: str,
    http: httpx2.AsyncClient,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    io_timeout: float,
    retries: int,
    refused: Callable[[wire.StatusError], bool] = lambda error: False,
) -> httpx2.Response:
    """Open a model's stream: a refused key → `bad_credentials` (`refused` names a provider's own way to say it), a
    failure to send again later → retried `retries` times, then `ModelUnavailable`; anything else propagates."""
    for attempt in range(retries + 1):
        try:
            return await wire.open_stream(http, url, body, headers, io_timeout)
        except (wire.StatusError, httpx2.TransportError) as error:
            if rejected_key(error) or (isinstance(error, wire.StatusError) and refused(error)):
                raise refused_key(name, error) from error
            if not (isinstance(error, httpx2.TransportError) or unavailable(error)):
                raise
            if attempt == retries:
                raise ModelUnavailable(f"{type(error).__name__}: {error}") from error
            await anyio.sleep(RETRY_BACKOFF * 2**attempt)
    raise AssertionError("unreachable")


class OpenAIChatModel:
    """Chat Completions over aid's own client (`aid.chat_api`, `aid.wire`; #65), for OpenAI and every provider that
    serves the same API. A provider with quirks subclasses it and overrides `request_settings`, `reasoning` or
    `usage`."""

    cache_reads_in_prompt: bool = True

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider` None: the one `OPENAI_COMPATIBLE` names by the prefix, if any."""
        self.model_name = name
        prefix, self._model = split_name(name)
        self._provider = provider or OPENAI_COMPATIBLE.get(prefix)
        self._http: httpx2.AsyncClient | None = None

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self._provider
        if provider is None and client_settings(self.model_name, settings).base_url is None:
            raise AidError(
                "unknown_provider",
                f"no provider {split_name(self.model_name)[0]!r}: use one of {', '.join(OPENAI_COMPATIBLE)}, "
                "anthropic, or set api_base in model_settings",
            )
        return client_key(
            self.model_name,
            settings,
            provider.key_env if provider else None,
            provider.base_url if provider else None,
            local=provider is not None and provider.local,
        )

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return httpx2.URL(chat_api.url(self._key(settings).base_url, "chat/completions")).host

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()

    def request_settings(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Each request's body fields from the session's settings, under the provider's defaults."""
        kwargs = _request_settings(settings, self._provider)
        if self._provider is None or self._provider.stream_usage:
            kwargs.setdefault("stream_options", {"include_usage": True})
        return kwargs

    def reasoning(self, delta: chat_api.Delta) -> str | None:
        """A streamed delta's reasoning text, beside the content: DeepSeek names it `reasoning_content`, OpenRouter
        and vLLM `reasoning` (measured 2026-10-27, vLLM serving glm-5.3-flash)."""
        extra = delta.model_extra or {}
        for key in ("reasoning_content", "reasoning"):
            if isinstance(thought := extra.get(key), str) and thought:
                return thought
        return None

    def usage(self, usage: chat_api.Usage) -> RequestUsage:
        return _openai_usage(usage)

    replays_reasoning: ClassVar[bool] = True
    """Reasoning text goes back beside the content (`reasoning_content`); False for a provider that refuses it."""

    def wire(self, messages: Sequence[ModelMessage], images: Encoded = NO_IMAGES) -> list[chat_api.Message]:
        return wire_messages(messages, reasoning=self.replays_reasoning, images=images)

    def response_format(self, schema: JsonSchemaValue) -> chat_api.ResponseFormat:
        """How a request asks for structured output: the schema itself, where the provider takes one."""
        return {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}

    def schema_note(self, schema: JsonSchemaValue) -> str | None:
        """A system note stating the output schema, for a provider whose JSON mode takes no schema."""
        return None

    def _body(
        self, messages: list[ModelMessage], info: ModelInfo, settings: dict[str, JsonValue], images: Encoded
    ) -> bytes:
        wired = self.wire(messages, images)
        if info.output_type is not str:
            schema = TypeAdapter(info.output_type).json_schema()
            settings["response_format"] = cast(JsonValue, self.response_format(schema))
            if (note := self.schema_note(schema)) is not None:
                # After the leading instructions, so every request keeps the same cached prefix.
                lead = next((i for i, w in enumerate(wired) if w["role"] != "system"), len(wired))
                wired.insert(lead, {"role": "system", "content": note})
        body: dict[str, JsonValue] = {
            **settings,
            "model": self._model,
            "messages": cast(JsonValue, wired),
            "stream": True,
        }
        if info.tools:
            body["tools"] = cast(JsonValue, [tool.definition() for tool in info.tools])
        return json.dumps(body).encode()

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        key = self._key(info.settings)
        settings = self.request_settings(info.settings)
        io_timeout = _TIMEOUT.validate_python(settings.pop("timeout"))
        extra_headers, extra_body = _extra(self.model_name, settings)
        settings.update(extra_body)
        headers = {
            "authorization": f"Bearer {key.api_key}",
            **(self._provider.headers if self._provider else {}),
            **extra_headers,
        }
        body = self._body(messages, info, settings, await info.encoded(messages))
        if self._http is None:
            self._http = wire.client(timeout=DEFAULT_TIMEOUT)
        url = chat_api.url(key.base_url, "chat/completions")
        response = await open_model_stream(self.model_name, self._http, url, body, headers, io_timeout, key.max_retries)
        ids: dict[int, str] = {}
        try:
            async for data in wire.events(response):
                if data == chat_api.DONE:
                    break
                chunk = chat_api.Chunk.model_validate_json(data)
                if chunk.error is not None:
                    raise _chunk_error(self.model_name, chunk.error)
                if chunk.model:
                    yield StreamDelta(model_name=chunk.model)
                if chunk.usage is not None:
                    yield StreamDelta(usage=self.usage(chunk.usage))
                for choice in chunk.choices:
                    if choice.finish_reason == "length":
                        yield StreamDelta(truncated=True)
                    delta = choice.delta
                    if delta.content:
                        yield StreamDelta(0, TextPart(delta.content))
                    if thought := self.reasoning(delta):
                        yield StreamDelta(1, ThinkingPart(thought))
                    for call in delta.tool_calls or []:
                        if call.id:
                            ids[call.index] = call.id
                        function = call.function
                        yield StreamDelta(
                            100 + call.index,
                            ToolCallPart(
                                (function.name if function else None) or "",
                                (function.arguments if function else None) or "",
                                ids.get(call.index, ""),
                            ),
                        )
        except BROKEN as error:
            raise StreamBroken(repr(error)) from error
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _chunk_error(name: str, error: chat_api.ChunkError) -> Exception:
    """An error event after a 200: a 5xx/429 code or an overload sends again; anything else is the model's."""
    code = error.code
    if (isinstance(code, int) and (code >= 500 or code in TRANSIENT_STATUS)) or "overload" in (error.type or ""):
        return ModelUnavailable(f"{code}: {error.message}")
    return AidError("model_error", f"{name}: {code}: {error.message}")


def _openai_usage(usage: chat_api.Usage) -> RequestUsage:
    """`input_tokens` includes cached tokens here, as Chat Completions counts them."""
    prompt = usage.prompt_tokens_details
    cached = (prompt.cached_tokens if prompt else None) or 0
    completion = usage.completion_tokens_details
    return RequestUsage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        cache_read_tokens=cached,
        reasoning_tokens=completion.reasoning_tokens if completion else None,
    )


async def collect(model: Model, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
    response = ModelResponse(parts=[], model_name=model.model_name)
    indexes: dict[int, int] = {}
    async for delta in model.stream(messages, info):
        accumulate(response, indexes, delta)
    return response


ANTHROPIC_KEY_ENV: Final = "ANTHROPIC_API_KEY"
ANTHROPIC_MAX_TOKENS: Final = 32_000
"""`max_tokens` when the settings name none: the Messages API requires one. Every Claude 4 model allows it."""
ANTHROPIC_LEGACY_MAX_TOKENS: Final = 4096
"""The same for `claude-3*` models, whose limits start at 4096."""


def anthropic_messages(
    messages: Sequence[ModelMessage], images: Encoded = NO_IMAGES
) -> tuple[list[anthropic_api.TextBlock], list[anthropic_api.Message]]:
    """Stored parts as the Messages API's (system, messages). Leading system prompts become `system`; a later one
    stays in place as user text. Consecutive same-role parts share one message, so the tool results of one
    response arrive together. Thinking replays only signed or redacted, as Anthropic produced it."""
    system: list[anthropic_api.TextBlock] = []
    turns: list[anthropic_api.Message] = []

    def add(role: Literal["user", "assistant"], block: anthropic_api.Block) -> None:
        if turns and turns[-1].role == role:
            turns[-1].content.append(block)
        else:
            turns.append(anthropic_api.Message(role=role, content=[block]))

    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ThinkingPart() if part.model_extra and "redacted_data" in part.model_extra:
                        add(
                            "assistant",
                            anthropic_api.RedactedThinkingBlock(data=str(part.model_extra["redacted_data"])),
                        )
                    case ThinkingPart() if part.signature:
                        add("assistant", anthropic_api.ThinkingBlock(thinking=part.content, signature=part.signature))
                    case TextPart() if part.content:
                        add("assistant", anthropic_api.TextBlock(text=part.content))
                    case ToolCallPart():
                        try:
                            arguments = part.args_as_dict()
                        except ValueError:
                            arguments = {}
                        add(
                            "assistant",
                            anthropic_api.ToolUseBlock(id=part.tool_call_id, name=part.tool_name, input=arguments),
                        )
                    case _:
                        pass
            continue
        for part in message.parts:
            match part:
                case SystemPromptPart() if not turns:
                    system.append(anthropic_api.TextBlock(text=part.content))
                case SystemPromptPart():
                    add("user", anthropic_api.TextBlock(text=part.content))
                case UserPromptPart():
                    add("user", anthropic_api.TextBlock(text=str(part.content)))
                case ImagePart():
                    source = anthropic_api.ImageSource(media_type=part.media_type, data=encoded(images, part))
                    image = anthropic_api.ImageBlock(source=source)
                    if (answer := _tool_result(turns, part.tool_call_id)) is None:
                        add("user", image)
                    else:
                        if isinstance(answer.content, str):
                            answer.content = [anthropic_api.TextBlock(text=answer.content)]
                        answer.content.append(image)
                case ToolReturnPart():
                    add(
                        "user",
                        anthropic_api.ToolResultBlock(tool_use_id=part.tool_call_id, content=part.model_response_str()),
                    )
                case RetryPromptPart() if part.tool_name:
                    add(
                        "user",
                        anthropic_api.ToolResultBlock(
                            tool_use_id=part.tool_call_id, content=part.model_response(), is_error=True
                        ),
                    )
                case RetryPromptPart():
                    add("user", anthropic_api.TextBlock(text=part.model_response()))
                case _:
                    raise ValueError("response part in a model request")
    return system, turns


def _tool_result(turns: list[anthropic_api.Message], call: str | None) -> anthropic_api.ToolResultBlock | None:
    """The answer to `call` in the last user turn, where its image goes: a tool result may hold images."""
    if call is None or not turns or turns[-1].role != "user":
        return None
    return next(
        (b for b in turns[-1].content if isinstance(b, anthropic_api.ToolResultBlock) and b.tool_use_id == call), None
    )


def anthropic_tool(definition: chat_api.FunctionTool) -> anthropic_api.ToolDefinition:
    function = definition["function"]
    return anthropic_api.ToolDefinition(
        name=function["name"],
        description=function.get("description", ""),
        input_schema=cast(JsonValue, function.get("parameters", {})),
    )


class AnthropicModel:
    """The Messages API over aid's own client (`aid.anthropic_api`, `aid.wire`; #65): signed thinking replayed,
    beta headers (`extra_headers`) and caching passed through. `cache_control` defaults to automatic caching of the
    longest prefix."""

    cache_reads_in_prompt: bool = False

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider`: another server of the Messages API, its base URL and key variable; None is Anthropic's."""
        self.model_name = name
        self._model = split_name(name)[1]
        self._provider = provider
        self._http: httpx2.AsyncClient | None = None

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self._provider
        key_env = provider.key_env if provider else ANTHROPIC_KEY_ENV
        return client_key(self.model_name, settings, key_env, provider.base_url if provider else None)

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return httpx2.URL(anthropic_api.messages_url(self._key(settings).base_url)).host

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()

    def _request(
        self, messages: list[ModelMessage], info: ModelInfo, settings: dict[str, JsonValue], images: Encoded
    ) -> anthropic_api.Request:
        system, wired = anthropic_messages(messages, images)
        settings.setdefault(
            "max_tokens", ANTHROPIC_LEGACY_MAX_TOKENS if self._model.startswith("claude-3") else ANTHROPIC_MAX_TOKENS
        )
        settings.setdefault("cache_control", {"type": "ephemeral"})
        if info.output_type is not str:
            config = settings.get("output_config")
            settings["output_config"] = {
                **(config if isinstance(config, dict) else {}),
                "format": {"type": "json_schema", "schema": TypeAdapter(info.output_type).json_schema()},
            }
        try:
            return anthropic_api.Request.model_validate(
                {
                    **settings,
                    "model": self._model,
                    "messages": wired,
                    "stream": True,
                    "system": system or None,
                    "tools": [anthropic_tool(tool.definition()) for tool in info.tools] or None,
                }
            )
        except ValidationError as error:
            raise AidError("bad_model_settings", f"{self.model_name}: {error}") from error

    async def _open(self, key: ClientKey, body: bytes, headers: dict[str, str], io_timeout: float) -> httpx2.Response:
        if self._http is None:
            self._http = wire.client(timeout=DEFAULT_TIMEOUT)
        url = anthropic_api.messages_url(key.base_url)
        return await open_model_stream(self.model_name, self._http, url, body, headers, io_timeout, key.max_retries)

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        key = self._key(info.settings)
        settings = _request_settings(info.settings)
        io_timeout = _TIMEOUT.validate_python(settings.pop("timeout"))
        extra_headers, extra_body = _extra(self.model_name, settings)
        settings.update(extra_body)
        headers = {
            anthropic_api.KEY_HEADER: key.api_key,
            "anthropic-version": anthropic_api.VERSION,
            **(self._provider.headers if self._provider else {}),
            **extra_headers,
        }
        body = self._request(messages, info, settings, await info.encoded(messages)).body()
        response = await self._open(key, body, headers, io_timeout)
        arguments: dict[int, bool] = {}
        try:
            async for data in wire.events(response):
                match anthropic_api.event(data):
                    case anthropic_api.MessageStart(message=message):
                        yield StreamDelta(model_name=message.model)
                        yield StreamDelta(usage=_anthropic_usage(message.usage))
                    case anthropic_api.MessageDeltaEvent(delta=delta, usage=usage):
                        yield StreamDelta(usage=_anthropic_usage(usage))
                        if delta.stop_reason == "max_tokens":
                            yield StreamDelta(truncated=True)
                    case anthropic_api.BlockStart(index=index, content_block=block):
                        match block:
                            case anthropic_api.TextBlock():
                                yield StreamDelta(index, TextPart(block.text))
                            case anthropic_api.ThinkingBlock():
                                yield StreamDelta(index, ThinkingPart(block.thinking, signature=block.signature))
                            case anthropic_api.RedactedThinkingBlock():
                                yield StreamDelta(index, ThinkingPart("", redacted_data=block.data))
                            case anthropic_api.ToolUseBlock():
                                arguments[index] = False
                                yield StreamDelta(index, ToolCallPart(block.name, "", block.id))
                            case _:
                                pass
                    case anthropic_api.BlockDelta(index=index, delta=delta):
                        match delta:
                            case anthropic_api.TextDelta():
                                yield StreamDelta(index, TextPart(delta.text))
                            case anthropic_api.ThinkingDelta():
                                yield StreamDelta(index, ThinkingPart(delta.thinking))
                            case anthropic_api.SignatureDelta():
                                yield StreamDelta(index, ThinkingPart("", signature=delta.signature))
                            case anthropic_api.InputJsonDelta() if delta.partial_json:
                                arguments[index] = True
                                yield StreamDelta(index, ToolCallPart("", delta.partial_json, ""))
                            case _:
                                pass
                    case anthropic_api.BlockStop(index=index) if arguments.get(index) is False:
                        # A tool without arguments streams no JSON at all.
                        yield StreamDelta(index, ToolCallPart("", "{}", ""))
                    case anthropic_api.ErrorEvent(error=error):
                        if error.type in anthropic_api.OVERLOADED:
                            raise ModelUnavailable(f"{error.type}: {error.message}")
                        raise AidError("model_error", f"{self.model_name}: {error.type}: {error.message}")
                    case _:
                        pass
        except BROKEN as error:
            raise StreamBroken(repr(error)) from error
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _anthropic_usage(usage: anthropic_api.Usage) -> RequestUsage:
    """Anthropic's `input_tokens` excludes cache reads and writes; aid keeps that split."""
    return RequestUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )


GOOGLE_KEY_ENV: Final = "GEMINI_API_KEY"


def _signature(part: TextPart | ToolCallPart) -> str | None:
    signature = (part.model_extra or {}).get("thought_signature")
    return signature if isinstance(signature, str) else None


def google_contents(
    messages: Sequence[ModelMessage], images: Encoded = NO_IMAGES
) -> tuple[list[gemini.Part], list[gemini.Content]]:
    """Stored parts as the Gemini API's (system instruction parts, contents), merged by role as
    `anthropic_messages` does. A thought signature goes back on the part that carried it: Gemini 3 refuses a
    function call without its own. Thinking replays only signed."""
    system: list[gemini.Part] = []
    turns: list[gemini.Content] = []
    ids: dict[str, str | None] = {}

    def add(role: str, part: gemini.Part) -> None:
        if turns and turns[-1].role == role:
            turns[-1].parts.append(part)
        else:
            turns.append(gemini.Content(role=role, parts=[part]))

    def answer(part: ToolReturnPart | RetryPromptPart, response: dict[str, JsonValue]) -> gemini.Part:
        block = gemini.FunctionResponse(name=part.tool_name or "", response=response, id=ids.get(part.tool_call_id))
        return gemini.Part(function_response=block)

    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ThinkingPart() if part.signature:
                        add("model", gemini.Part(text=part.content, thought=True, thought_signature=part.signature))
                    case TextPart() if part.content or part.model_extra:
                        add("model", gemini.Part(text=part.content, thought_signature=_signature(part)))
                    case ToolCallPart():
                        try:
                            arguments = part.args_as_dict()
                        except ValueError:
                            arguments = {}
                        google_id = (part.model_extra or {}).get("google_id")
                        google_id = google_id if isinstance(google_id, str) else None
                        ids[part.tool_call_id] = google_id
                        call = gemini.FunctionCall(name=part.tool_name, args=arguments, id=google_id)
                        add("model", gemini.Part(function_call=call, thought_signature=_signature(part)))
                    case _:
                        pass
            continue
        for part in message.parts:
            match part:
                case SystemPromptPart() if not turns:
                    system.append(gemini.Part(text=part.content))
                case SystemPromptPart():
                    add("user", gemini.Part(text=part.content))
                case UserPromptPart():
                    add("user", gemini.Part(text=str(part.content)))
                case ImagePart():
                    blob = gemini.Blob(mime_type=part.media_type, data=encoded(images, part))
                    add("user", gemini.Part(inline_data=blob))
                case ToolReturnPart():
                    add("user", answer(part, {"output": part.model_response_str()}))
                case RetryPromptPart() if part.tool_name:
                    add("user", answer(part, {"error": part.model_response()}))
                case RetryPromptPart():
                    add("user", gemini.Part(text=part.model_response()))
                case _:
                    raise ValueError("response part in a model request")
    return system, turns


def google_tool(tool: Tool) -> gemini.FunctionDeclaration:
    return gemini.FunctionDeclaration(
        name=tool.name, description=tool.description, parameters_json_schema=tool.arguments.model_json_schema()
    )


class GoogleModel:
    """The Gemini API (`streamGenerateContent`) over aid's own client (`aid.gemini`, `aid.wire`; #65). Thought
    signatures replay on their parts; Gemini assigns call ids only sometimes, so aid makes its own and keeps
    Gemini's beside them."""

    cache_reads_in_prompt: bool = True

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider`: another server of the Gemini API, its base URL and key variable; None is Google AI Studio."""
        self.model_name = name
        self._model = split_name(name)[1]
        self._provider = provider
        self._http: httpx2.AsyncClient | None = None

    def _key(self, settings: Mapping[str, JsonValue]) -> ClientKey:
        provider = self._provider
        key_env = provider.key_env if provider else GOOGLE_KEY_ENV
        return client_key(self.model_name, settings, key_env, provider.base_url if provider else None)

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return httpx2.URL(gemini.stream_url(self._key(settings).base_url, self._model)).host

    async def aclose(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()

    def _request(
        self, messages: list[ModelMessage], info: ModelInfo, settings: dict[str, JsonValue], images: Encoded
    ) -> bytes:
        try:
            chosen = gemini.Settings.model_validate(settings)
        except ValidationError as error:
            raise AidError("bad_model_settings", f"{self.model_name}: {error}") from error
        generation = chosen.generation()
        if info.output_type is not str:
            generation = (generation or gemini.GenerationConfig()).model_copy(
                update={
                    "response_mime_type": "application/json",
                    "response_json_schema": TypeAdapter(info.output_type).json_schema(),
                }
            )
        system, contents = google_contents(messages, images)
        request = gemini.Request(
            contents=contents,
            system_instruction=gemini.Content(parts=system) if system else None,
            tools=[gemini.Tool(function_declarations=[google_tool(tool) for tool in info.tools])]
            if info.tools
            else None,
            generation_config=generation,
            safety_settings=chosen.safety_settings,
            tool_config=chosen.tool_config,
            cached_content=chosen.cached_content,
        )
        return request.body()

    async def _open(self, key: ClientKey, body: bytes, io_timeout: float) -> httpx2.Response:
        if self._http is None:
            self._http = wire.client(timeout=DEFAULT_TIMEOUT)
        headers = {gemini.KEY_HEADER: key.api_key, **(self._provider.headers if self._provider else {})}
        url = gemini.stream_url(key.base_url, self._model)
        return await open_model_stream(
            self.model_name,
            self._http,
            url,
            body,
            headers,
            io_timeout,
            key.max_retries,
            lambda error: gemini.refused_key(error.body),
        )

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        key = self._key(info.settings)
        settings = _request_settings(info.settings, self._provider)
        timeout = _TIMEOUT.validate_python(settings.pop("timeout"))
        body = self._request(messages, info, settings, await info.encoded(messages))
        response = await self._open(key, body, timeout)
        calls = 0
        try:
            async for data in wire.events(response):
                chunk = gemini.Response.model_validate_json(data)
                if chunk.model_version:
                    yield StreamDelta(model_name=chunk.model_version)
                if chunk.usage_metadata is not None:
                    yield StreamDelta(usage=_google_usage(chunk.usage_metadata))
                for candidate in chunk.candidates:
                    if candidate.finish_reason == "MAX_TOKENS":
                        yield StreamDelta(truncated=True)
                    for part in candidate.content.parts if candidate.content else []:
                        extra: dict[str, str] = {}
                        if part.thought_signature:
                            extra["thought_signature"] = part.thought_signature
                        if part.function_call is not None:
                            call = part.function_call
                            if call.id:
                                extra["google_id"] = call.id
                            yield StreamDelta(
                                100 + calls,
                                ToolCallPart(
                                    call.name, json.dumps(call.args), f"call_{uuid.uuid4().hex[:24]}", **extra
                                ),
                            )
                            calls += 1
                        elif part.thought:
                            yield StreamDelta(1, ThinkingPart(part.text or "", signature=part.thought_signature))
                        elif part.text or extra:
                            yield StreamDelta(0, TextPart(part.text or "", **extra))
        except BROKEN as error:
            raise StreamBroken(repr(error)) from error
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _google_usage(usage: gemini.UsageMetadata) -> RequestUsage:
    """Gemini counts cached tokens inside the prompt, as Chat Completions does, and thinking apart from the
    candidates while billing it as output."""
    thoughts = usage.thoughts_token_count
    return RequestUsage(
        input_tokens=usage.prompt_token_count + usage.tool_use_prompt_token_count,
        output_tokens=usage.candidates_token_count + (thoughts or 0),
        cache_read_tokens=usage.cached_content_token_count,
        reasoning_tokens=thoughts,
    )


def model_for(name: str) -> OpenAIChatModel | AnthropicModel | GoogleModel:
    """The transport for a `provider/model` name."""
    match split_name(name)[0]:
        case "anthropic":
            return AnthropicModel(name)
        case "google":
            return GoogleModel(name)
        case _:
            return OpenAIChatModel(name)
