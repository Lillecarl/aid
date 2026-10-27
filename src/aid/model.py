"""Model transports over the provider SDKs, and streamed response assembly. No agent execution lives here."""

from __future__ import annotations

import base64
import collections.abc
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, cast, runtime_checkable

import anthropic
import anyio
import httpx
import openai
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from anthropic.types.message_create_params import MessageCreateParamsStreaming
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from aid.messages import (
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
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from anthropic.types import ContentBlockParam, MessageDeltaUsage, MessageParam, TextBlockParam, ToolParam
    from anthropic.types import Usage as AnthropicUsage
    from google import genai
    from google.genai.types import (
        ContentDict,
        FunctionCallDict,
        FunctionDeclarationDict,
        FunctionResponseDict,
        GenerateContentConfigDict,
        GenerateContentResponseUsageMetadata,
        PartDict,
    )
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema
    from pydantic.json_schema import JsonSchemaValue

    from aid.agent import Tool


@dataclass
class ModelInfo:
    tools: Sequence[Tool] = ()
    output_type: Any = str
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


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


class Model(Protocol):
    model_name: str

    def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]: ...
    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse: ...


class Transport(Model, Protocol):
    """A model aid built from a name: it owns its clients, and whoever resolved it closes them."""

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


class ReasoningAssistantParam(ChatCompletionAssistantMessageParam, total=False):
    """An assistant message with its reasoning beside the content, as DeepSeek and others take it back."""

    reasoning_content: str


def wire_messages(messages: Sequence[ModelMessage], *, reasoning: bool = True) -> list[ChatCompletionMessageParam]:
    """Stored parts as Chat Completions messages, keeping tool identities and, with `reasoning`, reasoning text."""
    result: list[ChatCompletionMessageParam] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            thoughts = [p for p in message.parts if isinstance(p, ThinkingPart)]
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if not text and not calls:
                # Providers refuse an assistant message with neither (DeepSeek: 400). A failed request could
                # store one, and that stored history must still send.
                continue
            item: ReasoningAssistantParam = {"role": "assistant", "content": text or None}
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
            for part in message.parts:
                match part:
                    case SystemPromptPart():
                        result.append({"role": "system", "content": part.content})
                    case UserPromptPart():
                        result.append({"role": "user", "content": str(part.content)})
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
    """None: the SDK's own default (api.openai.com)."""
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
"""`provider/model` prefixes served through the OpenAI SDK, for servers models.dev names no URL for and local ones.
Others resolve through `aid.providers`."""


class ClientSettings(BaseModel):
    """The `model_settings` keys that configure the SDK client; every other key goes into each request."""

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
class _ClientKey:
    base_url: str | None
    api_key: str
    max_retries: int


def client_settings(name: str, settings: Mapping[str, JsonValue]) -> ClientSettings:
    try:
        return ClientSettings.model_validate(settings)
    except ValidationError as e:
        raise AidError("bad_model_settings", f"{name}: {e}") from e


def _client_key(
    name: str, settings: Mapping[str, JsonValue], key_env: str | None, base_url: str | None, *, local: bool = False
) -> _ClientKey:
    """The client's settings over the provider's: `key_env` and `base_url` apply where the settings name none."""
    client = client_settings(name, settings)
    api_key = client.api_key or (os.environ.get(key_env) if key_env else None)
    if not api_key:
        if not local:
            raise AidError("missing_credentials", f"{name}: set {key_env or 'api_key in model_settings'}")
        api_key = "local"
    return _ClientKey(client.base_url or base_url, api_key, client.max_retries)


def _request_settings(settings: Mapping[str, JsonValue], provider: Provider | None = None) -> dict[str, JsonValue]:
    kwargs: dict[str, JsonValue] = {**(provider.defaults if provider else {})}
    kwargs.update((key, value) for key, value in settings.items() if key not in CLIENT_SETTINGS)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return kwargs


class OpenAIChatModel:
    """Chat Completions through the OpenAI SDK, for OpenAI and every provider that serves the same API. A provider
    with quirks subclasses it and overrides `request_settings`, `reasoning` or `usage`."""

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider` None: the one `OPENAI_COMPATIBLE` names by the prefix, if any."""
        self.model_name = name
        prefix, self._model = split_name(name)
        self._provider = provider or OPENAI_COMPATIBLE.get(prefix)
        self._clients: dict[_ClientKey, openai.AsyncOpenAI] = {}

    def _client(self, settings: Mapping[str, JsonValue]) -> openai.AsyncOpenAI:
        provider = self._provider
        if provider is None and client_settings(self.model_name, settings).base_url is None:
            raise AidError(
                "unknown_provider",
                f"no provider {split_name(self.model_name)[0]!r}: use one of {', '.join(OPENAI_COMPATIBLE)}, "
                "anthropic, or set api_base in model_settings",
            )
        key = _client_key(
            self.model_name,
            settings,
            provider.key_env if provider else None,
            provider.base_url if provider else None,
            local=provider is not None and provider.local,
        )
        if (client := self._clients.get(key)) is None:
            client = openai.AsyncOpenAI(
                api_key=key.api_key,
                base_url=key.base_url,
                max_retries=key.max_retries,
                default_headers=dict(self._provider.headers) if self._provider else None,
                http_client=openai.DefaultAsyncHttpx2Client(),
            )
            self._clients[key] = client
        return client

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return self._client(settings).base_url.host

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.close()

    def request_settings(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Each request's keyword arguments from the session's settings, under the provider's defaults."""
        kwargs = _request_settings(settings, self._provider)
        if self._provider is None or self._provider.stream_usage:
            kwargs.setdefault("stream_options", {"include_usage": True})
        return kwargs

    def reasoning(self, delta: ChoiceDelta) -> str | None:
        """A streamed delta's reasoning text, beside the content: DeepSeek names it `reasoning_content`, OpenRouter
        and vLLM `reasoning` (measured 2026-10-27, vLLM serving glm-5.3-flash). The SDK keeps the extra field."""
        extra = delta.model_extra or {}
        for key in ("reasoning_content", "reasoning"):
            if isinstance(thought := extra.get(key), str) and thought:
                return thought
        return None

    def usage(self, usage: CompletionUsage) -> RequestUsage:
        return _openai_usage(usage)

    replays_reasoning: ClassVar[bool] = True
    """Reasoning text goes back beside the content (`reasoning_content`); False for a provider that refuses it."""

    def wire(self, messages: Sequence[ModelMessage]) -> list[ChatCompletionMessageParam]:
        return wire_messages(messages, reasoning=self.replays_reasoning)

    def response_format(self, schema: JsonSchemaValue) -> ResponseFormatJSONSchema | ResponseFormatJSONObject:
        """How a request asks for structured output: the schema itself, where the provider takes one."""
        return {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}

    def schema_note(self, schema: JsonSchemaValue) -> str | None:
        """A system note stating the output schema, for a provider whose JSON mode takes no schema."""
        return None

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        client = self._client(info.settings)
        wired = self.wire(messages)
        # The settings are the session's JSON, under the SDK's keyword names; the SDK and the server check them.
        request = cast(
            CompletionCreateParamsStreaming,
            {**self.request_settings(info.settings), "model": self._model, "messages": wired, "stream": True},
        )
        if info.tools:
            request["tools"] = [tool.definition() for tool in info.tools]
        if info.output_type is not str:
            schema = TypeAdapter(info.output_type).json_schema()
            request["response_format"] = self.response_format(schema)
            if (note := self.schema_note(schema)) is not None:
                # After the leading instructions, so every request keeps the same cached prefix.
                lead = next((i for i, w in enumerate(wired) if w.get("role") != "system"), len(wired))
                wired.insert(lead, {"role": "system", "content": note})
        stream = await client.chat.completions.create(**request)
        ids: dict[int, str] = {}
        try:
            async for chunk in stream:
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
        finally:
            with anyio.CancelScope(shield=True):
                await stream.close()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _openai_usage(usage: CompletionUsage) -> RequestUsage:
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


def anthropic_messages(messages: Sequence[ModelMessage]) -> tuple[list[TextBlockParam], list[MessageParam]]:
    """Stored parts as the Messages API's (system, messages). Leading system prompts become `system`; a later one
    stays in place as user text. Consecutive same-role parts share one message, so the tool results of one
    response arrive together. Thinking replays only signed or redacted, as Anthropic produced it."""
    system: list[TextBlockParam] = []
    turns: list[tuple[Literal["user", "assistant"], list[ContentBlockParam]]] = []

    def add(role: Literal["user", "assistant"], block: ContentBlockParam) -> None:
        if turns and turns[-1][0] == role:
            turns[-1][1].append(block)
        else:
            turns.append((role, [block]))

    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ThinkingPart() if part.model_extra and "redacted_data" in part.model_extra:
                        add("assistant", {"type": "redacted_thinking", "data": str(part.model_extra["redacted_data"])})
                    case ThinkingPart() if part.signature:
                        add("assistant", {"type": "thinking", "thinking": part.content, "signature": part.signature})
                    case TextPart() if part.content:
                        add("assistant", {"type": "text", "text": part.content})
                    case ToolCallPart():
                        try:
                            arguments = part.args_as_dict()
                        except ValueError:
                            arguments = {}
                        add(
                            "assistant",
                            {
                                "type": "tool_use",
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "input": {**arguments},
                            },
                        )
                    case _:
                        pass
            continue
        for part in message.parts:
            match part:
                case SystemPromptPart() if not turns:
                    system.append({"type": "text", "text": part.content})
                case SystemPromptPart():
                    add("user", {"type": "text", "text": part.content})
                case UserPromptPart():
                    add("user", {"type": "text", "text": str(part.content)})
                case ToolReturnPart():
                    add(
                        "user",
                        {"type": "tool_result", "tool_use_id": part.tool_call_id, "content": part.model_response_str()},
                    )
                case RetryPromptPart() if part.tool_name:
                    add(
                        "user",
                        {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": part.model_response(),
                            "is_error": True,
                        },
                    )
                case RetryPromptPart():
                    add("user", {"type": "text", "text": part.model_response()})
                case _:
                    raise ValueError("response part in a model request")
    return system, [{"role": role, "content": blocks} for role, blocks in turns]


def anthropic_tool(definition: ChatCompletionFunctionToolParam) -> ToolParam:
    function = definition["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {}),
    }


class AnthropicModel:
    """The Messages API through the Anthropic SDK: signed thinking replayed, beta headers and caching passed
    through. `cache_control` defaults to automatic caching of the longest prefix."""

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider`: another server of the Messages API, its base URL and key variable; None is Anthropic's."""
        self.model_name = name
        self._model = split_name(name)[1]
        self._provider = provider
        self._clients: dict[_ClientKey, anthropic.AsyncAnthropic] = {}

    def _client(self, settings: Mapping[str, JsonValue]) -> anthropic.AsyncAnthropic:
        provider = self._provider
        key_env = provider.key_env if provider else ANTHROPIC_KEY_ENV
        key = _client_key(self.model_name, settings, key_env, provider.base_url if provider else None)
        if (client := self._clients.get(key)) is None:
            client = anthropic.AsyncAnthropic(
                api_key=key.api_key,
                base_url=key.base_url,
                max_retries=key.max_retries,
                default_headers=dict(self._provider.headers) if self._provider else None,
            )
            self._clients[key] = client
        return client

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        return self._client(settings).base_url.host

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.close()

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        client = self._client(info.settings)
        system, wire = anthropic_messages(messages)
        # The settings are the session's JSON, under the SDK's keyword names; the SDK and the server check them.
        request = cast(
            MessageCreateParamsStreaming,
            {**_request_settings(info.settings), "model": self._model, "messages": wire, "stream": True},
        )
        request.setdefault(
            "max_tokens", ANTHROPIC_LEGACY_MAX_TOKENS if self._model.startswith("claude-3") else ANTHROPIC_MAX_TOKENS
        )
        request.setdefault("cache_control", {"type": "ephemeral"})
        if info.tools:
            request["tools"] = [anthropic_tool(tool.definition()) for tool in info.tools]
        if info.output_type is not str:
            schema = TypeAdapter(info.output_type).json_schema()
            request["output_config"] = {
                **request.get("output_config", {}),
                "format": {"type": "json_schema", "schema": schema},
            }
        if system:
            request["system"] = system
        stream = await client.messages.create(**request)
        arguments: dict[int, bool] = {}
        try:
            async for event in stream:
                match event:
                    case RawMessageStartEvent():
                        yield StreamDelta(model_name=event.message.model)
                        yield StreamDelta(usage=_anthropic_usage(event.message.usage))
                    case RawMessageDeltaEvent():
                        yield StreamDelta(usage=_anthropic_usage(event.usage))
                        if event.delta.stop_reason == "max_tokens":
                            yield StreamDelta(truncated=True)
                    case RawContentBlockStartEvent():
                        index, block = event.index, event.content_block
                        match block:
                            case TextBlock():
                                yield StreamDelta(index, TextPart(block.text))
                            case ThinkingBlock():
                                yield StreamDelta(index, ThinkingPart(block.thinking, signature=block.signature))
                            case RedactedThinkingBlock():
                                yield StreamDelta(index, ThinkingPart("", redacted_data=block.data))
                            case ToolUseBlock():
                                arguments[index] = False
                                yield StreamDelta(index, ToolCallPart(block.name, "", block.id))
                            case _:
                                pass
                    case RawContentBlockDeltaEvent():
                        index, delta = event.index, event.delta
                        match delta:
                            case TextDelta():
                                yield StreamDelta(index, TextPart(delta.text))
                            case ThinkingDelta():
                                yield StreamDelta(index, ThinkingPart(delta.thinking))
                            case SignatureDelta():
                                yield StreamDelta(index, ThinkingPart("", signature=delta.signature))
                            case InputJSONDelta() if delta.partial_json:
                                arguments[index] = True
                                yield StreamDelta(index, ToolCallPart("", delta.partial_json, ""))
                            case _:
                                pass
                    case RawContentBlockStopEvent() if arguments.get(event.index) is False:
                        # A tool without arguments streams no JSON at all.
                        yield StreamDelta(event.index, ToolCallPart("", "{}", ""))
                    case _:
                        pass
        finally:
            with anyio.CancelScope(shield=True):
                await stream.close()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _anthropic_usage(usage: AnthropicUsage | MessageDeltaUsage) -> RequestUsage:
    """Anthropic's `input_tokens` excludes cache reads and writes; aid keeps that split."""
    return RequestUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )


GOOGLE_KEY_ENV: Final = "GEMINI_API_KEY"


def _signature(value: bytes | None) -> dict[str, str]:
    """A thought signature as a part's extra field: base64, so history stays JSON."""
    return {"thought_signature": base64.b64encode(value).decode()} if value else {}


def _signed(block: PartDict, part: TextPart | ToolCallPart) -> PartDict:
    if signature := (part.model_extra or {}).get("thought_signature"):
        block["thought_signature"] = base64.b64decode(signature)
    return block


def google_contents(messages: Sequence[ModelMessage]) -> tuple[list[PartDict], list[ContentDict]]:
    """Stored parts as the Gemini API's (system instruction parts, contents), merged by role as
    `anthropic_messages` does. A thought signature goes back on the part that carried it: Gemini 3 refuses a
    function call without its own. Thinking replays only signed."""
    system: list[PartDict] = []
    turns: list[tuple[str, list[PartDict]]] = []
    ids: dict[str, str | None] = {}

    def add(role: str, block: PartDict) -> None:
        if turns and turns[-1][0] == role:
            turns[-1][1].append(block)
        else:
            turns.append((role, [block]))

    def answer(part: ToolReturnPart | RetryPromptPart, response: dict[str, str]) -> PartDict:
        block: FunctionResponseDict = {"name": part.tool_name, "response": response}
        if google_id := ids.get(part.tool_call_id):
            block["id"] = google_id
        return {"function_response": block}

    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ThinkingPart() if part.signature:
                        add(
                            "model",
                            {
                                "text": part.content,
                                "thought": True,
                                "thought_signature": base64.b64decode(part.signature),
                            },
                        )
                    case TextPart() if part.content or part.model_extra:
                        add("model", _signed({"text": part.content}, part))
                    case ToolCallPart():
                        try:
                            arguments = part.args_as_dict()
                        except ValueError:
                            arguments = {}
                        call: FunctionCallDict = {"name": part.tool_name, "args": arguments}
                        if google_id := (part.model_extra or {}).get("google_id"):
                            call["id"] = google_id
                        ids[part.tool_call_id] = google_id
                        add("model", _signed({"function_call": call}, part))
                    case _:
                        pass
            continue
        for part in message.parts:
            match part:
                case SystemPromptPart() if not turns:
                    system.append({"text": part.content})
                case SystemPromptPart():
                    add("user", {"text": part.content})
                case UserPromptPart():
                    add("user", {"text": str(part.content)})
                case ToolReturnPart():
                    add("user", answer(part, {"output": part.model_response_str()}))
                case RetryPromptPart() if part.tool_name:
                    add("user", answer(part, {"error": part.model_response()}))
                case RetryPromptPart():
                    add("user", {"text": part.model_response()})
                case _:
                    raise ValueError("response part in a model request")
    return system, [{"role": role, "parts": parts} for role, parts in turns]


def google_tool(tool: Tool) -> FunctionDeclarationDict:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters_json_schema": tool.arguments.model_json_schema(),
    }


class GoogleModel:
    """The Gemini API (`generateContent`) through Google's SDK, imported on first use: it adds half a second to a
    worker's start (measured 2026-10-24). Thought signatures replay on their parts; Gemini assigns call ids only
    sometimes, so aid makes its own and keeps Gemini's beside them."""

    def __init__(self, name: str, provider: Provider | None = None) -> None:
        """`provider`: another server of the Gemini API, its base URL and key variable; None is Google AI Studio."""
        self.model_name = name
        self._model = split_name(name)[1]
        self._provider = provider
        self._clients: dict[_ClientKey, tuple[genai.Client, httpx.AsyncClient]] = {}

    def _client(self, settings: Mapping[str, JsonValue]) -> genai.Client:
        provider = self._provider
        key_env = provider.key_env if provider else GOOGLE_KEY_ENV
        key = _client_key(self.model_name, settings, key_env, provider.base_url if provider else None)
        if (pair := self._clients.get(key)) is None:
            from google import genai  # noqa: PLC0415 -- half a second of every worker's start otherwise
            from google.genai.types import HttpOptions, HttpRetryOptions  # noqa: PLC0415 -- the same

            # Ours, not the SDK's: it would pick aiohttp whenever that is installed.
            http = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            options = HttpOptions(
                httpx_async_client=http,
                base_url=key.base_url,
                headers=dict(self._provider.headers) if self._provider and self._provider.headers else None,
                retry_options=HttpRetryOptions(attempts=key.max_retries + 1) if key.max_retries else None,
            )
            pair = (genai.Client(vertexai=False, api_key=key.api_key, http_options=options), http)
            self._clients[key] = pair
        return pair[0]

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        # google-genai keeps the URL it sends to only here: aid's own setting is None for its default.
        base_url = self._client(settings)._api_client._http_options.base_url  # pyright: ignore[reportPrivateUsage] -- no public accessor
        return httpx.URL(base_url or "").host

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client, http in clients:
            await client.aio.aclose()
            client.close()
            await http.aclose()

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        client = self._client(info.settings)
        from google.genai.types import GenerateContentConfig  # noqa: PLC0415 -- loaded with the client above

        settings = _request_settings(info.settings, self._provider)
        # aid runs the tool loop; the SDK's own would also log a warning on each worker's first request.
        ours: GenerateContentConfigDict = {
            "http_options": {"timeout": int(_TIMEOUT.validate_python(settings.pop("timeout")) * 1000)},
            "automatic_function_calling": {"disable": True},
        }
        if info.tools:
            ours["tools"] = [{"function_declarations": [google_tool(tool) for tool in info.tools]}]
        if info.output_type is not str:
            ours["response_mime_type"] = "application/json"
            ours["response_json_schema"] = TypeAdapter(info.output_type).json_schema()
        system, contents = google_contents(messages)
        if system:
            ours["system_instruction"] = {"parts": system}
        # The session's settings are Google's field names; the SDK's model checks them.
        config = GenerateContentConfig.model_validate({**settings, **ours})
        stream = await client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType] -- PIL, absent, in its contents union
            model=self._model, contents=contents, config=config
        )
        calls = 0
        try:
            async for chunk in stream:
                if chunk.model_version:
                    yield StreamDelta(model_name=chunk.model_version)
                if chunk.usage_metadata is not None:
                    yield StreamDelta(usage=_google_usage(chunk.usage_metadata))
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason == "MAX_TOKENS":
                        yield StreamDelta(truncated=True)
                    for part in (candidate.content.parts if candidate.content else None) or []:
                        extra = _signature(part.thought_signature)
                        if part.function_call is not None:
                            call = part.function_call
                            if call.id:
                                extra["google_id"] = call.id
                            yield StreamDelta(
                                100 + calls,
                                ToolCallPart(
                                    call.name or "",
                                    json.dumps(call.args or {}),
                                    f"call_{uuid.uuid4().hex[:24]}",
                                    **extra,
                                ),
                            )
                            calls += 1
                        elif part.thought:
                            yield StreamDelta(
                                1, ThinkingPart(part.text or "", signature=extra.get("thought_signature"))
                            )
                        elif part.text or extra:
                            yield StreamDelta(0, TextPart(part.text or "", **extra))
        finally:
            # Typed as an iterator; the SDK returns an async generator, which holds the response open.
            if isinstance(stream, collections.abc.AsyncGenerator):
                with anyio.CancelScope(shield=True):
                    await stream.aclose()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _google_usage(usage: GenerateContentResponseUsageMetadata) -> RequestUsage:
    """Gemini counts cached tokens inside the prompt, as Chat Completions does, and thinking apart from the
    candidates while billing it as output."""
    thoughts = usage.thoughts_token_count
    return RequestUsage(
        input_tokens=(usage.prompt_token_count or 0) + (usage.tool_use_prompt_token_count or 0),
        output_tokens=(usage.candidates_token_count or 0) + (thoughts or 0),
        cache_read_tokens=usage.cached_content_token_count or 0,
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
