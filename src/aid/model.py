"""Model transports over the provider SDKs, and streamed response assembly. No agent execution lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import anthropic
import anyio
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
from pydantic import TypeAdapter

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
    from collections.abc import AsyncGenerator, Sequence

    from openai.types import CompletionUsage

    from aid.agent import Tool


@dataclass
class ModelInfo:
    tools: Sequence[Tool] = ()
    output_type: Any = str
    settings: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class RequestUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class StreamDelta:
    index: int = 0
    part: TextPart | ThinkingPart | ToolCallPart | None = None
    usage: RequestUsage | None = None
    model_name: str | None = None


class Model(Protocol):
    model_name: str

    def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]: ...
    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse: ...


def wire_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Stored parts as Chat Completions messages, keeping tool identities and reasoning text."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            content: Any = text or None
            thoughts = [p for p in message.parts if isinstance(p, ThinkingPart)]
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if not content and not calls:
                # Providers refuse an assistant message with neither (DeepSeek: 400). A failed request could
                # store one, and that stored history must still send.
                continue
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                item["tool_calls"] = [
                    {
                        "id": p.tool_call_id,
                        "type": "function",
                        "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
                    }
                    for p in calls
                ]
            if thoughts:
                item["reasoning_content"] = "".join(p.content for p in thoughts)
            result.append(item)
        else:
            for part in message.parts:
                match part:
                    case SystemPromptPart():
                        result.append({"role": "system", "content": part.content})
                    case UserPromptPart():
                        result.append({"role": "user", "content": part.content})
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


OPENAI_COMPATIBLE: Final[dict[str, Provider]] = {
    "openai": Provider(None, "OPENAI_API_KEY"),
    "deepseek": Provider("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": Provider("https://api.x.ai/v1", "XAI_API_KEY"),
    "together": Provider("https://api.together.xyz/v1", "TOGETHERAI_API_KEY"),
    "fireworks": Provider("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "ollama": Provider("http://localhost:11434/v1", "OLLAMA_API_KEY", local=True),
    "vllm": Provider("http://localhost:8000/v1", "VLLM_API_KEY", local=True),
    "lmstudio": Provider("http://localhost:1234/v1", "LM_STUDIO_API_KEY", local=True),
}
"""`provider/model` prefixes served through the OpenAI SDK. Another prefix works with an explicit `api_base`."""

CLIENT_SETTINGS: Final = frozenset({"api_key", "api_base", "base_url", "num_retries", "max_retries"})
"""`model_settings` keys that configure the SDK client; every other key goes into each request."""
DEFAULT_TIMEOUT: Final = 600


def split_name(name: str) -> tuple[str, str]:
    """`provider/model` → (provider, model); a bare name is OpenAI's."""
    provider, _, model = name.partition("/")
    return (provider, model) if model else ("openai", provider)


@dataclass(frozen=True)
class _ClientKey:
    base_url: str | None
    api_key: str
    max_retries: int


def _client_key(name: str, provider: Provider | None, settings: dict[str, Any]) -> _ClientKey:
    base_url = settings.get("api_base") or settings.get("base_url") or (provider and provider.base_url)
    if provider is None and not base_url:
        raise AidError(
            "unknown_provider",
            f"no provider {split_name(name)[0]!r}: use one of {', '.join(OPENAI_COMPATIBLE)}, anthropic, "
            "or set api_base in model_settings",
        )
    key_env = provider.key_env if provider else None
    api_key = settings.get("api_key") or (os.environ.get(key_env) if key_env else None)
    if not api_key:
        if provider is None or not provider.local:
            raise AidError("missing_credentials", f"{name}: set {key_env or 'api_key in model_settings'}")
        api_key = "local"
    retries = settings.get("max_retries", settings.get("num_retries", 0))
    return _ClientKey(base_url, api_key, retries)


def _request_settings(settings: dict[str, Any]) -> dict[str, Any]:
    kwargs = {key: value for key, value in settings.items() if key not in CLIENT_SETTINGS}
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return kwargs


class OpenAIChatModel:
    """Chat Completions through the OpenAI SDK, for OpenAI and every provider that serves the same API."""

    def __init__(self, name: str) -> None:
        self.model_name = name
        provider, self._model = split_name(name)
        self._provider = OPENAI_COMPATIBLE.get(provider)
        self._clients: dict[_ClientKey, openai.AsyncOpenAI] = {}

    def _client(self, settings: dict[str, Any]) -> openai.AsyncOpenAI:
        key = _client_key(self.model_name, self._provider, settings)
        if (client := self._clients.get(key)) is None:
            client = openai.AsyncOpenAI(
                api_key=key.api_key,
                base_url=key.base_url,
                max_retries=key.max_retries,
                http_client=openai.DefaultAsyncHttpx2Client(),
            )
            self._clients[key] = client
        return client

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.close()

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        client = self._client(info.settings)
        kwargs = _request_settings(info.settings)
        if info.tools:
            kwargs["tools"] = [tool.definition() for tool in info.tools]
        if info.output_type is not str:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": TypeAdapter(info.output_type).json_schema()},
            }
        if split_name(self.model_name)[0] != "gemini":
            kwargs.setdefault("stream_options", {"include_usage": True})
        stream = await client.chat.completions.create(
            model=self._model, messages=cast(Any, wire_messages(messages)), stream=True, **kwargs
        )
        ids: dict[int, str] = {}
        try:
            async for chunk in stream:
                if chunk.model:
                    yield StreamDelta(model_name=chunk.model)
                if chunk.usage is not None:
                    yield StreamDelta(usage=_openai_usage(chunk.usage))
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        yield StreamDelta(0, TextPart(delta.content))
                    # DeepSeek and others stream reasoning beside the content; the SDK keeps the extra field.
                    if thought := (delta.model_extra or {}).get("reasoning_content"):
                        yield StreamDelta(1, ThinkingPart(thought))
                    for call in delta.tool_calls or []:
                        if call.id:
                            ids[call.index] = call.id
                        function = call.function
                        yield StreamDelta(
                            100 + call.index,
                            ToolCallPart(
                                (function and function.name) or "",
                                (function and function.arguments) or "",
                                ids.get(call.index, ""),
                            ),
                        )
        finally:
            with anyio.CancelScope(shield=True):
                await stream.close()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        return await collect(self, messages, info)


def _openai_usage(usage: CompletionUsage) -> RequestUsage:
    """`input_tokens` includes cached tokens here, as Chat Completions counts them. DeepSeek reports its cache
    hits beside the standard details."""
    prompt = usage.prompt_tokens_details
    cached = (prompt and prompt.cached_tokens) or (usage.model_extra or {}).get("prompt_cache_hit_tokens") or 0
    completion = usage.completion_tokens_details
    return RequestUsage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        cache_read_tokens=cached,
        reasoning_tokens=(completion and completion.reasoning_tokens) or 0,
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


def anthropic_messages(messages: Sequence[ModelMessage]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stored parts as the Messages API's (system, messages). Leading system prompts become `system`; a later one
    stays in place as user text. Consecutive same-role parts share one message, so the tool results of one
    response arrive together. Thinking replays only signed or redacted, as Anthropic produced it."""
    system: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []

    def add(role: str, block: dict[str, Any]) -> None:
        if result and result[-1]["role"] == role:
            result[-1]["content"].append(block)
        else:
            result.append({"role": role, "content": [block]})

    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                match part:
                    case ThinkingPart() if part.model_extra and "redacted_data" in part.model_extra:
                        add("assistant", {"type": "redacted_thinking", "data": part.model_extra["redacted_data"]})
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
                            {"type": "tool_use", "id": part.tool_call_id, "name": part.tool_name, "input": arguments},
                        )
                    case _:
                        pass
            continue
        for part in message.parts:
            match part:
                case SystemPromptPart() if not result:
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
    return system, result


def anthropic_tool(definition: dict[str, Any]) -> dict[str, Any]:
    function = definition["function"]
    return {"name": function["name"], "description": function["description"], "input_schema": function["parameters"]}


class AnthropicModel:
    """The Messages API through the Anthropic SDK: signed thinking replayed, beta headers and caching passed
    through. `cache_control` defaults to automatic caching of the longest prefix."""

    def __init__(self, name: str) -> None:
        self.model_name = name
        self._model = split_name(name)[1]
        self._clients: dict[_ClientKey, anthropic.AsyncAnthropic] = {}

    def _client(self, settings: dict[str, Any]) -> anthropic.AsyncAnthropic:
        api_key = settings.get("api_key") or os.environ.get(ANTHROPIC_KEY_ENV)
        if not api_key:
            raise AidError("missing_credentials", f"{self.model_name}: set {ANTHROPIC_KEY_ENV}")
        retries = settings.get("max_retries", settings.get("num_retries", 0))
        key = _ClientKey(settings.get("api_base") or settings.get("base_url"), api_key, retries)
        if (client := self._clients.get(key)) is None:
            client = anthropic.AsyncAnthropic(api_key=key.api_key, base_url=key.base_url, max_retries=key.max_retries)
            self._clients[key] = client
        return client

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.close()

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        client = self._client(info.settings)
        kwargs = _request_settings(info.settings)
        kwargs.setdefault(
            "max_tokens", ANTHROPIC_LEGACY_MAX_TOKENS if self._model.startswith("claude-3") else ANTHROPIC_MAX_TOKENS
        )
        kwargs.setdefault("cache_control", {"type": "ephemeral"})
        if info.tools:
            kwargs["tools"] = [anthropic_tool(tool.definition()) for tool in info.tools]
        if info.output_type is not str:
            schema = TypeAdapter(info.output_type).json_schema()
            kwargs["output_config"] = {
                **kwargs.get("output_config", {}),
                "format": {"type": "json_schema", "schema": schema},
            }
        system, wire = anthropic_messages(messages)
        if system:
            kwargs["system"] = system
        stream = await client.messages.create(model=self._model, messages=cast(Any, wire), stream=True, **kwargs)
        arguments: dict[int, bool] = {}
        try:
            async for event in stream:
                match event:
                    case RawMessageStartEvent():
                        yield StreamDelta(model_name=event.message.model)
                        yield StreamDelta(usage=_anthropic_usage(event.message.usage))
                    case RawMessageDeltaEvent():
                        yield StreamDelta(usage=_anthropic_usage(event.usage))
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


def _anthropic_usage(usage: Any) -> RequestUsage:
    """Anthropic's `input_tokens` excludes cache reads and writes; aid keeps that split."""
    return RequestUsage(
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=getattr(usage, "output_tokens", None) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def model_for(name: str) -> OpenAIChatModel | AnthropicModel:
    """The transport for a `provider/model` name."""
    return AnthropicModel(name) if split_name(name)[0] == "anthropic" else OpenAIChatModel(name)
