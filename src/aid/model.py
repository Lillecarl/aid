"""LiteLM transport and streamed response assembly. No agent execution lives here."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio
import litelm
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

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


class Model(Protocol):
    model_name: str

    def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]: ...
    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse: ...


def wire_messages(messages: Sequence[ModelMessage], *, anthropic: bool = False) -> list[dict[str, Any]]:
    """Translate stored parts without losing tool identities or signed Anthropic thinking blocks."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            content: Any = text or None
            thoughts = [p for p in message.parts if isinstance(p, ThinkingPart)]
            if anthropic and thoughts:
                content = []
                for thought in thoughts:
                    if thought.model_extra and "redacted_data" in thought.model_extra:
                        content.append({"type": "redacted_thinking", "data": thought.model_extra["redacted_data"]})
                    elif thought.signature:
                        content.append(
                            {"type": "thinking", "thinking": thought.content, "signature": thought.signature}
                        )
                if text:
                    content.append({"type": "text", "text": text})
            item: dict[str, Any] = {"role": "assistant", "content": content}
            calls = [p for p in message.parts if isinstance(p, ToolCallPart)]
            if calls:
                item["tool_calls"] = [
                    {
                        "id": p.tool_call_id,
                        "type": "function",
                        "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
                    }
                    for p in calls
                ]
            if thoughts and not anthropic:
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


class LiteLMModel:
    def __init__(self, model: str) -> None:
        self.model_name = model

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        kwargs = dict(info.settings)
        kwargs.setdefault("num_retries", 0)
        kwargs.setdefault("timeout", 600)
        if info.tools:
            kwargs["tools"] = [tool.definition() for tool in info.tools]
        if info.output_type is not str:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": TypeAdapter(info.output_type).json_schema()},
            }
        if not self.model_name.startswith(("anthropic/", "gemini/")):
            kwargs.setdefault("stream_options", {"include_usage": True})
        completion = cast(Callable[..., Awaitable[Any]], litelm.acompletion)  # pyright: ignore[reportUnknownMemberType] -- LiteLM's transport interface has no annotations.
        stream: Any = await completion(
            self.model_name,
            messages=wire_messages(messages, anthropic=self.model_name.startswith("anthropic/")),
            stream=True,
            **kwargs,
        )
        ids: dict[int, str] = {}
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_details: Any = getattr(usage, "prompt_tokens_details", None) or {}
                    completion_details: Any = getattr(usage, "completion_tokens_details", None) or {}
                    if not isinstance(prompt_details, dict):
                        prompt_details = prompt_details.model_dump()
                    if not isinstance(completion_details, dict):
                        completion_details = completion_details.model_dump()
                    prompt_details = cast(dict[str, Any], prompt_details)
                    completion_details = cast(dict[str, Any], completion_details)
                    yield StreamDelta(
                        usage=RequestUsage(
                            input_tokens=usage.prompt_tokens or 0,
                            output_tokens=usage.completion_tokens or 0,
                            cache_read_tokens=prompt_details.get("cached_tokens", 0) or 0,
                            cache_write_tokens=prompt_details.get("cache_creation_tokens", 0) or 0,
                            reasoning_tokens=completion_details.get("reasoning_tokens", 0) or 0,
                        )
                    )
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        yield StreamDelta(0, TextPart(delta.content))
                    thinking = getattr(delta, "thinking_blocks", None)
                    if thinking:
                        for block in thinking:
                            part = ThinkingPart(block.get("thinking", ""), signature=block.get("signature"))
                            if block.get("type") == "redacted_thinking":
                                part = ThinkingPart("", redacted_data=block["data"])
                            yield StreamDelta(10 + block.get("index", 0), part)
                    elif thought := getattr(delta, "reasoning_content", None):
                        yield StreamDelta(1, ThinkingPart(thought))
                    calls = cast(list[Any], delta.tool_calls or [])
                    for call in calls:
                        if call.id:
                            ids[call.index] = call.id
                        yield StreamDelta(
                            100 + call.index,
                            ToolCallPart(
                                call.function.name or "", call.function.arguments or "", ids.get(call.index, "")
                            ),
                        )
        finally:
            with anyio.CancelScope(shield=True):
                await stream.aclose()

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        response = ModelResponse(parts=[], model_name=self.model_name)
        indexes: dict[int, int] = {}
        async for delta in self.stream(messages, info):
            accumulate(response, indexes, delta)
        return response
