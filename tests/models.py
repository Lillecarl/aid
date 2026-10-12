"""Deterministic model transports for testing aid's execution loop without provider calls."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from aid.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import ModelInfo, RequestUsage, StreamDelta, accumulate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

AgentInfo = ModelInfo


@dataclass
class DeltaToolCall:
    name: str | None = None
    json_args: str | None = None
    tool_call_id: str | None = None


@dataclass
class DeltaThinkingPart:
    content: str = ""
    signature: str | None = None


type DeltaToolCalls = dict[int, DeltaToolCall]
type DeltaThinkingCalls = dict[int, DeltaThinkingPart]


class FunctionModel:
    model_name = "function:test"

    def __init__(
        self, function: Callable[..., Any] | None = None, *, stream_function: Callable[..., Any] | None = None
    ) -> None:
        self.function = function
        self.stream_function = stream_function

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        if self.function is not None:
            result = self.function(messages, info)
            if inspect.isawaitable(result):
                result = await result
            return result
        response = ModelResponse(parts=[], model_name=self.model_name)
        indexes: dict[int, int] = {}
        async for delta in self.stream(messages, info):
            accumulate(response, indexes, delta)
        return response

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        yield StreamDelta(usage=RequestUsage(input_tokens=1))
        if self.stream_function is None:
            response = await self.request(messages, info)
            for index, part in enumerate(response.parts):
                if isinstance(part, TextPart | ThinkingPart | ToolCallPart):
                    yield StreamDelta(index, part)
        else:
            async for value in self.stream_function(messages, info):
                if isinstance(value, str):
                    yield StreamDelta(0, TextPart(value))
                else:
                    for index, part in value.items():
                        if isinstance(part, DeltaToolCall):
                            yield StreamDelta(
                                100 + index,
                                ToolCallPart(part.name or "", part.json_args or "", part.tool_call_id or ""),
                            )
                        elif isinstance(part, DeltaThinkingPart):
                            yield StreamDelta(10 + index, ThinkingPart(part.content, signature=part.signature))
        yield StreamDelta(usage=RequestUsage(output_tokens=1))


def example(schema: dict[str, Any]) -> Any:
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return example(next((s for s in schema["anyOf"] if s.get("type") != "null"), schema["anyOf"][0]))
    match schema.get("type"):
        case "object":
            return {
                name: example(value)
                for name, value in schema.get("properties", {}).items()
                if name in schema.get("required", [])
            }
        case "array":
            return []
        case "integer" | "number":
            return 0
        case "boolean":
            return False
        case _:
            return "a"


class TestModel:
    __test__ = False
    model_name = "test"

    def __init__(
        self,
        *,
        call_tools: list[str] | None = None,
        custom_output_text: str | None = None,
        custom_output_args: Any = None,
    ) -> None:
        self.call_tools = call_tools
        self.custom_output_text = custom_output_text
        self.custom_output_args = custom_output_args

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        start = max(
            (
                i
                for i, m in enumerate(messages)
                if any(
                    isinstance(p, UserPromptPart) and not str(p.content).startswith("[Session metadata")
                    for p in m.parts
                )
            ),
            default=0,
        )
        results = {
            p.tool_name: p.content
            for m in messages[start:]
            for p in m.parts
            if isinstance(p, ToolReturnPart | RetryPromptPart)
        }
        tools = [t for t in info.tools if self.call_tools is None or t.name in self.call_tools]
        calls = [
            ToolCallPart(t.name or "", example(t.arguments.model_json_schema()), f"test-{i}")
            for i, t in enumerate(tools)
            if t.name not in results
        ]
        if calls:
            return ModelResponse(parts=calls)
        if self.custom_output_text is not None:
            answer = self.custom_output_text
        elif info.output_type is not str:
            answer = json.dumps(
                self.custom_output_args
                if self.custom_output_args is not None
                else example(TypeAdapter(info.output_type).json_schema())
            )
        else:
            answer = json.dumps(results)
        return ModelResponse(parts=[TextPart(answer)])

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        response = await self.request(messages, info)
        yield StreamDelta(usage=RequestUsage(input_tokens=1, output_tokens=1))
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart | ThinkingPart | ToolCallPart):
                yield StreamDelta(index, part)
