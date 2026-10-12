"""Aid's durable model conversation format, independent of the transport library."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence


class _Record(BaseModel):
    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class SystemPromptPart(_Record):
    content: str
    part_kind: Literal["system-prompt"] = "system-prompt"

    def __init__(self, content: str, **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class UserPromptPart(_Record):
    content: Any
    part_kind: Literal["user-prompt"] = "user-prompt"

    def __init__(self, content: Any, **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class TextPart(_Record):
    content: str
    part_kind: Literal["text"] = "text"

    def __init__(self, content: str, **kwargs: Any) -> None:
        super().__init__(content=content, **kwargs)


class ThinkingPart(_Record):
    content: str
    signature: str | None = None
    part_kind: Literal["thinking"] = "thinking"

    def __init__(self, content: str, signature: str | None = None, **kwargs: Any) -> None:
        super().__init__(content=content, signature=signature, **kwargs)


class ToolCallPart(_Record):
    tool_name: str
    args: str | dict[str, Any]
    tool_call_id: str
    part_kind: Literal["tool-call"] = "tool-call"

    def __init__(self, tool_name: str, args: str | dict[str, Any], tool_call_id: str, **kwargs: Any) -> None:
        super().__init__(tool_name=tool_name, args=args, tool_call_id=tool_call_id, **kwargs)

    def args_as_dict(self) -> dict[str, Any]:
        value = json.loads(self.args) if isinstance(self.args, str) else self.args
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be an object")
        return cast(dict[str, Any], value)

    def args_as_json_str(self) -> str:
        return self.args if isinstance(self.args, str) else json.dumps(self.args)


class ToolReturnPart(_Record):
    tool_name: str
    content: Any
    tool_call_id: str
    outcome: Literal["success", "failed", "denied", "interrupted"] = "success"
    part_kind: Literal["tool-return"] = "tool-return"

    def __init__(self, tool_name: str, content: Any, tool_call_id: str, **kwargs: Any) -> None:
        super().__init__(tool_name=tool_name, content=content, tool_call_id=tool_call_id, **kwargs)

    def model_response_str(self) -> str:
        return self.content if isinstance(self.content, str) else json.dumps(self.content, separators=(",", ":"))


class RetryPromptPart(_Record):
    content: Any
    tool_name: str | None = None
    tool_call_id: str = ""
    part_kind: Literal["retry-prompt"] = "retry-prompt"

    def __init__(self, content: Any, tool_name: str | None = None, tool_call_id: str = "", **kwargs: Any) -> None:
        super().__init__(content=content, tool_name=tool_name, tool_call_id=tool_call_id, **kwargs)

    def model_response(self) -> str:
        return str(self.content)


type Part = Annotated[
    SystemPromptPart | UserPromptPart | TextPart | ThinkingPart | ToolCallPart | ToolReturnPart | RetryPromptPart,
    Field(discriminator="part_kind"),
]


class ModelRequest(_Record):
    parts: list[Part]
    kind: Literal["request"] = "request"

    def __init__(self, parts: Sequence[Part], **kwargs: Any) -> None:
        super().__init__(parts=list(parts), **kwargs)


class ModelResponse(_Record):
    parts: list[Part]
    model_name: str | None = None
    kind: Literal["response"] = "response"

    def __init__(self, parts: Sequence[Part], model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(parts=list(parts), model_name=model_name, **kwargs)


type ModelMessage = Annotated[ModelRequest | ModelResponse, Field(discriminator="kind")]
ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])
