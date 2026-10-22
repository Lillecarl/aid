"""Agent configuration and validated Python tools. The backend owns execution and durable state."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_type_hints

import anyio.to_thread
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aid.model import Model


class ModelRetry(Exception):
    """A tool refusal or correctable input error sent back to the model."""


@dataclass(frozen=True)
class RunContext:
    tool_call_id: str


@dataclass
class Tool:
    fn: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    takes_ctx: bool | None = None
    kind: str | None = None
    """ACP's tool kind (`read`, `edit`, `delete`, `move`, `search`, `execute`, …): the web UI labels the call by it,
    and session summaries count a call of a write kind as writing its `path`."""
    errors_to_model: bool = False
    """Any exception goes back to the model as a retry, as an MCP server answers a failing tool. Otherwise only
    `ModelRetry` does, and other exceptions end the turn."""
    arguments: type[BaseModel] = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.name or self.fn.__name__
        self.description = self.description or inspect.getdoc(self.fn) or ""
        hints = get_type_hints(self.fn, include_extras=True)
        parameters = list(inspect.signature(self.fn).parameters.values())
        if self.takes_ctx is None:
            self.takes_ctx = bool(parameters and hints.get(parameters[0].name) is RunContext)
        fields: dict[str, Any] = {}
        for parameter in parameters[1:] if self.takes_ctx else parameters:
            if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                raise TypeError(f"tool {self.name}: unsupported parameter {parameter.name}")
            default = ... if parameter.default is parameter.empty else parameter.default
            fields[parameter.name] = (hints.get(parameter.name, Any), default)
        self.arguments = create_model(f"{self.name}Arguments", __config__=ConfigDict(extra="forbid"), **fields)

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }

    async def call(self, args: dict[str, Any], call_id: str) -> Any:
        try:
            validated = self.arguments.model_validate(args)
        except ValidationError as error:
            raise ModelRetry(str(error)) from error
        kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
        positional = (RunContext(call_id),) if self.takes_ctx else ()
        try:
            if inspect.iscoroutinefunction(self.fn):
                return await self.fn(*positional, **kwargs)
            return await anyio.to_thread.run_sync(lambda: self.fn(*positional, **kwargs))
        except ModelRetry:
            raise
        except Exception as error:
            if not self.errors_to_model:
                raise
            raise ModelRetry(f"{type(error).__name__}: {error}") from error


@dataclass
class Toolset:
    tools: Sequence[Callable[..., Any] | Tool]


@dataclass
class Agent:
    """Configure a LiteLM model, instructions, tools, and optional Pydantic-validated output."""

    model: str | Model
    instructions: str = ""
    tools: Sequence[Callable[..., Any] | Tool] = ()
    toolsets: Sequence[Toolset] = ()
    output_type: Any = str
    model_settings: dict[str, Any] = field(default_factory=dict[str, Any])
    output_retries: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.model, str) and ":" in self.model and "/" not in self.model:
            raise ValueError("LiteLM models use provider/model, not provider:model")
        if self.output_retries < 0:
            raise ValueError("output_retries must be nonnegative")

    def output_adapter(self) -> TypeAdapter[Any]:
        return TypeAdapter(self.output_type)
