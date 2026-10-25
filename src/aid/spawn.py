"""A spawned child's spec from its parent's (Lillecarl/aid#50, phase 3): never looser than the parent.

Pure: the spawn plugin gathers the parent's spec, its offered tools, its depth and its children, and this decides.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from aid.protocol import AidError
from aid.spec import NativeSpec, ToolPattern

if TYPE_CHECKING:
    from aid.spec import AgentSpec

MAX_CHILDREN: Final = 8
"""Live children one session may hold."""
MAX_DEPTH: Final = 3
"""Ancestors a spawned session may have: a child of a child of a child spawns no more."""
LABEL_CHARS: Final = 16
PARENT_CHARS: Final = 64 - 1 - LABEL_CHARS
"""Characters of the parent's name a child keeps, so `<parent>.<label>` fits the 64 of a name."""


class SpawnRequest(BaseModel):
    """What a session asks for its child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Annotated[str, Field(pattern=rf"^[a-z0-9][a-z0-9-]{{0,{LABEL_CHARS - 1}}}$")]
    instructions: Annotated[str, Field(min_length=1)]
    tools: list[ToolPattern]
    model: Annotated[str, Field(pattern=r"^\S+/\S+$")] | None = None


def child_name(parent: str, label: str) -> str:
    return f"{parent[:PARENT_CHARS]}.{label}"


def _provider(model: str | None) -> str | None:
    return None if model is None else model.split("/", 1)[0]


def child_spec(
    parent_name: str, parent: AgentSpec, offered: list[str], request: SpawnRequest, *, depth: int, children: int
) -> NativeSpec:
    """The child's spec, or `AidError` saying why not. `offered`: the names the parent is offered now; `depth`: the
    parent's ancestors; `children`: its live children.

    The child runs `request.instructions` on the parent's model or one `spawn_models` allows. Its ceiling is the
    offered names its patterns match, escaped: a tool the parent gains later never reaches it. Everything else that
    bounds the parent (modes, sandbox, hooks, supervisor, environment, `spawn_models`) it inherits as is."""
    if not isinstance(parent, NativeSpec):
        raise AidError("unsupported", "only native sessions spawn")
    if depth >= MAX_DEPTH:
        raise AidError("forbidden", f"a session with {depth} ancestors spawns no more (limit {MAX_DEPTH})")
    if children >= MAX_CHILDREN:
        raise AidError("forbidden", f"this session has {children} children, the limit; delete one first")
    model = request.model or parent.model
    if model is None:
        raise AidError("bad_args", "this session runs its agent's own model: name the child's model")
    if model != parent.model and not any(re.fullmatch(p, model) for p in parent.spawn_models or []):
        raise AidError("forbidden", f"a child of this session may not run on {model}")
    if unmatched := [p for p in request.tools if not any(re.fullmatch(p, name) for name in offered)]:
        raise AidError("forbidden", f"these patterns match none of this session's tools: {unmatched}")
    names = sorted({name for name in offered if any(re.fullmatch(p, name) for p in request.tools)})
    same_provider = _provider(model) == _provider(parent.model)
    return NativeSpec.model_validate(
        {
            **parent.model_dump(),
            "agent": None,
            "target": None,
            "instructions": request.instructions,
            "model": model,
            "model_settings": parent.model_settings if same_provider else {},
            "max_context": parent.max_context if model == parent.model else None,
            "tools": [re.escape(name) for name in names],
            "seed_history": None,
            "parent": parent_name,
            "tags": {},
        }
    )
