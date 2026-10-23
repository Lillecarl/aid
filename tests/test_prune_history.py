from __future__ import annotations

import itertools
import json
import time
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest

from aid.agent import Agent
from aid.backends.native import (
    METADATA,
    THOUGHT_LIMIT,
    prune_history,
    turn_metadata,
)
from aid.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from tests.models import FunctionModel
from tests.test_native_backend import backend

if TYPE_CHECKING:
    from pathlib import Path

    from tests.models import AgentInfo

pytestmark = pytest.mark.anyio


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(
            parts=[
                ThinkingPart(content="t" * (THOUGHT_LIMIT + 100), signature="sig"),
                TextPart(content="old"),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="read", content="x" * 1000, tool_call_id="c1")]),
        ModelResponse(parts=[ThinkingPart(content="fresh", signature="live"), TextPart(content="new")]),
    ]


def test_prune_history_flags_past_but_not_present() -> None:
    history = _history()
    _request, old, _returned, new = prune_history(history, 3)
    old_thought = old.parts[0]
    assert isinstance(old_thought, ThinkingPart)
    assert old_thought.content == "t" * THOUGHT_LIMIT + "…"
    assert old_thought.signature is None
    new_thought = new.parts[0]
    assert isinstance(new_thought, ThinkingPart)
    assert (new_thought.content, new_thought.signature) == ("fresh", "live")


def test_prune_history_leaves_tool_results_alone() -> None:
    """Size is the spill mechanism's job at production time, not the send filter's."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name="read", content="x" * 1000, tool_call_id="c0")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="read", content="y" * 1000, tool_call_id="c1")]),
    ]
    assert prune_history(history, 1) == history


def test_prune_history_leaves_small_parts() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[ThinkingPart(content="brief"), TextPart(content="ok")]),
    ]
    assert prune_history(history, 2) == history


def test_prune_history_does_not_mutate_store() -> None:
    history = _history()
    prune_history(history, 3)
    thought = history[1].parts[0]
    assert isinstance(thought, ThinkingPart)
    assert thought.content == "t" * (THOUGHT_LIMIT + 100)
    assert thought.signature == "sig"


async def test_policy_filters_wire_but_not_store(tmp_path: Path) -> None:
    """The model is sent shells and clips; the finished turn itself is stored whole."""
    history = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[ThinkingPart(content="t" * (THOUGHT_LIMIT + 100), signature="sig")]),
    ]
    seen: list[list[ModelMessage]] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        seen.append(messages)
        return ModelResponse(parts=[TextPart(content="done")])

    async def emit(event: Any) -> None:
        pass

    async with backend(tmp_path, Agent(FunctionModel(record)), history=history) as session:
        result = await session.prompt("go", emit)
    stored = ModelMessagesTypeAdapter.validate_json(await anyio.Path(tmp_path / "history.json").read_bytes())
    assert result.output == "done"
    assert len(seen) == 1
    wire = seen[0]
    assert set(_metadata_of(wire)) == {"context_pct", "unixtime"}
    wire_thought = wire[1].parts[0]
    assert isinstance(wire_thought, ThinkingPart)
    assert wire_thought.signature is None
    assert len(stored) == len(history) + 2
    stored_thought = stored[1].parts[0]
    assert isinstance(stored_thought, ThinkingPart)
    assert stored_thought.signature == "sig"
    new_response = stored[-1]
    assert isinstance(new_response, ModelResponse)
    assert new_response.parts == [TextPart(content="done")]


def _metadata_of(messages: list[ModelMessage]) -> dict[str, Any]:
    """The newest metadata payload, found by its header."""
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith(METADATA):
                _, _, payload = part.content.partition("\n")
                loaded = json.loads(payload)
                assert isinstance(loaded, dict)
                return cast(dict[str, Any], loaded)
    raise AssertionError("no metadata on the wire")


def test_turn_metadata_reports_the_gauge_and_the_time() -> None:
    before = time.time()
    meta = _metadata_of([ModelRequest(parts=[turn_metadata(1000, 100_000)])])
    assert meta == {"context_pct": 1, "unixtime": meta["unixtime"]}
    assert int(before) <= meta["unixtime"] <= time.time()
    assert _metadata_of([ModelRequest(parts=[turn_metadata(1000, 100)])])["context_pct"] == 1000


async def test_sends_only_grow(tmp_path: Path) -> None:
    """Each request's wire starts with the previous one, through a tool loop and across turns: a provider's
    prefix cache and Anthropic's thinking binding both need it. The metadata rides once per turn, stored. (Past
    unsigned thinking is the exception: `prune_history` clips it once the next turn starts.)"""
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("now", "{}", f"c{len(seen)}")])

    def now() -> str:
        return "noon"

    async def emit(event: Any) -> None:
        pass

    async with backend(tmp_path, Agent(FunctionModel(respond), tools=[now])) as session:
        await session.prompt("first", emit)
        await session.prompt("second", emit)
    assert len(seen) == 4
    for before, after in itertools.pairwise(seen):
        assert after[: len(before)] == before
    stored = ModelMessagesTypeAdapter.validate_json(await anyio.Path(tmp_path / "history.json").read_bytes())
    metadata = [
        p for m in stored for p in m.parts if isinstance(p, UserPromptPart) and str(p.content).startswith(METADATA)
    ]
    assert len(metadata) == 2
