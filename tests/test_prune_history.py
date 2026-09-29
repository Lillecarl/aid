from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from aid.backends.pydantic_ai import (
    THOUGHT_LIMIT,
    _send_policy,  # pyright: ignore[reportPrivateUsage]
    prune_history,
)

if TYPE_CHECKING:
    from pydantic_ai.models.function import AgentInfo

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


async def test_policy_filters_wire_but_not_store() -> None:
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

    agent: Agent[Any, Any] = Agent(FunctionModel(record), capabilities=[ProcessHistory(_send_policy(2))])
    result = await agent.run("go", message_history=history)
    assert result.output == "done"
    assert len(seen) == 1
    wire_thought = seen[0][1].parts[0]
    assert isinstance(wire_thought, ThinkingPart)
    assert wire_thought.signature is None
    stored = result.all_messages()
    assert len(stored) == len(history) + 2
    stored_thought = stored[1].parts[0]
    assert isinstance(stored_thought, ThinkingPart)
    assert stored_thought.signature is None
    new_response = stored[-1]
    assert isinstance(new_response, ModelResponse)
    assert new_response.parts == [TextPart(content="done")]
