from __future__ import annotations

from acp.schema import (
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)

from aid.backends import acp, native
from aid.messages import (
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from aid.protocol import TOOL_TEXT_LIMIT, ToolCall, ToolDiff


def test_acp_tool_call_carries_input_and_paths() -> None:
    start = ToolCallStart(
        session_update="tool_call",
        tool_call_id="t1",
        title="Read a.py",
        kind="read",
        status="pending",
        raw_input={"path": "a.py"},
        locations=[ToolCallLocation(path="/w/a.py", line=3), ToolCallLocation(path="/w/b.py")],
    )
    assert acp.to_event(start) == ToolCall(
        tool_call_id="t1",
        title="Read a.py",
        kind="read",
        status="pending",
        input={"path": "a.py"},
        paths=["/w/a.py:3", "/w/b.py"],
    )


def test_acp_tool_update_carries_output_and_diffs() -> None:
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="t1",
        status="completed",
        content=[
            ContentToolCallContent(type="content", content=TextContentBlock(type="text", text="done")),
            FileEditToolCallContent(type="diff", path="/w/a.py", old_text="x = 1\n", new_text="x = 2\n"),
            FileEditToolCallContent(type="diff", path="/w/new.py", old_text=None, new_text="y\n"),
        ],
        raw_output={"ignored": "when there is text content"},
    )
    assert acp.to_event(update) == ToolCall(
        tool_call_id="t1",
        status="completed",
        output="done",
        diffs=[ToolDiff(path="/w/a.py", old="x = 1\n", new="x = 2\n"), ToolDiff(path="/w/new.py", new="y\n")],
    )


def test_acp_raw_output_is_the_fallback_and_is_clipped() -> None:
    update = ToolCallProgress(
        session_update="tool_call_update", tool_call_id="t1", raw_output="z" * (TOOL_TEXT_LIMIT + 5)
    )
    event = acp.to_event(update)
    assert isinstance(event, ToolCall)
    assert event.output is not None
    assert event.output.endswith("… 5 more characters")


def test_native_tool_events() -> None:
    call = ToolCallPart(tool_name="add", args='{"a": 1}', tool_call_id="c1")
    done = ToolReturnPart(tool_name="add", content={"sum": 3}, tool_call_id="c1")
    retry = RetryPromptPart(tool_name="add", content="a must be even", tool_call_id="c2")
    assert native.to_event(call) == ToolCall(tool_call_id="c1", title="add", status="in_progress", input={"a": 1})
    assert native.to_event(done) == ToolCall(tool_call_id="c1", status="completed", output='{"sum":3}')
    failed = native.to_event(retry)
    assert isinstance(failed, ToolCall)
    assert (failed.status, "a must be even" in (failed.output or "")) == ("failed", True)


def test_native_tool_call_carries_paths() -> None:
    call = ToolCallPart(tool_name="read", args='{"path": "a.py", "offset": 1}', tool_call_id="c1")
    assert native.to_event(call) == ToolCall(
        tool_call_id="c1",
        title="read",
        status="in_progress",
        input={"path": "a.py", "offset": 1},
        paths=["a.py"],
    )
