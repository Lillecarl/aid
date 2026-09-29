from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from aid.protocol import TextDelta, ThoughtDelta, ToolCall, ToolDiff, Usage
from aid.transcript import (
    TranscriptFollower,
    TurnEnded,
    TurnUsage,
    find_transcript,
    human_prompt,
    items_from_entry,
    last_version,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


def assistant(block: dict[str, Any], stop_reason: str = "tool_use") -> dict[str, Any]:
    return {"type": "assistant", "message": {"id": "msg_1", "stop_reason": stop_reason, "content": [block]}}


def tool_result(tool_use_id: str, result: dict[str, Any]) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": tool_use_id}
    return {"type": "user", "toolUseResult": result, "message": {"content": [block]}}


def billed(message_id: str, model: str, block: dict[str, Any]) -> dict[str, Any]:
    usage = {"input_tokens": 1, "output_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5}
    message = {"id": message_id, "model": model, "usage": usage, "content": [block]}
    return {"type": "assistant", "message": message}


def test_turn_usage_counts_each_message_once() -> None:
    tally = TurnUsage()
    for entry in (
        billed("msg_a", "claude-x", {"type": "thinking", "thinking": "hm"}),
        billed("msg_a", "claude-x", {"type": "text", "text": "a"}),
        billed("msg_a", "claude-x", {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}),
        {"type": "user", "message": {"content": "not billed"}},
        billed("msg_b", "claude-y", {"type": "text", "text": "b"}),
        billed("msg_c", "<synthetic>", {"type": "text", "text": "API error"}),
    ):
        tally.add(entry)
    assert tally.usage() == Usage(
        input_tokens=3,
        output_tokens=30,
        cache_read_tokens=300,
        cache_write_tokens=15,
        thought_tokens=0,
        requests=3,
        models=["claude-x", "claude-y"],
    )


@pytest.mark.parametrize(
    ("entry", "items"),
    [
        (assistant({"type": "text", "text": "hi"}, "end_turn"), [TextDelta(text="hi")]),
        (assistant({"type": "thinking", "thinking": "hmm"}), [ThoughtDelta(text="hmm")]),
        (
            assistant({"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}),
            [
                ToolCall(
                    tool_call_id="toolu_1", title="Bash", kind="execute", status="in_progress", input={"command": "ls"}
                )
            ],
        ),
        (
            assistant({"type": "tool_use", "id": "toolu_4", "name": "Read", "input": {"file_path": "/a.py"}}),
            [
                ToolCall(
                    tool_call_id="toolu_4",
                    title="Read",
                    kind="read",
                    status="in_progress",
                    input={"file_path": "/a.py"},
                    paths=["/a.py"],
                )
            ],
        ),
        (
            tool_result(
                "toolu_5", {"filePath": "/a.py", "oldString": "x = 1", "newString": "x = 2", "originalFile": "big"}
            ),
            [
                ToolCall(
                    tool_call_id="toolu_5", status="completed", diffs=[ToolDiff(path="/a.py", old="x = 1", new="x = 2")]
                )
            ],
        ),
        (
            tool_result("toolu_6", {"type": "create", "filePath": "/b.py", "content": "new", "originalFile": None}),
            [ToolCall(tool_call_id="toolu_6", status="completed", diffs=[ToolDiff(path="/b.py", old=None, new="new")])],
        ),
        (
            tool_result("toolu_7", {"type": "update", "filePath": "/b.py", "content": "two", "originalFile": "one"}),
            [
                ToolCall(
                    tool_call_id="toolu_7", status="completed", diffs=[ToolDiff(path="/b.py", old="one", new="two")]
                )
            ],
        ),
        (
            {
                "type": "user",
                "toolUseResult": {},
                "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
            },
            [ToolCall(tool_call_id="toolu_1", status="completed", output="ok")],
        ),
        (
            {
                "type": "user",
                "toolUseResult": {},
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_3",
                            "content": [
                                {"type": "text", "text": "a"},
                                {"type": "image"},
                                {"type": "text", "text": "b"},
                            ],
                        }
                    ]
                },
            },
            [ToolCall(tool_call_id="toolu_3", status="completed", output="a\nb")],
        ),
        (
            {
                "type": "user",
                "toolUseResult": "Error",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_2", "is_error": True}]},
            },
            [ToolCall(tool_call_id="toolu_2", status="failed")],
        ),
        ({"type": "system", "subtype": "turn_duration", "durationMs": 5}, [TurnEnded()]),
        (
            {"type": "user", "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]}},
            [TurnEnded(interrupted=True)],
        ),
        ({"type": "user", "message": {"content": "a prompt"}}, []),
        ({"type": "system", "subtype": "stop_hook_summary"}, []),
        ({**assistant({"type": "text", "text": "sub"}), "isSidechain": True}, []),
        ({"type": "attachment"}, []),
    ],
)
def test_items_from_entry(entry: dict[str, Any], items: list[object]) -> None:
    assert items_from_entry(entry) == items


def test_find_transcript(tmp_path: Path) -> None:
    project = tmp_path / "projects" / "-some-dir"
    project.mkdir(parents=True)
    (project / "abc.jsonl").write_text("")
    assert find_transcript(tmp_path, "abc") == project / "abc.jsonl"
    assert find_transcript(tmp_path, "nope") is None


@pytest.mark.parametrize(
    ("entry", "text"),
    [
        ({"type": "user", "origin": {"kind": "human"}, "message": {"content": "typed"}}, "typed"),
        (
            {"type": "user", "origin": {"kind": "human"}, "message": {"content": [{"type": "text", "text": "pasted"}]}},
            "pasted",
        ),
        ({"type": "user", "isMeta": True, "origin": {"kind": "channel"}, "message": {"content": "<channel>"}}, None),
        ({"type": "user", "origin": {"kind": "task-notification"}, "message": {"content": "<task-notif>"}}, None),
        ({"type": "user", "message": {"content": "<command-name>/compact</command-name>"}}, None),
        ({"type": "user", "isMeta": True, "origin": {"kind": "human"}, "message": {"content": "hook feedback"}}, None),
        (assistant({"type": "text", "text": "hi"}), None),
    ],
)
def test_human_prompt(entry: dict[str, Any], text: str | None) -> None:
    assert human_prompt(entry) == text


def test_last_version(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    lines = [{"type": "user", "version": "2.1.1"}, {"type": "assistant", "version": "2.1.283"}, {"type": "summary"}]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + '{"version": "cut sho')
    assert last_version(path) == "2.1.283"
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert last_version(empty) is None


async def test_follower_joins_partial_lines_and_sees_late_writes(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_bytes(json.dumps({"n": 0}).encode() + b"\n")
    follower = TranscriptFollower(path)
    seen: list[dict[str, Any]] = []

    async def write() -> None:
        with path.open("ab") as f:
            line = json.dumps({"n": 1}).encode()
            f.write(line[:4])
            f.flush()
            await anyio.sleep(0.1)
            f.write(line[4:] + b"\n" + json.dumps({"n": 2}).encode() + b"\n")

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(write)
            async for entry in follower.follow():
                seen.append(entry)
                if entry["n"] == 2:
                    break
    assert seen == [{"n": 0}, {"n": 1}, {"n": 2}]


async def test_follower_starts_at_offset(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    first = json.dumps({"n": 0}).encode() + b"\n"
    path.write_bytes(first + json.dumps({"n": 1}).encode() + b"\n")
    assert await TranscriptFollower(path, offset=len(first)).read_new() == [{"n": 1}]
