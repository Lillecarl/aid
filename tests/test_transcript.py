from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from aid.protocol import TextDelta, ThoughtDelta, ToolCall
from aid.transcript import TranscriptFollower, TurnEnded, find_transcript, items_from_entry

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


def assistant(block: dict[str, Any], stop_reason: str = "tool_use") -> dict[str, Any]:
    return {"type": "assistant", "message": {"id": "msg_1", "stop_reason": stop_reason, "content": [block]}}


@pytest.mark.parametrize(
    ("entry", "items"),
    [
        (assistant({"type": "text", "text": "hi"}, "end_turn"), [TextDelta(text="hi")]),
        (assistant({"type": "thinking", "thinking": "hmm"}), [ThoughtDelta(text="hmm")]),
        (
            assistant({"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}),
            [ToolCall(tool_call_id="toolu_1", title="Bash", status="in_progress")],
        ),
        (
            {
                "type": "user",
                "toolUseResult": {},
                "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
            },
            [ToolCall(tool_call_id="toolu_1", status="completed")],
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
