from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.client import register_plugin
from aid.guard import guard, verdict
from aid.plugins import Grant
from aid.protocol import PermissionChoice, PermissionDecider, PermissionDecision, PermissionRequest
from aid.spec import PermissionMode
from tests.conftest import acp_spec

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from aid.paths import Paths

OPTIONS = [
    PermissionChoice(option_id="yes", name="Yes", kind="allow_once"),
    PermissionChoice(option_id="no", name="No", kind="reject_once"),
]


def _request(tool_input: JsonValue) -> PermissionRequest:
    return PermissionRequest(request_id="r", tool_call_id="t", input=tool_input, options=OPTIONS)


@pytest.mark.parametrize(
    ("tool_input", "answer"),
    [
        ({"argv": ["ls", "-la"], "cwd": "."}, "yes"),
        ({"command": "rg -n TODO src"}, "yes"),
        ({"argv": ["rm", "-rf", "/"]}, None),
        ({"command": "ls; rm -rf ~"}, None),
        ({"command": "cat $(which sh)"}, None),
        ({"command": "cat 'unclosed"}, None),
        ({"command": "find . -delete"}, None),
        ({"file_path": "notes.md"}, None),
        ("ls", None),
    ],
)
def test_the_guard_allows_what_plainly_reads(tool_input: JsonValue, answer: str | None) -> None:
    assert verdict(_request(tool_input)) == answer


def test_no_allow_option_leaves_it() -> None:
    request = _request({"argv": ["ls"]}).model_copy(update={"options": OPTIONS[1:]})
    assert verdict(request) is None


@pytest.mark.anyio
async def test_the_guard_answers_as_a_plugin(daemon: Paths, tmp_path: Path) -> None:
    read = ""
    with anyio.fail_after(30):
        async with aid.connect(daemon) as control:
            session = await control.create("ask", acp_spec(tmp_path, PermissionMode.ASK))
            await register_plugin(control, daemon, "guard", frozenset({Grant.READ, Grant.PERMISSIONS}))
            async with aid.connect(daemon, plugin="guard") as plugin, anyio.create_task_group() as tg:
                tg.start_soon(guard, plugin)
                read = (await session.run("permission-to-read")).text
                # `rm -rf /`: the guard leaves it, and a person answers.
                async for event in session.stream("permission"):
                    if isinstance(event, PermissionRequest):
                        await anyio.sleep(0.5)  # Time for the guard to judge it.
                        assert (await session.status()).permissions, "the guard answered rm -rf /"
                        await session.answer(event.request_id, "no")
                tg.cancel_scope.cancel()
            history = await session.history()
    # Out here, not in the block: a failure inside the task group comes out as an ExceptionGroup.
    decisions = [e.item for e in history.entries if isinstance(e.item, PermissionDecision)]
    assert read == "yes"
    assert [(d.option_id, d.by, d.plugin) for d in decisions] == [
        ("yes", PermissionDecider.PLUGIN, "guard"),
        ("no", PermissionDecider.PERSON, None),
    ]
