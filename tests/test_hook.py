from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING

import anyio
import pytest

from aid import hook
from aid.protocol import Hook, RequestAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio


def test_the_request_is_aids_hook_request() -> None:
    payload = {"hook_event_name": "Stop", "session_id": "s", "nested": {"a": [1, None]}}
    parsed = RequestAdapter.validate_python(hook.request("tty", "Stop", payload))
    assert isinstance(parsed, Hook)
    assert (parsed.session, parsed.event, parsed.payload) == ("tty", "Stop", payload)


async def run_hook(control: str, session: str, event: str) -> tuple[int, str, float]:
    started = time.monotonic()
    result = await anyio.run_process(
        [sys.executable, hook.__file__, control, session, event],
        input=json.dumps({"hook_event_name": event}).encode(),
        check=False,
    )
    return result.returncode, result.stdout.decode(), time.monotonic() - started


async def test_no_daemon_means_nothing_to_say(tmp_path: Path) -> None:
    code, out, took = await run_hook(f"ipc://{tmp_path / 'none.sock'}", "tty", "UserPromptSubmit")
    assert (code, out) == (0, "")
    assert took < hook.QUICK_WAIT


async def test_an_unknown_session_means_nothing_to_say(daemon: Paths) -> None:
    code, out, _ = await run_hook(daemon.control, "nobody", "Stop")
    assert (code, out) == (0, "")
