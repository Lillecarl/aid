"""Claude SDK backend skeleton: round-trip, resume, and session-id persistence. Daemon-free."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

from aid.backends.claude_sdk import SESSION_ID_FILE, open_sdk
from aid.protocol import AidError, TextDelta
from aid.spec import AgentKind, ClaudeSdkSpec, PermissionMode

if TYPE_CHECKING:
    from pathlib import Path

    from aid.protocol import SessionEvent

pytestmark = pytest.mark.anyio

FAKE_ID = "sess-123"

FAKE_CLAUDE = """\
#!/bin/sh
echo "$@" >> "$FAKE_CLAUDE_ARGV"
cat >> "$FAKE_CLAUDE_STDIN"
echo '{"type":"system","subtype":"init","session_id":"sess-123"}'
if [ "$FAKE_CLAUDE_MODE" = permission ]; then
  echo '{"type":"control_request","request":{"subtype":"can_use_tool","tool_name":"Bash"},"request_id":"r1"}'
else
  echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}'
  echo '{"type":"result","subtype":"success","result":"hi there"}'
fi
"""


def write_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A fake `claude`: logs argv and stdin, prints canned NDJSON with init and result."""
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    argv_log = tmp_path / "argv.log"
    stdin_log = tmp_path / "stdin.log"
    argv_log.write_text("")
    stdin_log.write_text("")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV", str(argv_log))
    monkeypatch.setenv("FAKE_CLAUDE_STDIN", str(stdin_log))
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    return fake, argv_log, stdin_log


def make_spec(tmp_path: Path, fake: Path) -> ClaudeSdkSpec:
    return ClaudeSdkSpec(cwd=str(tmp_path), model="opus", command=[str(fake)])


async def test_prompt_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, argv_log, stdin_log = write_fake(tmp_path, monkeypatch)
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        assert backend.started().agent == "claude-sdk"
        assert backend.started().model == "opus"
        assert backend.started().agent_session is None
        assert backend.started().resumed is False
        output = await backend.prompt("hello", emit)
        assert backend.started().agent_session == FAKE_ID
    assert output.output == "hi there"
    assert output.stop_reason == "success"
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["hi there"]
    assert "hello" in stdin_log.read_text()
    assert "--resume" not in argv_log.read_text()
    assert "--model opus" in argv_log.read_text()
    assert (tmp_path / "state" / SESSION_ID_FILE).read_text() == FAKE_ID


async def test_resume_argv_on_second_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, argv_log, _ = write_fake(tmp_path, monkeypatch)
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()

    async def emit(event: SessionEvent) -> None:
        return None

    async with open_sdk(make_spec(tmp_path, fake), state) as first:
        await first.prompt("hello", emit)
    async with open_sdk(make_spec(tmp_path, fake), state) as second:
        assert second.started().resumed is True
        assert second.started().agent_session == FAKE_ID
        await second.prompt("again", emit)
    first_argv, second_argv = argv_log.read_text().splitlines()
    assert "--resume" not in first_argv
    assert f"--resume {FAKE_ID}" in second_argv


async def test_can_use_tool_raises_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "permission")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()

    async def emit(event: SessionEvent) -> None:
        return None

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        with pytest.raises(AidError, match="not_implemented"):
            await backend.prompt("run echo hi", emit)


def test_spec_shape() -> None:
    spec = ClaudeSdkSpec(cwd="/", model="opus")
    assert spec.kind is AgentKind.CLAUDE_SDK
    assert spec.command == ["claude"]
    assert spec.permission is PermissionMode.DENY
    with pytest.raises(ValidationError):
        ClaudeSdkSpec(cwd="/", model="opus", bogus=1)
    with pytest.raises(ValidationError):
        spec.model = "haiku"
