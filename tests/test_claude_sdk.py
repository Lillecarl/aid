"""Claude SDK backend: round-trip, resume, live deltas, and the permission flow. Daemon-free."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

from aid.backends.claude_sdk import SESSION_ID_FILE, ClaudeSdkBackend, open_sdk
from aid.protocol import (
    AidError,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    RemoteControl,
    TextDelta,
)
from aid.spec import AgentKind, ClaudeSdkSpec, PermissionMode

if TYPE_CHECKING:
    from pathlib import Path

    from aid.protocol import SessionEvent

pytestmark = pytest.mark.anyio

FAKE_ID = "sess-123"

PERMISSION_LINE = (
    '{"type":"control_request","request_id":"cr-1","request":{"subtype":"can_use_tool",'
    '"tool_name":"Bash","action_description":"Run echo hi","tool_use_id":"tu-1","input":{"command":"echo hi"}}}'
)

FAKE_CLAUDE = """\
#!/bin/sh
echo "$@" >> "$FAKE_CLAUDE_ARGV"
read userline
echo "$userline" >> "$FAKE_CLAUDE_STDIN"
echo '{"type":"system","subtype":"init","session_id":"sess-123"}'
case "$FAKE_CLAUDE_MODE" in
  drip)
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"part one"}]}}'
    sleep 2
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"part two"}]}}'
    echo '{"type":"result","subtype":"success","result":"part one"}'
    ;;
  ask)
    read rcline
    echo "$rcline" >> "$FAKE_CLAUDE_STDIN"
    echo '__PERMISSION_LINE__'
    if read -t 15 answer; then
      echo "$answer" >> "$FAKE_CLAUDE_STDIN"
    fi
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}'
    echo '{"type":"result","subtype":"success","result":"done"}'
    ;;
  ask_timeout)
    read rcline
    echo "$rcline" >> "$FAKE_CLAUDE_STDIN"
    echo '__PERMISSION_LINE__'
    if read -t 5 answer; then
      echo "$answer" >> "$FAKE_CLAUDE_STDIN"
    fi
    ;;
  ask_hang)
    echo '__PERMISSION_LINE__'
    sleep 30
    ;;
  rc)
    read rcline
    echo "$rcline" >> "$FAKE_CLAUDE_STDIN"
    id=$(echo "$rcline" | grep -o '"request_id": *"[^"]*"' | cut -d'"' -f4)
    echo '{"type":"control_response","response":{"subtype":"success","request_id":"'$id'","response":{"session_url":"https://claude.ai/code/session_x","connect_url":"https://claude.ai/code?environment="}}}'
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}'
    echo '{"type":"result","subtype":"success","result":"hi"}'
    ;;
  rc_fail)
    read rcline
    echo "$rcline" >> "$FAKE_CLAUDE_STDIN"
    id=$(echo "$rcline" | grep -o '"request_id": *"[^"]*"' | cut -d'"' -f4)
    echo '{"type":"control_response","response":{"subtype":"error","request_id":"'$id'","response":"/login"}}'
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}'
    echo '{"type":"result","subtype":"success","result":"hi"}'
    ;;
  linger)
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}'
    echo '{"type":"result","subtype":"success","result":"done"}'
    cat > /dev/null
    ;;
  *)
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}'
    echo '{"type":"result","subtype":"success","result":"hi there"}'
    ;;
esac
""".replace("__PERMISSION_LINE__", PERMISSION_LINE)


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


def make_spec(tmp_path: Path, fake: Path, **kwargs: object) -> ClaudeSdkSpec:
    return ClaudeSdkSpec(cwd=str(tmp_path), model="opus", command=[str(fake)], **kwargs)  # pyright: ignore[reportArgumentType] -- tests pass valid spec fields


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


async def test_allow_answer_completes_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _, stdin_log = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ask")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake, permission=PermissionMode.ALLOW), state) as backend:
        output = await backend.prompt("run echo hi", emit)
    assert output.output == "done"
    assert output.stop_reason == "success"
    [request] = [event for event in events if isinstance(event, PermissionRequest)]
    assert (request.tool_name, request.title) == ("Bash", "Run echo hi")
    assert request.input == {"command": "echo hi"}
    [decision] = [event for event in events if isinstance(event, PermissionDecision)]
    assert (decision.option_id, decision.by) == ("allow", PermissionDecider.POLICY)
    stdin = stdin_log.read_text()
    assert '"cr-1"' in stdin and '"allow"' in stdin


async def test_deny_answer_refuses_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _, stdin_log = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ask")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake, permission=PermissionMode.DENY), state) as backend:
        output = await backend.prompt("run echo hi", emit)
    assert output.stop_reason == "success"
    [decision] = [event for event in events if isinstance(event, PermissionDecision)]
    assert (decision.option_id, decision.by) == ("deny", PermissionDecider.POLICY)
    stdin = stdin_log.read_text()
    assert '"cr-1"' in stdin and '"deny"' in stdin


async def test_ask_timeout_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nobody answers in ASK mode: the wait ends refused after `permission_timeout`."""
    fake, _, stdin_log = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ask_timeout")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    spec = make_spec(tmp_path, fake, permission=PermissionMode.ASK, permission_timeout=1)
    async with open_sdk(spec, state) as backend:
        output = await backend.prompt("run echo hi", emit)
    assert output.output == ""
    [decision] = [event for event in events if isinstance(event, PermissionDecision)]
    assert (decision.option_id, decision.by) == ("deny", PermissionDecider.TIMEOUT)
    assert '"deny"' in stdin_log.read_text()


async def test_cancelled_ask_settles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`cancel` during an ASK wait settles the turn without hanging."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ask_hang")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []
    asked = anyio.Event()

    async def emit(event: SessionEvent) -> None:
        events.append(event)
        if isinstance(event, PermissionRequest):
            asked.set()

    errors: list[AidError] = []
    backends: list[ClaudeSdkBackend] = []

    async def run() -> None:
        try:
            async with open_sdk(make_spec(tmp_path, fake, permission=PermissionMode.ASK), state) as backend:
                backends.append(backend)
                await backend.prompt("run echo hi", emit)
        except AidError as error:
            errors.append(error)

    with anyio.fail_after(15):
        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await asked.wait()
            await backends[0].cancel()
    assert [decision.by for decision in events if isinstance(decision, PermissionDecision)] == [
        PermissionDecider.CANCEL
    ]
    assert [error.code for error in errors] == ["sdk_cancelled"]


async def test_deltas_arrive_before_process_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The drip fake sleeps between lines: the first delta lands while the turn still runs."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "drip")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    seen: list[str] = []
    first = anyio.Event()

    async def emit(event: SessionEvent) -> None:
        if isinstance(event, TextDelta):
            seen.append(event.text)
            first.set()

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        outputs: list[Output] = []

        async def run() -> None:
            outputs.append(await backend.prompt("hello", emit))

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            with anyio.fail_after(10):
                await first.wait()
            await anyio.sleep(0.2)
            assert outputs == []
        assert seen == ["part one", "part two"]
        assert outputs[0].output == "part onepart two"
        assert outputs[0].stop_reason == "success"


async def test_result_ends_the_turn_while_the_process_lingers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real claude stays alive on open stdin after its result; the turn still ends, closing stdin behind it."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "linger")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        with anyio.fail_after(15):
            output = await backend.prompt("hello", emit)
    assert output.output == "done"
    assert output.stop_reason == "success"


def test_spec_shape() -> None:
    spec = ClaudeSdkSpec(cwd="/", model="opus")
    assert spec.kind is AgentKind.CLAUDE_SDK
    assert spec.command == ["claude"]
    assert spec.permission is PermissionMode.DENY
    with pytest.raises(ValidationError):
        ClaudeSdkSpec(cwd="/", model="opus", bogus=1)  # pyright: ignore[reportCallIssue] -- extra is forbidden
    with pytest.raises(ValidationError):
        spec.model = "haiku"


async def test_remote_control_url_emitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The RC enable goes out after the user line; its answer emits the bridge URLs."""
    fake, _, stdin_log = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "rc")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        output = await backend.prompt("hello", emit)
    assert output.output == "hi"
    stdin = stdin_log.read_text()
    assert '"remote_control"' in stdin and '"enabled":true' in stdin.replace(" ", "")
    [remote] = [event for event in events if isinstance(event, RemoteControl)]
    assert remote.session_url == "https://claude.ai/code/session_x"
    assert remote.connect_url == "https://claude.ai/code?environment="


async def test_remote_control_failure_is_loud_in_logs_not_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused bridge enable warns and continues: the phone link is additive, not the work."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "rc_fail")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        output = await backend.prompt("hello", emit)
    assert output.output == "hi"
    assert [event for event in events if isinstance(event, RemoteControl)] == []
