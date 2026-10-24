"""Claude SDK backend: round-trip, resume, live deltas, and the permission flow. Daemon-free."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

from aid.backends.claude_sdk import SESSION_ID_FILE, ClaudeSdkBackend, open_sdk
from aid.mcp import AID_TOOLS_RULE, MCP_CONFIG_FILE
from aid.protocol import (
    AidError,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    RemoteControl,
    TextDelta,
    ThoughtDelta,
    ToolCall,
)
from aid.spec import BUILTIN_MCP_SERVER, AgentKind, ClaudeSdkSpec, PermissionMode

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
echo '{"type":"system","subtype":"init","session_id":"sess-123","tools":["Bash","Read","mcp__aid__send_message"]}'
# Claude Code sends can_use_tool only to a permission prompt tool; without one it refuses by itself (2.1.310).
case " $* " in
  *" --permission-prompt-tool stdio "*) ;;
  *) case "$FAKE_CLAUDE_MODE" in ask*)
       echo '{"type":"result","subtype":"success","result":"Claude requested permissions to use Bash"}'
       exit 0 ;;
     esac ;;
esac
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
  stream)
    echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}}'
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"s"}]}}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" there"}}}'
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}'
    echo '{"type":"result","subtype":"success","result":"hi there"}'
    ;;
  tools)
    echo '{"type":"assistant","parent_tool_use_id":null,"message":{"role":"assistant","content":[{"type":"tool_use","id":"tu-9","name":"Read","input":{"file_path":"pyproject.toml"}}]}}'
    echo '{"type":"assistant","parent_tool_use_id":"tu-9","message":{"role":"assistant","content":[{"type":"tool_use","id":"tu-sub","name":"Bash","input":{"command":"ls"}}]}}'
    echo '{"type":"user","parent_tool_use_id":null,"tool_use_result":{"type":"text","file":{"filePath":"pyproject.toml"}},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-9","content":"name = aid"}]}}'
    echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"aid"}]}}'
    echo '{"type":"result","subtype":"success","result":"aid"}'
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
    assert "--include-partial-messages" in argv_log.read_text()
    # aid's own MCP server, for send_message and list_sessions, with its tools allowed.
    assert (
        f"--mcp-config {tmp_path / 'state' / MCP_CONFIG_FILE} --allowedTools {AID_TOOLS_RULE}" in argv_log.read_text()
    )
    assert BUILTIN_MCP_SERVER in json.loads((tmp_path / "state" / MCP_CONFIG_FILE).read_text())["mcpServers"]
    assert "--model opus" in argv_log.read_text()
    assert (tmp_path / "state" / SESSION_ID_FILE).read_text() == FAKE_ID


async def test_partial_messages_stream_text_and_thinking_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deltas stream as they come; the finished block's line adds the text to the output without repeating it, and
    a thinking delta with no text (Haiku's display "updates") shows nothing."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "stream")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        output = await backend.prompt("hello", emit)
    assert output.output == "hi there"
    assert [(type(e).__name__, e.text) for e in events if isinstance(e, TextDelta | ThoughtDelta)] == [
        ("ThoughtDelta", "hmm"),
        ("TextDelta", "hi"),
        ("TextDelta", " there"),
    ]


async def test_tool_calls_and_their_results_reach_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude Code's tool_use blocks and tool_result lines, as the transcript path turns them; a subagent's are its
    own conversation and stay out."""
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "tools")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    async with open_sdk(make_spec(tmp_path, fake), state) as backend:
        output = await backend.prompt("hello", emit)
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert output.output == "aid"
    assert [(c.tool_call_id, c.status) for c in calls] == [("tu-9", "in_progress"), ("tu-9", "completed")]
    assert (calls[0].title, calls[1].output) == ("Read", "name = aid")


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


async def test_tool_modes_become_claudes_own_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """allow and deny go to Claude Code as tool flags, ask as a setting; a change reaches the next prompt."""
    fake, argv_log, _ = write_fake(tmp_path, monkeypatch)
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()

    async def emit(event: SessionEvent) -> None:
        return None

    modes = {"Edit": PermissionMode.ALLOW, "WebFetch": PermissionMode.DENY, "Read": PermissionMode.ASK}
    async with open_sdk(make_spec(tmp_path, fake, tool_permissions=modes), state) as backend:
        assert backend.session_tools() == []
        await backend.prompt("hello", emit)
        tools = backend.session_tools()
        backend.set_permissions(PermissionMode.ASK, {"Read": None, "WebFetch": PermissionMode.ALLOW})
        await backend.prompt("again", emit)
    first, second = argv_log.read_text().splitlines()
    assert f"--allowedTools {AID_TOOLS_RULE} Edit --disallowedTools WebFetch" in first
    assert '--settings {"permissions": {"ask": ["Read"]}}' in first
    assert f"--allowedTools {AID_TOOLS_RULE} Edit WebFetch --verbose" in second
    assert "--settings" not in second and "--disallowedTools" not in second
    assert [(t.name, t.group) for t in tools] == [
        ("Bash", "claude"),
        ("Read", "claude"),
        ("mcp__aid__send_message", "aid"),
    ]


async def test_a_tool_set_to_ask_waits_whatever_the_session_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _, _ = write_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ask_timeout")
    state = anyio.Path(str(tmp_path / "state"))
    await state.mkdir()
    events: list[SessionEvent] = []

    async def emit(event: SessionEvent) -> None:
        events.append(event)

    spec = make_spec(
        tmp_path,
        fake,
        permission=PermissionMode.ALLOW,
        tool_permissions={"Bash": PermissionMode.ASK},
        permission_timeout=1,
    )
    async with open_sdk(spec, state) as backend:
        await backend.prompt("run echo hi", emit)
    [decision] = [event for event in events if isinstance(event, PermissionDecision)]
    assert decision.by is PermissionDecider.TIMEOUT


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
