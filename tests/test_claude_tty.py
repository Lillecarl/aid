from __future__ import annotations

import json
import os
import shlex
import sys
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import pytest
from libpymux import Server

import aid
from aid.backends.claude_tty import (
    HOOK_EVENTS,
    HOOK_SCRIPT,
    Screen,
    classify,
    claude_argv,
    hook_settings,
    launcher_script,
    read_settings,
    split_settings,
)
from aid.mcp import claude_config
from aid.paths import ENV_CHANNEL, ENV_RUNTIME_DIR, ENV_SESSION, ENV_STATE_DIR
from aid.protocol import (
    AidError,
    HistoryItem,
    Lifecycle,
    MessageEntry,
    Output,
    PaneView,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    Started,
    TextDelta,
    ToolCall,
    Usage,
)
from aid.spec import ClaudeTtySpec, McpHttp
from tests.conftest import acp_spec, fake_spec, needs_pymux
from tests.fake_claude import COMPACT_SUMMARY, COUNT, MODEL, NOTICE, PERMISSION_INPUT
from tests.test_tools import Rpc

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

TIMEOUT = 30


@pytest.mark.parametrize(
    ("capture", "screen"),
    [
        ("", Screen.STARTING),
        (" Do you trust this folder?\n ❯ No, exit\n   Yes, I trust this folder\n", Screen.TRUST),
        ('header\n────\n❯ Try "refactor <filepath>"\n────\n', Screen.READY),
        ("header\n────\n❯\n────\n", Screen.READY),
        (
            "  WARNING: Loading development channels\n\n  --dangerously-load-development-channels is for local\n"
            "  Channels: server:aid\n\n  ❯ 1. I am using this for local development\n    2. Exit\n",
            Screen.CHANNELS,
        ),
        ("loading ❯ not at the start\n", Screen.STARTING),
    ],
)
def test_classify(capture: str, screen: Screen) -> None:
    assert classify(capture) is screen


def test_launcher_script_quotes_everything() -> None:
    script = launcher_script(
        {"A": "x y", "B": "it's", "TERM": "not-the-pane"}, "/w d", ["claude", "--session-id", "id;rm"]
    )
    terminal = " ".join(f'${{{name}+"{name}=${name}"}}' for name in ("TERM", "COLORTERM", "TERMINFO", "TERMINFO_DIRS"))
    assert script.splitlines() == [
        "#!/bin/sh",
        "cd '/w d' || exit 1",
        f"exec env -i {terminal} 'A=x y' 'B=it'\"'\"'s' claude --session-id 'id;rm'",
    ]


def test_variadic_options_go_before_other_arguments() -> None:
    spec = ClaudeTtySpec(cwd="/", args=["first prompt"])
    assert claude_argv(spec, spec.args, "id", resume=False, mcp_config="/s/mcp.json", settings="/s/settings.json") == [
        "claude",
        "--mcp-config",
        "/s/mcp.json",
        "--allowedTools",
        "mcp__aid",
        "--dangerously-load-development-channels",
        "server:aid",
        "--settings",
        "/s/settings.json",
        "--session-id",
        "id",
        "first prompt",
    ]
    without = spec.model_copy(update={"aid_tools": False})
    assert claude_argv(without, ["x"], "id", resume=True, mcp_config=None, settings="/s") == [
        "claude",
        "--settings",
        "/s",
        "--resume",
        "id",
        "x",
    ]


def test_a_persons_settings_join_aids_hooks(tmp_path: Path) -> None:
    value, rest = split_settings(["--model", "opus", "--settings", "mine.json", "--settings=ignored", "hi"])
    assert (value, rest) == ("ignored", ["--model", "opus", "hi"])
    assert split_settings(["--settings", "a.json"]) == ("a.json", [])

    (tmp_path / "mine.json").write_text(json.dumps({"model": "opus", "hooks": {"Stop": [{"hooks": ["theirs"]}]}}))
    own = read_settings("mine.json", str(tmp_path))
    assert read_settings('{"model": "haiku"}', "/nowhere") == {"model": "haiku"}
    merged = hook_settings("ipc:///run/aid/control.sock", "s1", own)

    assert merged["model"] == "opus"
    assert set(merged["hooks"]) == set(HOOK_EVENTS)
    assert merged["hooks"]["Stop"][0] == {"hooks": ["theirs"]}
    [aid_stop] = merged["hooks"]["Stop"][1]["hooks"]
    assert shlex.split(aid_stop["command"])[1:] == [HOOK_SCRIPT, "ipc:///run/aid/control.sock", "s1", "Stop"]


pytestmark = pytest.mark.anyio


@needs_pymux
async def test_prompt_round_trip(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            plain = await session.run("hello")
            multi = await session.run("one\ntwo; three")
            counted = await session.run("count")
            events = [e async for e in session.stream("tool")]
    assert plain.output == "echo: hello"
    assert plain.stop_reason == "end_turn"
    assert multi.output == "echo: one\ntwo; three"
    assert counted.text == "".join(f"{i} " for i in range(COUNT))
    two = 2  # the Bash call's message and the answer's
    assert events == [
        ToolCall(
            tool_call_id="toolu_fake", title="Bash", kind="execute", status="in_progress", input={"command": "true"}
        ),
        ToolCall(tool_call_id="toolu_fake", status="completed", output="ok"),
        TextDelta(text="ran it"),
        Usage(
            input_tokens=3 * two,
            output_tokens=5 * two,
            cache_read_tokens=100 * two,
            cache_write_tokens=7 * two,
            thought_tokens=2 * two,
            requests=two,
            models=[MODEL],
        ),
        Output(output="ran it", stop_reason="end_turn"),
    ]


@needs_pymux
async def test_hooks_tell_what_claude_does(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    own = tmp_path / "own-settings.json"
    own.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "touch own-hook-ran"}]}]}}))
    spec = fake_spec(tmp_path, pymux_socket).model_copy(update={"args": ["--settings", str(own)]})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", spec)
            result = await session.run("notify")
            while (status := await session.status()).working:  # noqa: ASYNC110 -- Claude runs Stop after the turn ends
                await anyio.sleep(0.1)
            [info] = await client.sessions()
    assert result.text == "notified"
    assert status.attention == NOTICE
    assert (info.working, info.attention) == (False, NOTICE)
    assert (tmp_path / "own-hook-ran").exists()


@needs_pymux
@pytest.mark.parametrize(("option", "said"), [("allow", "allowed"), ("deny", "denied: A person refused this in aid.")])
async def test_claude_asks_aid_for_permission(
    daemon: Paths, tmp_path: Path, pymux_socket: str, option: str, said: str
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            events: list[aid.SessionEvent] = []
            waiting: list[PermissionRequest] = []
            async for event in session.stream("permission"):
                events.append(event)
                if isinstance(event, PermissionRequest):
                    waiting = (await session.status()).permissions
                    await session.answer(event.request_id, option)
            after = (await session.status()).permissions
    request = next(e for e in events if isinstance(e, PermissionRequest))
    assert (request.tool_call_id, request.tool_name, request.title) == (
        "toolu_perm",
        "Bash",
        "Remove the build directory",
    )
    assert request.input == PERMISSION_INPUT
    assert waiting == [request]
    assert after == []
    assert PermissionDecision(request_id=request.request_id, option_id=option, by=PermissionDecider.PERSON) in events
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == said


@needs_pymux
async def test_a_permission_answered_in_the_pane(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            events = [event async for event in session.stream("permission-pane")]
            after = (await session.status()).permissions
    request = next(e for e in events if isinstance(e, PermissionRequest))
    assert PermissionDecision(request_id=request.request_id, option_id=None, by=PermissionDecider.TERMINAL) in events
    assert after == []
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "the hook said ''"


async def typed_items(session: aid.Session) -> list[HistoryItem]:
    """History after the typed prompt, once its turn has ended."""
    while True:
        items = [e.item for e in (await session.history(limit=1000)).entries]
        typed = next((i for i, item in enumerate(items) if item == PromptEntry(text="typed")), None)
        if typed is not None and any(isinstance(item, Output) for item in items[typed:]):
            return items[typed:]
        await anyio.sleep(0.1)


@needs_pymux
async def test_a_turn_typed_into_the_pane_is_recorded(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            first = await session.run("hello")
            address = await session.pane()
            pane = next(p for p in Server(address.socket).panes if p.id == address.pane)
            await anyio.to_thread.run_sync(pane.send_keys, "typed", False)
            await anyio.to_thread.run_sync(pane.send_key, "Enter")
            typed = await typed_items(session)
            after = await session.run("after")
    assert first.text == "echo: hello"
    prompt, text, usage, output = typed
    assert (prompt, text, output) == (
        PromptEntry(text="typed"),
        TextDelta(text="echo: typed"),
        Output(output="echo: typed", stop_reason="end_turn"),
    )
    assert isinstance(usage, Usage)
    assert usage.models == [MODEL]
    # aid's prompts still get their own turns, and the typed one is not taken for aid's.
    assert after.text == "echo: after"


async def type_command(session: aid.Session, command: str) -> None:
    address = await session.pane()
    pane = next(p for p in Server(address.socket).panes if p.id == address.pane)
    await anyio.to_thread.run_sync(pane.send_keys, command, False)
    await anyio.to_thread.run_sync(pane.send_key, "Enter")


async def recorded(session: aid.Session, kind: type[HistoryItem], after: int = -1) -> HistoryItem:
    while True:
        for entry in (await session.history(after=after, limit=1000)).entries:
            if isinstance(entry.item, kind):
                return entry.item
        await anyio.sleep(0.1)


@needs_pymux
async def test_clear_compact_and_exit(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            await session.run("hello")
            first = (await session.status()).agent_session
            mark = (await session.history()).entries[-1].seq

            await type_command(session, "/clear")
            cleared = await recorded(session, Lifecycle, mark)
            restarted = await recorded(session, Started, mark)
            after = await session.run("after the clear")
            status = await session.status()

            mark = (await session.history()).entries[-1].seq
            await type_command(session, "/compact")
            compacted = await recorded(session, Lifecycle, mark)

            mark = (await session.history()).entries[-1].seq
            await type_command(session, "/exit")
            ended = await recorded(session, Lifecycle, mark)
            while (await session.status()).running:  # noqa: ASYNC110 -- the worker ends after Claude
                await anyio.sleep(0.1)
    assert cleared == Lifecycle(event="cleared")
    assert isinstance(restarted, Started)
    assert restarted.agent_session not in (None, first)
    assert (restarted.resumed, restarted.model) == (False, MODEL)
    assert after.text == "echo: after the clear"
    assert status.agent_session == restarted.agent_session
    session_dir = daemon.session_dir("tty")
    assert (session_dir / "claude-session-id").read_text() == restarted.agent_session
    assert compacted == Lifecycle(event="compacted", detail="manual", summary=COMPACT_SUMMARY)
    assert ended == Lifecycle(event="ended", detail="prompt_input_exit")


@needs_pymux
async def test_cancel_interrupts_the_turn(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            events: list[aid.SessionEvent] = []
            async for event in session.stream("slow"):
                events.append(event)
                if event == TextDelta(text="waiting"):
                    await session.cancel()
            after = await session.run("again")
    assert events[-1] == Output(output="waiting", stop_reason="cancelled")
    assert after.output == "echo: again"


@needs_pymux
async def test_stop_resumes_the_same_transcript(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            await session.run("first")
            await session.stop()
            second = await session.run("second")
    transcripts = list((tmp_path / "claude-config" / "projects").glob("*/*.jsonl"))
    assert len(transcripts) == 1
    assert second.output == "echo: second"


@needs_pymux
async def test_trust_dialog(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    env = {"FAKE_CLAUDE_TRUST": "1"}
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(AidError):
                await client.create("untrusted", fake_spec(tmp_path, pymux_socket, env=env))
            session = await client.create("trusted", fake_spec(tmp_path, pymux_socket, env=env, trust_cwd=True))
            result = await session.run("hi")
    assert result.output == "echo: hi"


@needs_pymux
async def test_mcp_servers_reach_claude(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    servers = [McpHttp(name="web", url="http://127.0.0.1:1/mcp", headers={"Authorization": "Bearer t"})]
    spec = fake_spec(tmp_path, pymux_socket).model_copy(update={"mcp_servers": servers})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", spec)
            result = await session.run("mcp")
    config = next(daemon.state_dir.rglob("mcp.json"))
    got = json.loads(result.text)["mcpServers"]
    builtin = got.pop("aid")
    assert got == claude_config(servers)["mcpServers"]
    assert (builtin["command"], builtin["args"]) == (sys.executable, ["-m", "aid.mcp_server"])
    assert builtin["env"]["AID_SESSION"] == "tty"
    assert builtin["env"]["AID_CHANNEL"] == "1"
    assert config.stat().st_mode & 0o777 == 0o600


@needs_pymux
async def test_messages_reach_claude_through_its_channel(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    env = {
        ENV_SESSION: "tty",
        ENV_CHANNEL: "1",
        ENV_RUNTIME_DIR: str(daemon.runtime_dir),
        ENV_STATE_DIR: str(daemon.state_dir),
    }
    command = [sys.executable, "-m", "aid.mcp_server"]
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("tty", fake_spec(tmp_path, pymux_socket))
            async with await anyio.open_process(command, env={**os.environ, **env}) as server:
                assert server.stdin is not None
                assert server.stdout is not None
                rpc = Rpc(server.stdin, server.stdout)
                hello = {"name": "test", "version": "0"}
                init = await rpc.call(
                    "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": hello}
                )
                await rpc.notify("notifications/initialized")
                await client.send_message("tty", "ping", sender="bob")
                await client.send_message("tty", "from a person")
                first = await rpc.notification("notifications/claude/channel")
                second = await rpc.notification("notifications/claude/channel")
                await server.stdin.aclose()
            history = await client.session("tty").history()
    assert init["capabilities"]["experimental"] == {"claude/channel": {}}
    assert "send_message" in init["instructions"]
    assert first == {"content": "ping", "meta": {"from": "bob"}}
    assert second == {"content": "from a person", "meta": {}}
    assert [e.item for e in history.entries if not isinstance(e.item, Started)] == [
        MessageEntry(sender="bob", text="ping"),
        MessageEntry(sender=None, text="from a person"),
    ]


@needs_pymux
async def test_screen_is_the_pane_as_html(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            await session.run("hello")
            plain = await session.screen()
            styled = await session.screen(stylesheet=True)
            await client.create("acp", acp_spec(tmp_path))
            with pytest.raises(AidError, match="no terminal"):
                await client.session("acp").screen()
            await session.stop()
            with pytest.raises(AidError, match="stopped"):
                await session.screen()
    assert plain.html.startswith('<pre class="pyte-screen">')
    assert "fake claude" in plain.html
    assert (plain.stylesheet, plain.overlay) == (None, None)
    assert styled.stylesheet is not None
    assert "--pyte-" in styled.stylesheet


@needs_pymux
async def test_claude_gets_the_panes_terminal(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    # A TERM in aid's own environment is not the pane's terminal.
    spec = fake_spec(tmp_path, pymux_socket, env={"TERM": "not-the-pane"})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            result = await (await client.create("tty", spec)).run("term")
    assert result.output == "pyte"


@needs_pymux
async def test_panes_start_at_aids_size(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            await client.create("tty", fake_spec(tmp_path, pymux_socket))
            listed = await anyio.run_process(
                ["pymux", "-S", pymux_socket, "list-panes", "-a", "-F", "#{window_name} #{pane_width}x#{pane_height}"]
            )
    assert "aid:tty 200x50" in listed.stdout.decode().splitlines()


@needs_pymux
async def test_screen_waits_for_the_pane_to_change(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            first = await session.screen()
            started = anyio.current_time()
            quiet = await session.screen(since=first.revision, wait=0.5)
            waited = anyio.current_time() - started
            changed: list[PaneView] = []

            async with anyio.create_task_group() as tg:

                async def wait_for_it() -> None:
                    changed.append(await session.screen(since=first.revision, wait=20))

                tg.start_soon(wait_for_it)
                await session.run("hello")
            started = anyio.current_time()
            behind = await session.screen(since=first.revision, wait=20)
            answered = anyio.current_time() - started
    assert first.revision >= 0
    assert quiet.revision == first.revision
    assert 0.4 < waited < 5
    assert changed[0].revision != first.revision
    # A revision already left answers at once: nothing between a frame and the next wait is missed.
    assert behind.revision != first.revision
    assert answered < 2


@needs_pymux
async def test_pane_address(daemon: Paths, tmp_path: Path, pymux_socket: str) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            address = await client.create("tty", fake_spec(tmp_path, pymux_socket))
            where = await address.pane()
            await client.create("acp", acp_spec(tmp_path))
            with pytest.raises(AidError, match="no terminal"):
                await client.session("acp").pane()
    assert where.socket == pymux_socket
    assert where.pane.startswith("%")
