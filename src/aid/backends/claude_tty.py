"""Backend for interactive Claude Code, run in a pymux window and driven through libpymux.

Input goes in as a person would give it: a bracketed paste, then Enter. Output comes back from the session
transcript (`aid.transcript`), because the terminal has no structured stream. Measured against Claude Code
2.1.283 in pymux:

- A new cwd opens a trust dialog whose default answer is "No, exit". Its cursor is also a `❯`.
- The transcript file appears on the first prompt, not at startup.
- A typed newline is Enter and submits. A bracketed paste keeps newlines and semicolons.
- `--resume <id>` appends to the same transcript.
- Escape interrupts a turn; the transcript then holds "[Request interrupted by user]" and no turn_duration.

And on 2.1.284, with `--dangerously-load-development-channels server:aid`:

- After the trust dialog comes "WARNING: Loading development channels", whose default is to go on.
- A channel event wakes an idle Claude. Its turn is in the transcript as a `user` entry with `isMeta` and
  `origin.kind == "channel"`, ended by turn_duration as usual. It is recorded like a typed turn, without a
  prompt: the daemon recorded the message when it arrived.

Hooks, measured on 2.1.284: aid passes its own `--settings` file (`hook_settings`) whose hooks run `aid/hook.py`.
Claude takes only the last `--settings`, so a person's own `--settings` in the args is merged into aid's.

- Payloads carry session_id, transcript_path, cwd, hook_event_name, and mostly prompt_id and permission_mode.
- UserPromptSubmit has `prompt`; Stop has `last_assistant_message`; Notification has `message` and
  `notification_type` (idle_prompt a minute after a turn, permission_prompt with a permission dialog).
- PermissionRequest has tool_name, tool_input and permission_suggestions, and no tool_use_id. The pane shows its
  own dialog while the hook runs; whichever answers first wins. When the pane wins, Claude neither kills the hook
  nor reads its late answer. The answer is `hookSpecificOutput.decision.behavior` allow or deny.
- `/clear` runs SessionEnd (reason clear); later events carry a new session_id and transcript. `/exit` runs
  SessionEnd with reason prompt_input_exit. PreCompact has `trigger`; PostCompact has `compact_summary`.
- SubagentStop has agent_id, agent_type and agent_transcript_path; one ran after a plain turn, agent_type "".
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import anyio
import anyio.to_thread
import structlog
from libpymux import Server
from pydantic import JsonValue

from aid import hook as hook_command
from aid.backends.permissions import PermissionWaits
from aid.env import agent_environment
from aid.mcp import AID_TOOLS_RULE, claude_config, session_servers
from aid.paths import default_paths
from aid.protocol import (
    Activity,
    Lifecycle,
    Output,
    PaneAddress,
    PaneView,
    PermissionChoice,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    Started,
    TextDelta,
    ToolCall,
    clip,
)
from aid.spec import BUILTIN_MCP_SERVER
from aid.transcript import (
    TranscriptFollower,
    TurnEnded,
    TurnUsage,
    config_dir,
    find_transcript,
    human_prompt,
    items_from_entry,
    last_version,
    usage_since,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from libpymux import Pane

    from aid.backends.base import Emit, Record, Report
    from aid.protocol import HistoryItem, SessionEvent
    from aid.spec import ClaudeTtySpec

log = structlog.get_logger(__name__)

SESSION_ID_FILE: Final = "claude-session-id"
LAUNCHER_FILE: Final = "launch.sh"
MCP_CONFIG_FILE: Final = "mcp.json"
PANE_TERMINAL: Final = ("TERM", "COLORTERM", "TERMINFO", "TERMINFO_DIRS")
# With no client attached a pymux session is 80x24 unless it is started with a size. Claude Code lays out its
# screen at the width it is given and keeps no record of more, so the size must be right from the start. A client
# that attaches resizes the window while it watches.
PANE_COLUMNS: Final = 200
PANE_ROWS: Final = 50
PYMUX_SESSION: Final = "aid"
START_TIMEOUT: Final = 60.0
# From aid's paste to the transcript's first entry of the turn. A new session's transcript appears then too.
TURN_START_TIMEOUT: Final = 60.0
POLL: Final = 0.2
# 0.4 s between the paste and Enter worked every time; nothing shorter was tried.
PASTE_SETTLE: Final = 0.4
PANE_CHECK: Final = 2.0
HOOKS_FILE: Final = "settings.json"
HOOK_SCRIPT: Final = str(Path(hook_command.__file__))
HOOK_EVENTS: Final = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "Stop",
    "StopFailure",
    "Notification",
    "PermissionRequest",
    "PostCompact",
    "SubagentStop",
)
# Claude kills a hook at its timeout; the pane's own dialog still asks. aid gives up a little before, to record it.
WAITING_HOOK_TIMEOUT: Final = 3600
HOOK_MARGIN: Final = 5
REQUEST_TURN_WAIT: Final = 5.0
ALLOW_OPTION: Final = "allow"
HOOK_OPTIONS: Final = [
    PermissionChoice(option_id=ALLOW_OPTION, name="Yes", kind="allow_once"),
    PermissionChoice(option_id="deny", name="No", kind="reject_once"),
]
# Notifications that need no one: an idle prompt is `working` false, a permission prompt a PermissionRequest.
QUIET_NOTIFICATIONS: Final = frozenset({"idle_prompt", "permission_prompt"})

_PROMPT_LINE = re.compile(r"^❯(\s|$)", re.MULTILINE)
_TRUST_DIALOG = "trust this folder"
_CHANNELS_DIALOG = "Loading development channels"


class Screen(Enum):
    STARTING = "starting"
    TRUST = "trust"
    CHANNELS = "channels"
    READY = "ready"


def classify(capture: str) -> Screen:
    if _TRUST_DIALOG in capture:
        return Screen.TRUST
    if _CHANNELS_DIALOG in capture:
        return Screen.CHANNELS
    if _PROMPT_LINE.search(capture):
        return Screen.READY
    return Screen.STARTING


def permission_title(tool_input: JsonValue) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("description", "command", "file_path", "url"):
        if isinstance(value := tool_input.get(key), str):
            return value
    return None


def hook_decision(option_id: str | None) -> dict[str, Any]:
    """PermissionRequest's answer: allow for the allow option, deny for anything else (a person's cancel too)."""
    if option_id == ALLOW_OPTION:
        decision: dict[str, Any] = {"behavior": "allow"}
    else:
        decision = {"behavior": "deny", "message": "A person refused this in aid."}
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}


def launcher_script(env: dict[str, str], cwd: str, argv: list[str]) -> str:
    """A script that starts Claude with exactly `env`, plus the pane's terminal variables.

    The pymux server's own environment belongs to whoever started it, so `env -i` drops it. The variables live
    in this file, mode 0700, and not in the window's command, which pymux keeps and shows.

    The terminal variables are the pane's, not aid's: the pane is the terminal. Measured: pymux gives a pane
    TERM=pyte, COLORTERM=truecolor and a TERMINFO_DIRS holding pyte's terminfo; without them Claude draws no colour.
    """
    assignments = " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(env.items()) if k not in PANE_TERMINAL)
    # `${VAR+...}` unquoted: a variable the pane does not set adds no word at all.
    terminal = " ".join(f'${{{name}+"{name}=${name}"}}' for name in PANE_TERMINAL)
    return f"#!/bin/sh\ncd {shlex.quote(cwd)} || exit 1\nexec env -i {terminal} {assignments} {shlex.join(argv)}\n"


def claude_argv(
    spec: ClaudeTtySpec, args: list[str], session_id: str, *, resume: bool, mcp_config: str | None, settings: str
) -> list[str]:
    """`args`: spec.args less any --settings, which `settings` holds merged into aid's."""
    # --mcp-config and --allowedTools are variadic: each goes before another option, or it would take the first
    # of the args too. A repeated --allowedTools in the args adds to this one.
    mcp = ["--mcp-config", mcp_config] if mcp_config else []
    aid = (
        ["--allowedTools", AID_TOOLS_RULE, "--dangerously-load-development-channels", f"server:{BUILTIN_MCP_SERVER}"]
        if spec.aid_tools
        else []
    )
    session = ["--resume" if resume else "--session-id", session_id]
    return [*spec.command, *mcp, *aid, "--settings", settings, *session, *args]


def split_settings(args: list[str]) -> tuple[str | None, list[str]]:
    """The last --settings value in Claude's args, and the args without any. Measured on 2.1.284: Claude takes
    only the last --settings, so aid's hooks and a person's own settings must be one document."""
    value: str | None = None
    rest: list[str] = []
    it = iter(args)
    for arg in it:
        if arg == "--settings":
            value = next(it, None)
        elif arg.startswith("--settings="):
            value = arg.removeprefix("--settings=")
        else:
            rest.append(arg)
    return value, rest


def read_settings(value: str, cwd: str) -> dict[str, Any]:
    """A --settings value as Claude reads it: JSON text, or a path to a JSON file (relative to the cwd)."""
    text = value if value.lstrip().startswith("{") else (Path(cwd) / value).read_text()
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError(f"--settings {value!r} is not a JSON object")
    return cast("dict[str, Any]", document)


def hook_settings(control: str, session: str, own: dict[str, Any]) -> dict[str, Any]:
    """`own` with aid's hook added to each event in HOOK_EVENTS, after any hook `own` has for it."""
    hooks: dict[str, list[Any]] = {event: list(entries) for event, entries in dict(own.get("hooks") or {}).items()}
    for event in HOOK_EVENTS:
        handler: dict[str, Any] = {
            "type": "command",
            "command": shlex.join([sys.executable, HOOK_SCRIPT, control, session, event]),
        }
        if event in hook_command.WAITING_EVENTS:
            handler["timeout"] = WAITING_HOOK_TIMEOUT
        hooks.setdefault(event, []).append({"hooks": [handler]})
    return {**own, "hooks": hooks}


@dataclass
class _Claim:
    """aid's prompt, waiting for the turn it starts."""

    emit: Emit
    opened: anyio.Event = field(default_factory=anyio.Event)
    ended: anyio.Event = field(default_factory=anyio.Event)
    output: Output | None = None


@dataclass
class _Turn:
    claim: _Claim | None
    """None for a turn nobody sent through aid: its entries are recorded under `id`."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    chunks: list[str] = field(default_factory=list[str])
    usage: TurnUsage = field(default_factory=TurnUsage)
    inputs: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Each tool call's input, by tool call id: a PermissionRequest names no call, only its input."""
    asking: dict[str, PermissionRequest] = field(default_factory=dict[str, PermissionRequest])
    """Requests no decision is recorded for yet, by request id. Whoever takes one out records it."""

    def call_with(self, tool_input: JsonValue) -> str | None:
        return next((call for call, seen in reversed(self.inputs.items()) if seen == tool_input), None)

    def take_asking(self, tool_call_id: str) -> list[PermissionRequest]:
        taken = [r for r in self.asking.values() if tool_call_id and r.tool_call_id == tool_call_id]
        for request in taken:
            del self.asking[request.request_id]
        return taken


class ClaudeTtyBackend:
    """Follows the transcript for the worker's life, so turns typed into the pane are recorded too.

    A turn that opens while aid's prompt waits is aid's; any other is recorded as its own. Claude Code marks a
    pasted prompt as it marks a typed one, so one typed in the moment between aid's paste and Claude taking it
    counts as aid's.
    """

    def __init__(
        self,
        server: Server,
        pane: Pane,
        transcripts: Path,
        id_file: anyio.Path,
        session_id: str,
        started: Started,
        offset: int,
    ) -> None:
        self._server = server
        self._pane = pane
        self._transcripts = transcripts
        self._id_file = id_file
        self._session_id = session_id
        self._following: anyio.CancelScope | None = None
        self._gone = anyio.Event()
        """Claude ended: the worker ends with it."""
        self._started = started
        # Where the transcript ended before Claude started: earlier turns are in history already, or predate aid.
        self._offset = offset
        self._record: Record | None = None
        self._report: Report | None = None
        self._claim: _Claim | None = None
        self._turn: _Turn | None = None
        self._working = False
        self._attention: str | None = None
        self._waits = PermissionWaits()
        self._subagent_read: dict[Path, int] = {}
        """How far each subagent transcript is counted: a subagent can stop, resume and stop again."""

    def started(self) -> Started:
        return self._started

    async def hook(self, event: str, payload: JsonValue) -> JsonValue:
        fields = cast("dict[str, JsonValue]", payload) if isinstance(payload, dict) else {}
        session_id = fields.get("session_id")
        if event != "SessionEnd" and isinstance(session_id, str) and session_id != self._session_id:
            await self._switch(session_id, fields.get("model"))
        match event:
            case "SessionEnd":
                reason = fields.get("reason")
                reason = reason if isinstance(reason, str) else None
                if reason == "clear":
                    await self._record_alone(Lifecycle(event="cleared"))
                else:
                    await self._record_alone(Lifecycle(event="ended", detail=reason))
                    self._gone.set()
            case "PostCompact":
                trigger, summary = fields.get("trigger"), fields.get("compact_summary")
                await self._record_alone(
                    Lifecycle(
                        event="compacted",
                        detail=trigger if isinstance(trigger, str) else None,
                        summary=clip(summary) if isinstance(summary, str) else None,
                    )
                )
            case "UserPromptSubmit":
                await self._activity(working=True, attention=None)
            case "Stop" | "StopFailure":
                await self._activity(working=False, attention=self._attention)
            case "Notification" if fields.get("notification_type") not in QUIET_NOTIFICATIONS:
                message = fields.get("message")
                await self._activity(working=self._working, attention=message if isinstance(message, str) else None)
            case "PermissionRequest":
                return await self._permission(fields)
            case "SubagentStop":
                path, kind = fields.get("agent_transcript_path"), fields.get("agent_type")
                if isinstance(path, str):
                    await self._subagent_usage(Path(path), kind if isinstance(kind, str) and kind else "subagent")
            case _:
                pass
        return None

    def answer_permission(
        self, request_id: str, option_id: str | None, plugin: str | None = None, text: str | None = None
    ) -> bool:
        return self._waits.answer(request_id, option_id, plugin, text)

    async def _permission(self, fields: dict[str, JsonValue]) -> JsonValue:
        """Ask aid's people while the pane asks too. Whoever answers first decides; the hook answers Claude only for
        a person in aid."""
        turn = await self._request_turn()
        tool_input = fields.get("tool_input")
        tool_name = fields.get("tool_name")
        request = PermissionRequest(
            request_id=uuid.uuid4().hex,
            tool_call_id=await self._call_with(turn, tool_input) or "",
            tool_name=tool_name if isinstance(tool_name, str) else None,
            title=permission_title(tool_input),
            input=tool_input,
            options=HOOK_OPTIONS,
        )
        turn.asking[request.request_id] = request
        await self._send(turn, request)
        decision = PermissionDecision(request_id=request.request_id, option_id=None, by=PermissionDecider.TIMEOUT)
        with anyio.move_on_after(WAITING_HOOK_TIMEOUT - HOOK_MARGIN):
            decision = await self._waits.wait(request)
        if turn.asking.pop(request.request_id, None) is not None:
            await self._send(turn, decision)
        answered = decision.by in (PermissionDecider.PERSON, PermissionDecider.PLUGIN)
        return hook_decision(decision.option_id) if answered else None

    async def _request_turn(self) -> _Turn:
        """The turn a PermissionRequest belongs to. The hook can come before the follower has read the turn's
        start; a turn it never finds is recorded as its own."""
        with anyio.move_on_after(REQUEST_TURN_WAIT):
            while self._turn is None:  # noqa: ASYNC110 -- the follower polls the transcript; nothing signals
                await anyio.sleep(POLL)
        return self._turn or _Turn(claim=None)

    async def _call_with(self, turn: _Turn, tool_input: JsonValue) -> str | None:
        with anyio.move_on_after(REQUEST_TURN_WAIT):
            while (call := turn.call_with(tool_input)) is None:  # noqa: ASYNC110 -- as above
                await anyio.sleep(POLL)
            return call
        return None

    async def _settle(self, turn: _Turn, requests: list[PermissionRequest], by: PermissionDecider) -> None:
        for request in requests:
            self._waits.settle(request.request_id, None, by)
            await self._send(turn, PermissionDecision(request_id=request.request_id, option_id=None, by=by))

    async def _activity(self, *, working: bool, attention: str | None) -> None:
        if (working, attention) == (self._working, self._attention):
            return
        self._working, self._attention = working, attention
        if self._report is not None:
            await self._report(Activity(working=working, attention=attention))

    async def follow(self, record: Record, report: Report) -> None:
        self._record = record
        self._report = report
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._follow_transcripts)
            await self._gone.wait()
            tg.cancel_scope.cancel()

    async def _follow_transcripts(self) -> None:
        """The session's transcript, then the next one's after a switch (`/clear`)."""
        while True:
            with anyio.CancelScope() as scope:
                self._following = scope
                while (path := await self._find_transcript()) is None:  # noqa: ASYNC110 -- Claude creates the file; nothing signals it
                    await anyio.sleep(POLL)
                async for entry in TranscriptFollower(path, self._offset).follow():
                    await self._take(entry)

    async def _switch(self, session_id: str, model: JsonValue) -> None:
        """Claude started another session in the same pane: follow its transcript from the start."""
        log.info("session_switch", session=session_id, previous=self._session_id)
        self._session_id, self._offset = session_id, 0
        await self._id_file.write_text(session_id)
        if self._turn is not None:
            turn, self._turn = self._turn, None
            await self._end(turn, "cancelled")
        self._started = self._started.model_copy(
            update={
                "agent_session": session_id,
                "resumed": False,
                "model": model if isinstance(model, str) else None,
            }
        )
        await self._record_alone(self._started)
        if self._following is not None:
            self._following.cancel()

    async def _subagent_usage(self, path: Path, agent: str) -> None:
        """Tokens a subagent spent since it last stopped, under the running turn if there is one."""
        start = self._subagent_read.get(path, 0)
        usage, self._subagent_read[path] = await anyio.to_thread.run_sync(usage_since, path, start)
        if not usage.requests:
            return
        usage = usage.model_copy(update={"agent": agent})
        if self._turn is not None:
            await self._send(self._turn, usage)
        else:
            await self._record_alone(usage)

    async def _record_alone(self, item: HistoryItem) -> None:
        """An entry of no turn: it gets a turn id of its own."""
        if self._record is not None:
            await self._record(uuid.uuid4().hex, item)

    async def _take(self, entry: dict[str, Any]) -> None:
        prompt = human_prompt(entry)
        items = items_from_entry(entry)
        if self._turn is None:
            if prompt is None and all(isinstance(item, TurnEnded) for item in items):
                return
            self._turn = _Turn(claim=self._claim)
            if self._claim is not None:
                self._claim.opened.set()
            elif prompt is not None:
                await self._send(self._turn, PromptEntry(text=prompt))
        elif prompt is not None:
            # Claude takes a prompt given mid-turn into the running turn. aid's paste makes the rest of it aid's.
            if self._turn.claim is None and self._claim is not None:
                self._turn.claim = self._claim
                self._claim.opened.set()
            elif self._turn.claim is None:
                await self._send(self._turn, PromptEntry(text=prompt))
        turn = self._turn
        turn.usage.add(entry)
        for item in items:
            if isinstance(item, TurnEnded):
                self._turn = None
                asking = list(turn.asking.values())
                turn.asking.clear()
                await self._settle(turn, asking, PermissionDecider.CANCEL)
                await self._end(turn, "cancelled" if item.interrupted else "end_turn")
                return
            if isinstance(item, TextDelta):
                turn.chunks.append(item.text)
            if isinstance(item, ToolCall):
                if item.input is not None:
                    turn.inputs[item.tool_call_id] = item.input
                if item.status in ("completed", "failed"):
                    # The call ran or failed with a request still open: the pane answered it.
                    await self._settle(turn, turn.take_asking(item.tool_call_id), PermissionDecider.TERMINAL)
            await self._send(turn, item)

    async def _send(self, turn: _Turn, item: SessionEvent | PromptEntry) -> None:
        if turn.claim is not None:
            if not isinstance(item, PromptEntry):  # aid recorded its own prompt
                await turn.claim.emit(item)
        elif self._record is not None:
            await self._record(turn.id, item)

    async def _end(self, turn: _Turn, reason: str) -> None:
        output = Output(output="".join(turn.chunks), stop_reason=reason)
        await self._send(turn, turn.usage.usage())
        if turn.claim is None:
            await self._send(turn, output)
        else:
            turn.claim.output = output
            turn.claim.ended.set()

    async def _pane_alive(self) -> bool:
        def alive() -> bool:
            return any(p.id == self._pane.id and not p.dead for p in self._server.panes)

        return await anyio.to_thread.run_sync(alive)

    async def _find_transcript(self) -> Path | None:
        return await anyio.to_thread.run_sync(find_transcript, self._transcripts, self._session_id)

    async def prompt(self, text: str, emit: Emit) -> Output:
        if self._record is None:
            raise RuntimeError("the transcript is not being followed")
        claim = _Claim(emit)
        self._claim = claim
        try:
            await anyio.to_thread.run_sync(self._pane.send_keys, f"\x1b[200~{text}\x1b[201~", False)
            await anyio.sleep(PASTE_SETTLE)
            await anyio.to_thread.run_sync(self._pane.send_key, "Enter")
            async with anyio.create_task_group() as tg:

                async def watch_pane() -> None:
                    while await self._pane_alive():  # noqa: ASYNC110 -- libpymux has no event stream
                        await anyio.sleep(PANE_CHECK)
                    raise RuntimeError("Claude exited during the turn")

                tg.start_soon(watch_pane)
                with anyio.fail_after(TURN_START_TIMEOUT):
                    await claim.opened.wait()
                await claim.ended.wait()
                tg.cancel_scope.cancel()
        finally:
            self._claim = None
            if self._turn is not None and self._turn.claim is claim:
                # aid stopped waiting: whatever the turn still writes is recorded as a turn of its own.
                self._turn.claim = None
        if claim.output is None:
            raise RuntimeError("the turn ended without an output")
        return claim.output

    async def cancel(self) -> None:
        await anyio.to_thread.run_sync(self._pane.send_key, "Escape")

    def pane_address(self) -> PaneAddress:
        return PaneAddress(socket=self._server.socket_path, pane=self._pane.id)

    async def screen(self, *, stylesheet: bool, since: int | None, wait: float) -> PaneView:
        if since is not None and since >= 0:
            # Abandoned on cancel: a viewer that leaves must not hold up the worker's stop for the whole wait.
            await anyio.to_thread.run_sync(
                lambda: self._pane.wait_for_change(since=since, timeout=wait), abandon_on_cancel=True
            )

        def capture() -> PaneView:
            # Asked of the server: Pane.revision is the snapshot from when the Pane was read, and a wait since that
            # answers at once, every time. Read before drawing, so the frame is at least this new and the next wait
            # misses nothing. The refresh is for in_mode and mode below, snapshot fields too.
            revision = self._pane.current_revision()
            self._pane.refresh()
            html = self._pane.capture_html()
            css = self._server.html_stylesheet(self._pane) if stylesheet else None
            # pymux draws a mode (copy mode, a popup) above the pane, and the pane's page does not hold it.
            overlay = (self._pane.mode or "a pymux mode") if self._pane.in_mode else None
            return PaneView(revision=revision, html=html, stylesheet=css, overlay=overlay)

        return await anyio.to_thread.run_sync(capture)


async def _ensure_server(spec: ClaudeTtySpec, socket: str) -> Server:
    server = Server(socket)
    if await anyio.to_thread.run_sync(server.is_alive):
        return server
    await anyio.Path(socket).parent.mkdir(parents=True, exist_ok=True)
    size = ["-x", str(PANE_COLUMNS), "-y", str(PANE_ROWS)]
    result = await anyio.run_process(
        [*spec.pymux_command, "-S", socket, "new-session", "-d", "-s", PYMUX_SESSION, *size], check=False
    )
    # Two workers can race to start it; the loser's error is fine once a server answers.
    if not await anyio.to_thread.run_sync(server.is_alive):
        raise RuntimeError(f"could not start pymux on {socket}: {result.stderr.decode(errors='replace').strip()}")
    return server


async def _wait_ready(pane: Pane, trust: bool) -> None:
    with anyio.fail_after(START_TIMEOUT):
        while True:
            screen = classify(await anyio.to_thread.run_sync(pane.capture))
            if screen is Screen.READY:
                return
            if screen is Screen.TRUST:
                if not trust:
                    raise RuntimeError("Claude Code asks whether to trust the cwd; set trust_cwd to answer yes")
                await anyio.to_thread.run_sync(pane.send_key, "Down")
                await anyio.to_thread.run_sync(pane.send_key, "Enter")
            if screen is Screen.CHANNELS:
                # Its first option, and the default, is "I am using this for local development".
                await anyio.to_thread.run_sync(pane.send_key, "Enter")
            await anyio.sleep(POLL)


async def _write_private(path: anyio.Path, text: str, mode: int) -> None:
    """Environments and MCP headers hold credentials; the file gets its mode before it gets them."""
    await path.touch(mode=mode)
    await path.chmod(mode)
    await path.write_text(text)


@asynccontextmanager
async def open_claude_tty(spec: ClaudeTtySpec, state_dir: anyio.Path) -> AsyncGenerator[ClaudeTtyBackend]:
    env = agent_environment(spec.env, inherit=spec.inherit_env)
    transcripts = config_dir(env)
    id_file = state_dir / SESSION_ID_FILE
    session_id = (await id_file.read_text()).strip() if await id_file.exists() else str(uuid.uuid4())
    await id_file.write_text(session_id)
    transcript = await anyio.to_thread.run_sync(find_transcript, transcripts, session_id)
    resume = transcript is not None
    offset = (await anyio.Path(transcript).stat()).st_size if transcript else 0
    # Known only from a transcript: a new session has none until its first turn.
    version = await anyio.to_thread.run_sync(last_version, transcript) if transcript else None
    mcp_config: str | None = None
    if servers := session_servers(spec, state_dir.name):
        mcp_file = state_dir / MCP_CONFIG_FILE
        await _write_private(mcp_file, json.dumps(claude_config(servers)), 0o600)
        mcp_config = str(mcp_file)
    own_settings, args = split_settings(spec.args)
    own = await anyio.to_thread.run_sync(read_settings, own_settings, spec.cwd) if own_settings else {}
    settings_file = state_dir / HOOKS_FILE
    settings = hook_settings(default_paths().control, state_dir.name, own)
    await _write_private(settings_file, json.dumps(settings), 0o600)
    argv = claude_argv(spec, args, session_id, resume=resume, mcp_config=mcp_config, settings=str(settings_file))

    launcher = state_dir / LAUNCHER_FILE
    await _write_private(launcher, launcher_script(env, spec.cwd, argv), 0o700)

    socket = spec.pymux_socket or str(default_paths().runtime_dir / "pymux.sock")
    server = await _ensure_server(spec, socket)

    def new_window() -> Pane:
        window = server.session.new_window(
            command=str(launcher), name=f"aid:{state_dir.name}", start_directory=spec.cwd, select=False
        )
        if window is None or not window.panes:
            raise RuntimeError("pymux made no window")
        return window.panes[0]

    pane = await anyio.to_thread.run_sync(new_window)
    log.info("pane_started", session=session_id, pane=pane.id, socket=socket, resumed=resume)
    try:
        await _wait_ready(pane, spec.trust_cwd)
        agent = f"Claude Code {version}" if version else None
        started = Started(pid=os.getpid(), agent_session=session_id, resumed=resume, agent=agent)
        yield ClaudeTtyBackend(server, pane, transcripts, id_file, session_id, started, offset)
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(lambda: server.cmd(["kill-window", "-t", pane.window_id], check=False))
