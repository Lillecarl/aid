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
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import anyio
import anyio.to_thread
from libpymux import Server

from aid.env import agent_environment
from aid.mcp import AID_TOOLS_RULE, claude_config, session_servers
from aid.paths import default_paths
from aid.protocol import Output, PaneAddress, PaneView, PromptEntry, Started, TextDelta
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
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from libpymux import Pane

    from aid.backends.base import Emit, Record
    from aid.protocol import SessionEvent
    from aid.spec import ClaudeTtySpec

log = logging.getLogger(__name__)

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


def claude_argv(spec: ClaudeTtySpec, session_id: str, *, resume: bool, mcp_config: str | None) -> list[str]:
    # --mcp-config and --allowedTools are variadic: each goes before another option, or it would take the first
    # of spec.args too. A repeated --allowedTools in spec.args adds to this one.
    mcp = ["--mcp-config", mcp_config] if mcp_config else []
    aid = (
        ["--allowedTools", AID_TOOLS_RULE, "--dangerously-load-development-channels", f"server:{BUILTIN_MCP_SERVER}"]
        if spec.aid_tools
        else []
    )
    return [*spec.command, *mcp, *aid, "--resume" if resume else "--session-id", session_id, *spec.args]


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


class ClaudeTtyBackend:
    """Follows the transcript for the worker's life, so turns typed into the pane are recorded too.

    A turn that opens while aid's prompt waits is aid's; any other is recorded as its own. Claude Code marks a
    pasted prompt as it marks a typed one, so one typed in the moment between aid's paste and Claude taking it
    counts as aid's.
    """

    def __init__(
        self, server: Server, pane: Pane, transcripts: Path, session_id: str, started: Started, offset: int
    ) -> None:
        self._server = server
        self._pane = pane
        self._transcripts = transcripts
        self._session_id = session_id
        self._started = started
        # Where the transcript ended before Claude started: earlier turns are in history already, or predate aid.
        self._offset = offset
        self._record: Record | None = None
        self._claim: _Claim | None = None
        self._turn: _Turn | None = None

    def started(self) -> Started:
        return self._started

    async def follow(self, record: Record) -> None:
        self._record = record
        while (path := await self._find_transcript()) is None:  # noqa: ASYNC110 -- Claude creates the file; nothing signals it
            await anyio.sleep(POLL)
        async for entry in TranscriptFollower(path, self._offset).follow():
            await self._take(entry)

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
                await self._end(turn, "cancelled" if item.interrupted else "end_turn")
                return
            if isinstance(item, TextDelta):
                turn.chunks.append(item.text)
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
    argv = claude_argv(spec, session_id, resume=resume, mcp_config=mcp_config)

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
    log.info("claude %s in pymux pane %s on %s (%s)", session_id, pane.id, socket, "resumed" if resume else "new")
    try:
        await _wait_ready(pane, spec.trust_cwd)
        agent = f"Claude Code {version}" if version else None
        started = Started(pid=os.getpid(), agent_session=session_id, resumed=resume, agent=agent)
        yield ClaudeTtyBackend(server, pane, transcripts, session_id, started, offset)
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(lambda: server.cmd(["kill-window", "-t", pane.window_id], check=False))
