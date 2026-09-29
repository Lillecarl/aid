"""A stand-in for interactive Claude Code: reads a terminal like it, writes a transcript like it.

Prompts: "slow" waits for Escape; "count" writes one text block per number; "tool" makes a tool call;
"mcp" says the --mcp-config file; "term" says TERM; "notify" runs the Notification hook; "permission" asks through the PermissionRequest hook and obeys it;
"permission-pane" asks, then answers yes in the pane while the hook still waits; anything else is echoed.
Each turn runs the UserPromptSubmit and Stop hooks from --settings. Typed /clear, /compact and /exit run theirs. FAKE_CLAUDE_TRUST=1 shows the trust dialog until a `.fake-trusted` file exists.
FAKE_CLAUDE_CHANNELS=1 shows the development channels warning, when the flag is given, until Enter.
"""

from __future__ import annotations

import json
import os
import select
import subprocess  # noqa: TID251 -- a synchronous stand-in; it runs hooks as Claude does
import sys
import termios
import time
import tty
import uuid
from pathlib import Path
from typing import Any

PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
DOWN = b"\x1b[B"
COUNT = 20
MODEL = "claude-fake-1"
NOTICE = "Claude has a question for you"
COMPACT_SUMMARY = "The person asked for a greeting."
PERMISSION_INPUT = {"command": "rm -rf build", "description": "Remove the build directory"}
USAGE = {
    "input_tokens": 3,
    "output_tokens": 5,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 7,
    "output_tokens_details": {"thinking_tokens": 2},
}


def transcript_path(session_id: str) -> Path:
    base = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    project = base / "projects" / str(Path.cwd()).replace("/", "-")
    project.mkdir(parents=True, exist_ok=True)
    return project / f"{session_id}.jsonl"


class Fake:
    def __init__(self, session_id: str, mcp_config: str | None, settings: str | None) -> None:
        self.session_id = session_id
        self.mcp_config = mcp_config
        self.settings: dict[str, Any] = json.loads(Path(settings).read_text()) if settings else {}
        self.buffer = b""

    def hook_processes(self, event: str, **fields: Any) -> list[subprocess.Popen[str]]:
        """Start the settings' hooks for `event` as Claude does: the payload on stdin, an answer on stdout."""
        payload = {
            "session_id": self.session_id,
            "transcript_path": str(transcript_path(self.session_id)),
            "cwd": str(Path.cwd()),
            "hook_event_name": event,
            **fields,
        }
        processes: list[subprocess.Popen[str]] = []
        for entry in self.settings.get("hooks", {}).get(event, []):
            for handler in entry["hooks"]:
                process = subprocess.Popen(
                    handler["command"],
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                assert process.stdin is not None
                process.stdin.write(json.dumps(payload))
                process.stdin.close()
                processes.append(process)
        return processes

    def hook(self, event: str, **fields: Any) -> dict[str, Any] | None:
        answer: dict[str, Any] | None = None
        for process in self.hook_processes(event, **fields):
            out, _ = process.communicate(timeout=600)
            if out.strip():
                answer = json.loads(out)
        return answer

    def permission(self, pane_answers: bool) -> None:
        """A Bash call that needs permission. The pane answers yes itself when `pane_answers`, while the hook waits."""
        use = {"type": "tool_use", "id": "toolu_perm", "name": "Bash", "input": PERMISSION_INPUT}
        self.assistant("msg_perm", use)
        result = {"type": "tool_result", "tool_use_id": "toolu_perm", "content": "ok"}
        fields: dict[str, Any] = {"tool_name": "Bash", "tool_input": PERMISSION_INPUT, "permission_suggestions": []}
        if pane_answers:
            [process] = self.hook_processes("PermissionRequest", **fields)
            time.sleep(1.5)  # The hook reaches aid, and aid opens its request.
            self.write({"type": "user", "toolUseResult": {"stdout": "ok"}, "message": {"content": [result]}})
            out, _ = process.communicate(timeout=30)
            self.say(f"the hook said {out.strip()!r}")
            return
        answer = self.hook("PermissionRequest", **fields)
        decision = (answer or {}).get("hookSpecificOutput", {}).get("decision", {})
        if decision.get("behavior") == "allow":
            self.write({"type": "user", "toolUseResult": {"stdout": "ok"}, "message": {"content": [result]}})
            self.say("allowed")
        else:
            self.say(f"denied: {decision.get('message')}")

    def write(self, entry: dict[str, Any]) -> None:
        entry = {"uuid": str(uuid.uuid4()), "sessionId": self.session_id, "isSidechain": False, **entry}
        with transcript_path(self.session_id).open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def assistant(self, message_id: str, block: dict[str, Any]) -> None:
        """Real Claude repeats a message's id, model and whole usage on every entry of it."""
        message = {"id": message_id, "model": MODEL, "usage": USAGE, "content": [block]}
        self.write({"type": "assistant", "message": message})

    def say(self, text: str) -> None:
        self.assistant("msg", {"type": "text", "text": text})

    def read_byte_chunk(self, timeout: float | None = None) -> bytes:
        ready, _, _ = select.select([0], [], [], timeout)
        return os.read(0, 4096) if ready else b""

    def next_event(self) -> tuple[str, str]:
        """("paste", text) | ("enter", "") | ("escape", "") | ("down", "")."""
        while True:
            if self.buffer.startswith(PASTE_START):
                end = self.buffer.find(PASTE_END)
                if end != -1:
                    text = self.buffer[len(PASTE_START) : end].decode()
                    self.buffer = self.buffer[end + len(PASTE_END) :]
                    return "paste", text
            elif self.buffer.startswith(DOWN):
                self.buffer = self.buffer[len(DOWN) :]
                return "down", ""
            elif self.buffer.startswith(b"\r"):
                self.buffer = self.buffer[1:]
                return "enter", ""
            elif self.buffer == b"\x1b":
                more = self.read_byte_chunk(0.05)
                if not more:
                    self.buffer = b""
                    return "escape", ""
                self.buffer += more
                continue
            elif self.buffer and not self.buffer.startswith(b"\x1b"):
                # Typed, not pasted: printable ASCII joins the prompt as the real one does; the rest is dropped.
                char, self.buffer = self.buffer[:1], self.buffer[1:]
                if b" " <= char < b"\x7f":
                    return "paste", char.decode()
                continue
            self.buffer += self.read_byte_chunk()

    def screen(self, text: str) -> None:
        sys.stdout.write("\x1b[2J\x1b[H" + text.replace("\n", "\r\n"))
        sys.stdout.flush()

    def trust(self) -> None:
        marker = Path.cwd() / ".fake-trusted"
        if os.environ.get("FAKE_CLAUDE_TRUST") != "1" or marker.exists():
            return
        self.screen(" Do you trust this folder?\n ❯ No, exit\n   Yes, I trust this folder\n")
        chose_yes = False
        while True:
            kind, _ = self.next_event()
            if kind == "down":
                chose_yes = True
            elif kind == "enter":
                if not chose_yes:
                    sys.exit(1)
                marker.touch()
                return

    def channels(self) -> None:
        if "--dangerously-load-development-channels" not in sys.argv or os.environ.get("FAKE_CLAUDE_CHANNELS") != "1":
            return
        self.screen(
            "  WARNING: Loading development channels\n\n  ❯ 1. I am using this for local development\n    2. Exit\n"
        )
        while self.next_event()[0] != "enter":
            pass

    def turn(self, text: str) -> None:
        self.hook("UserPromptSubmit", prompt=text)
        self.write({"type": "user", "origin": {"kind": "human"}, "message": {"role": "user", "content": text}})
        self.reply(text)
        self.hook("Stop", stop_hook_active=False)

    def reply(self, text: str) -> None:
        if text == "slow":
            self.say("waiting")
            while self.next_event()[0] != "escape":
                pass
            self.write(
                {"type": "user", "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]}}
            )
            return
        if text == "count":
            for i in range(COUNT):
                self.say(f"{i} ")
        elif text == "term":
            self.say(os.environ.get("TERM", "unset"))
        elif text == "mcp":
            self.say(Path(self.mcp_config).read_text() if self.mcp_config else "none")
        elif text in ("permission", "permission-pane"):
            self.permission(pane_answers=text == "permission-pane")
        elif text == "notify":
            self.hook("Notification", message=NOTICE, notification_type="elicitation_dialog")
            self.say("notified")
        elif text == "tool":
            use = {"type": "tool_use", "id": "toolu_fake", "name": "Bash", "input": {"command": "true"}}
            self.assistant("msg_tool", use)
            result = {"type": "tool_result", "tool_use_id": "toolu_fake", "content": "ok"}
            self.write({"type": "user", "toolUseResult": {"stdout": "ok"}, "message": {"content": [result]}})
            self.say("ran it")
        else:
            self.say(f"echo: {text}")
        self.write({"type": "system", "subtype": "stop_hook_summary"})
        self.write({"type": "system", "subtype": "turn_duration", "durationMs": 1})

    def run(self) -> None:
        self.trust()
        self.channels()
        pending = ""
        while True:
            self.screen(f" fake claude {self.session_id}\n{'─' * 20}\n❯ {pending}\n{'─' * 20}\n")
            kind, text = self.next_event()
            if kind == "paste":
                pending += text
            elif kind == "enter" and pending:
                self.command(pending) if pending.startswith("/") else self.turn(pending)
                pending = ""

    def command(self, text: str) -> None:
        """/clear starts a new session; /compact compacts this one; /exit ends Claude. Each runs its hooks."""
        if text == "/clear":
            self.hook("SessionEnd", reason="clear")
            self.session_id = str(uuid.uuid4())
            self.hook("SessionStart", source="clear", model=MODEL)
        elif text == "/compact":
            self.hook("PreCompact", trigger="manual", custom_instructions=None)
            self.hook("PostCompact", trigger="manual", compact_summary=COMPACT_SUMMARY)
        elif text == "/exit":
            self.hook("SessionEnd", reason="prompt_input_exit")
            sys.exit(0)


def main() -> None:
    args = sys.argv[1:]
    session_id = next(args[i + 1] for i, a in enumerate(args) if a in ("--session-id", "--resume"))
    mcp_config = next((args[i + 1] for i, a in enumerate(args) if a == "--mcp-config"), None)
    settings = next((args[i + 1] for i, a in enumerate(args) if a == "--settings"), None)
    old = termios.tcgetattr(0)
    tty.setraw(0)
    try:
        Fake(session_id, mcp_config, settings).run()
    finally:
        termios.tcsetattr(0, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
