"""A stand-in for interactive Claude Code: reads a terminal like it, writes a transcript like it.

Prompts: "slow" waits for Escape; "count" writes one text block per number; "tool" makes a tool call;
anything else is echoed. FAKE_CLAUDE_TRUST=1 shows the trust dialog until a `.fake-trusted` file exists.
"""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import tty
import uuid
from pathlib import Path
from typing import Any

PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
DOWN = b"\x1b[B"
COUNT = 20


def transcript_path(session_id: str) -> Path:
    base = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    project = base / "projects" / str(Path.cwd()).replace("/", "-")
    project.mkdir(parents=True, exist_ok=True)
    return project / f"{session_id}.jsonl"


class Fake:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.buffer = b""

    def write(self, entry: dict[str, Any]) -> None:
        entry = {"uuid": str(uuid.uuid4()), "sessionId": self.session_id, "isSidechain": False, **entry}
        with transcript_path(self.session_id).open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def say(self, text: str) -> None:
        self.write({"type": "assistant", "message": {"id": "msg", "content": [{"type": "text", "text": text}]}})

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
                self.buffer = self.buffer[1:]
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

    def turn(self, text: str) -> None:
        self.write({"type": "user", "message": {"role": "user", "content": text}})
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
        elif text == "tool":
            use = {"type": "tool_use", "id": "toolu_fake", "name": "Bash", "input": {"command": "true"}}
            self.write({"type": "assistant", "message": {"id": "msg", "content": [use]}})
            result = {"type": "tool_result", "tool_use_id": "toolu_fake", "content": "ok"}
            self.write({"type": "user", "toolUseResult": {"stdout": "ok"}, "message": {"content": [result]}})
            self.say("ran it")
        else:
            self.say(f"echo: {text}")
        self.write({"type": "system", "subtype": "stop_hook_summary"})
        self.write({"type": "system", "subtype": "turn_duration", "durationMs": 1})

    def run(self) -> None:
        self.trust()
        pending = ""
        while True:
            self.screen(f" fake claude {self.session_id}\n{'─' * 20}\n❯ {pending}\n{'─' * 20}\n")
            kind, text = self.next_event()
            if kind == "paste":
                pending += text
            elif kind == "enter" and pending:
                self.turn(pending)
                pending = ""


def main() -> None:
    args = sys.argv[1:]
    session_id = next(args[i + 1] for i, a in enumerate(args) if a in ("--session-id", "--resume"))
    old = termios.tcgetattr(0)
    tty.setraw(0)
    try:
        Fake(session_id).run()
    finally:
        termios.tcsetattr(0, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
