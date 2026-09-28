from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_RUNTIME_DIR = "AID_RUNTIME_DIR"
ENV_STATE_DIR = "AID_STATE_DIR"
# Set in each worker and in the MCP server a session's agent starts: the session the code runs for.
ENV_SESSION = "AID_SESSION"
# "1" makes `aid.mcp_server` the session's channel: interactive Claude's way to hear its messages.
ENV_CHANNEL = "AID_CHANNEL"


@dataclass(frozen=True)
class Paths:
    runtime_dir: Path
    state_dir: Path

    @property
    def control(self) -> str:
        return f"ipc://{self.runtime_dir / 'control.sock'}"

    @property
    def workers(self) -> str:
        return f"ipc://{self.runtime_dir / 'workers.sock'}"

    def session_dir(self, name: str) -> Path:
        return self.state_dir / "sessions" / name


def default_paths() -> Paths:
    uid = os.getuid()
    runtime = os.environ.get(ENV_RUNTIME_DIR) or str(
        Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")) / "aid"
    )
    state = os.environ.get(ENV_STATE_DIR) or str(
        Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "aid"
    )
    return Paths(runtime_dir=Path(runtime), state_dir=Path(state))
