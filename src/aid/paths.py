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
    def events(self) -> str:
        """The daemon's PUB socket: `events.topic(...)` names what it publishes."""
        return f"ipc://{self.runtime_dir / 'events.sock'}"

    @property
    def workers(self) -> str:
        return f"ipc://{self.runtime_dir / 'workers.sock'}"

    @property
    def plugins(self) -> str:
        """Plugins' requests, CURVE (`aid.plugins`)."""
        return f"ipc://{self.runtime_dir / 'plugins.sock'}"

    @property
    def plugin_events(self) -> str:
        """What the daemon publishes, CURVE, for plugins granted `read`."""
        return f"ipc://{self.runtime_dir / 'plugin-events.sock'}"

    @property
    def server_key(self) -> Path:
        """The daemon's CURVE public key while it runs, which a plugin's sockets need."""
        return self.runtime_dir / "server.key"

    def plugin_dir(self, name: str) -> Path:
        return self.state_dir / "plugins" / name

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
