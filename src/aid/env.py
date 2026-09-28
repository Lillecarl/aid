from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

# Variables a running Claude Code session exports about itself. An agent started from inside one inherits them
# and takes the parent's identity: with CLAUDE_CODE_CHILD_SESSION an interactive Claude turns transcript saving
# off, and WAPTY_ID points wrapty's hooks at the parent's control socket. Measured on Claude Code 2.1.283.
SESSION_MARKERS: Final = frozenset(
    {
        "AI_AGENT",
        "CLAUDECODE",
        "CLAUDE_CODE_BRIDGE_SESSION_ID",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "CLAUDE_CODE_SESSION_ATTENDED",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_EFFORT",
        "CLAUDE_PID",
        "WAPTY_ID",
    }
)


def agent_environment(extra: Mapping[str, str], *, inherit: bool = True) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k not in SESSION_MARKERS} if inherit else {}
    return {**base, **extra}
