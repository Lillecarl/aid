"""What the daemon publishes on its PUB socket (`Paths.events`): multipart `[topic, JSON]`.

- `sessions/`: the session list (`SessionInfo`s), when it changes.
- `session/<name>/status/`: the session's `SessionStatus`, when it changes.
- `session/<name>/history/`: each `HistoryEntry` as the session's history takes it, with its seq.

Topics end in `/`, so a subscription to one session's prefix never matches another session whose name starts the
same. PUB drops what a slow subscriber cannot take: a history gap shows as a jump in seq and is fetched with
`GetHistory`, and a status or list is repaired by the next change.
"""

from __future__ import annotations

from typing import Final

SESSIONS: Final = b"sessions/"


def session_prefix(name: str) -> bytes:
    return f"session/{name}/".encode()


def status_topic(name: str) -> bytes:
    return session_prefix(name) + b"status/"


def history_topic(name: str) -> bytes:
    return session_prefix(name) + b"history/"
