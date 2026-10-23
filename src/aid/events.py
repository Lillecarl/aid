"""What the daemon publishes on its PUB socket (`Paths.events`): multipart `[topic, JSON]`.

- `sessions/`: the session list (`SessionInfo`s), when it changes.
- `session/<name>/status/`: the session's `SessionStatus`, when it changes.
- `session/<name>/history/`: each `HistoryEntry` as the session's history takes it, with its seq.
- `session/<name>/turn/`: each turn's `TurnEvent` boundaries, started once the worker holds the prompt and
  ended once the daemon closes it. Advisory: every started has an ended, and a missed boundary repairs
  through history (the turn's entries) and status (its working flag).
- `session/<name>/tool/`: each `ToolCall` event as the worker sent it, with its turn (`ToolEvent`), for
  subscribers tracking tool lifecycles by tool_call_id. Incremental like the worker's own events, and
  advisory: drops repair through history paging.
- `session/<name>/permission/`: each permission request or decision as the worker sent it, with its turn
  (`PermissionEvent`). Observability only: answering stays on the permission paths, so a dropped event
  costs insight, never a hung agent.
- `session/<name>/live/`: output an agent piped through `usertee` (`LiveChunk`), for a person to watch. Never
  recorded; `GetLive` fetches the newest of it for a late subscriber, and a dropped chunk is gone.
- `session/<name>/supervision/`: what a supervised session's link did (`SupervisionEvent`): each firing
  with its reason and loop score, each nudge the supervisor sent, each goal it changed. A dropped event
  repairs through history paging (firings and nudges are messages there), status (attention), and the
  goals tool.

Topics end in `/`, so a subscription to one session's prefix never matches another session whose name starts the
same. PUB drops what a slow subscriber cannot take: a history gap shows as a jump in seq and is fetched with
`GetHistory`, and a status or list is repaired by the next change. Control never rides these topics — firing,
waking and answering stay on the reliable request and inbox paths — so a dropped event costs insight, never
action.
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


def turn_topic(name: str) -> bytes:
    return session_prefix(name) + b"turn/"


def tool_topic(name: str) -> bytes:
    return session_prefix(name) + b"tool/"


def permission_topic(name: str) -> bytes:
    return session_prefix(name) + b"permission/"


def live_topic(name: str) -> bytes:
    return session_prefix(name) + b"live/"


def supervision_topic(name: str) -> bytes:
    return session_prefix(name) + b"supervision/"
