"""Settled turn ends, platform-agnostic: feed a session's turn and supervision events in, poll settled
sessions out. Turns that end but come back — a supervisor nudge starting a wake turn — never settle.

Both chat plugins share this: Telegram's notices and Discord's channel posts settle through the same
filter, with the same tuning knobs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog

from aid.protocol import SessionStatus, SupervisionEvent, TurnEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aid.client import Client

log = structlog.get_logger(__name__)

IDLE_WINDOW: Final = 10.0
"""An unsupervised turn ended: nothing waking it within this long means it stays ended."""
VERDICT_TIMEOUT: Final = 300.0
"""A firing this old with no verdict from its overseer settles on the session's idleness instead."""
TICK_INTERVAL: Final = 2.0
"""How often a quiet watch re-checks for settled turns: notices lag a tick behind settling, at most."""
RECONCILE_INTERVAL: Final = 60.0
"""How often the watch loop re-reads the session list, repairing turns a dropped event hid."""


def describe_end(reason: str | None) -> str:
    """How a settled turn ended, in words rather than codes."""
    if reason is None:
        return "turn ended (unknown)"
    if reason == "done":
        return "turn ended"
    if reason == "cancelled":
        return "turn cancelled"
    return f"turn failed ({reason})"


@dataclass
class _Pending:
    turn: str
    deadline: float
    supervisor: str | None = None
    """The overseer whose verdict is awaited; None once it stayed silent or no firing came."""
    verdict_open: bool = False
    """The overseer's verdict turn runs now: its end without a nudge settles the episode."""


class Tracker:
    """Which sessions sit in an ended turn that nothing revived.

    Fed by a session's turn and supervision events; `tick` names the sessions whose episode settled.
    Time is the caller's monotonic clock, so tests drive it exactly.
    """

    def __init__(self, *, idle_window: float = IDLE_WINDOW, verdict_timeout: float = VERDICT_TIMEOUT) -> None:
        self._idle_window = idle_window
        self._verdict_timeout = verdict_timeout
        self._turn: dict[str, bool] = {}
        self._pending: dict[str, _Pending] = {}

    def set_busy(self, session: str, busy: bool) -> None:
        self._turn[session] = busy

    def turn_started(self, session: str) -> None:
        self._turn[session] = True
        self._pending.pop(session, None)
        for pending in self._pending.values():
            if pending.supervisor == session and not pending.verdict_open:
                pending.verdict_open = True

    def turn_ended(self, session: str, turn: str, now: float) -> None:
        for pending in self._pending.values():
            if pending.supervisor == session and pending.verdict_open:
                pending.supervisor = None
                pending.verdict_open = False
                pending.deadline = now + self._idle_window
        self._turn[session] = False
        self._pending[session] = _Pending(turn=turn, deadline=now + self._idle_window)

    def firing(self, session: str, supervisor: str, turn: str, now: float) -> None:
        pending = self._pending.get(session)
        if pending is None or pending.turn != turn:
            return
        pending.supervisor = supervisor
        pending.verdict_open = False
        pending.deadline = now + self._verdict_timeout

    def nudge(self, session: str) -> None:
        self._pending.pop(session, None)

    def tick(self, now: float) -> list[str]:
        """Sessions whose episode settled and that sit idle: each is named once."""
        return [
            session
            for session, pending in list(self._pending.items())
            if now >= pending.deadline
            and not self._turn.get(session, False)
            and self._pending.pop(session, None) is not None
        ]


def route_topic(topic: bytes) -> tuple[str, str] | None:
    """A session topic as its session and kind (`turn`, `supervision` or `status`); None for anything else."""
    parts = topic.split(b"/")
    if len(parts) != 4 or parts[0] != b"session" or parts[2] not in (b"turn", b"supervision", b"status"):
        return None
    return parts[1].decode(), parts[2].decode()


async def settled_turns(
    client: Client,
    on_settled: Callable[[str, str | None], Awaitable[None]],
    *,
    idle_window: float = IDLE_WINDOW,
    verdict_timeout: float = VERDICT_TIMEOUT,
    tick: float = TICK_INTERVAL,
    reconcile: float = RECONCILE_INTERVAL,
) -> None:
    """Call `on_settled(session, reason)` for turns that ended and stayed ended; a nudge reviving the loop
    cancels the notice. Subscribes to the session topics, repairs dropped events by re-reading the list."""
    tracker = Tracker(idle_window=idle_window, verdict_timeout=verdict_timeout)
    reasons: dict[str, str | None] = {}
    async with client.subscribed(b"session/", idle=tick) as published:
        for info in await client.sessions():
            tracker.set_busy(info.name, info.working)
        quiet = 0.0
        async for message in published:
            now = time.monotonic()
            if message is None:
                quiet += tick
                if quiet >= reconcile:
                    quiet = 0.0
                    for info in await client.sessions():
                        tracker.set_busy(info.name, info.working)
            else:
                quiet = 0.0
                try:
                    topic, payload = message
                    routed = route_topic(topic)
                    if routed is not None:
                        name, kind = routed
                        if kind == "turn":
                            event = TurnEvent.model_validate_json(payload)
                            if event.phase == "started":
                                tracker.turn_started(name)
                                reasons.pop(name, None)
                            else:
                                tracker.turn_ended(name, event.turn, now)
                                reasons[name] = event.reason
                        elif kind == "supervision":
                            event = SupervisionEvent.model_validate_json(payload)
                            if event.firing is not None:
                                tracker.firing(event.session, event.supervisor, event.firing.turn, now)
                            if event.nudge is not None:
                                tracker.nudge(event.session)
                        elif kind == "status":
                            tracker.set_busy(name, SessionStatus.model_validate_json(payload).busy)
                except Exception:
                    log.exception("settle_event_dropped")
                    continue
            for session in tracker.tick(now):
                await on_settled(session, reasons.pop(session, None))
