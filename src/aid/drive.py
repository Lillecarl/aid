"""Turn by turn, for plugins that drive sessions through work (campaigns, PR reviews): wait for an idle session,
run a turn that ends on its own, and tell a model outage from a failed turn.

An outage costs no attempt. Measured on the dynamist campaign (2026-11): a vLLM gateway answered 503 for hours, and
3 attempts 10 s apart failed 18% of all units for good on it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import anyio
import structlog

from aid.history import MAX_PAGE
from aid.protocol import AidError, HistoryEntry

if TYPE_CHECKING:
    from pydantic import JsonValue

    from aid.client import RunResult, Session

log = structlog.get_logger(__name__)

BUSY: Final = "busy"
"""The daemon's refusal of a prompt to a session in a turn."""
END_TURN: Final = "end_turn"
OUTAGE: Final = frozenset({"model_unavailable"})
"""Failure codes of a model out of reach: the native backend's, after its own resends."""


@dataclass(frozen=True)
class Pacing:
    attempts: int
    """Turns a piece of work gets; an outage spends none."""
    backoff: float
    """Seconds after a failed turn."""
    outage_backoff: float = 30.0
    """Seconds after the first outage; each one in a row waits twice as long, up to `outage_max`."""
    outage_max: float = 600.0

    def outage_delay(self, outages: int) -> float:
        return min(self.outage_backoff * 2 ** max(outages - 1, 0), self.outage_max)


def is_outage(failure: AidError) -> bool:
    return failure.code in OUTAGE


async def idle(session: Session) -> None:
    """Until the session runs no turn: a prompt to a busy session is refused."""
    async with contextlib.aclosing(session.follow_status()) as statuses:
        async for status in statuses:
            if not status.busy:
                return


async def newest_seq(session: Session) -> int:
    page = await session.history(limit=1)
    return page.entries[-1].seq if page.entries else -1


async def since(session: Session, seq: int) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    while True:
        page = await session.history(after=seq, limit=MAX_PAGE)
        entries.extend(page.entries)
        if not page.entries or not page.has_newer:
            return entries
        seq = page.entries[-1].seq


async def ended(session: Session, prompt: str) -> RunResult[JsonValue]:
    """A turn that ended on its own: one cut off (`max_tokens`, measured on vLLM's GLM spending a 4000-token budget
    on reasoning) has not answered, and counts as failed."""
    result = await session.run(prompt)
    if result.stop_reason != END_TURN:
        raise AidError("stopped", f"the turn stopped at {result.stop_reason}")
    return result


async def turn(session: Session, prompt: str, pacing: Pacing) -> str:
    """The answer of a turn that ended on its own, from up to `attempts` failed turns; the last failure raises."""
    attempts = outages = 0
    while True:
        await idle(session)
        try:
            result = await ended(session, prompt)
            return result.output if isinstance(result.output, str) else str(result.output)
        except AidError as failure:
            if failure.code == BUSY:
                continue
            if is_outage(failure):
                outages += 1
                delay = pacing.outage_delay(outages)
                log.warning("model_outage", session=session.name, error=failure.message, wait=delay)
                await anyio.sleep(delay)
                continue
            attempts += 1
            log.warning("turn_failed", session=session.name, error=failure.message)
            if attempts >= pacing.attempts:
                raise
            await anyio.sleep(pacing.backoff)
