"""What supervision decides, without IO: scores, episodes, goals, and the texts a supervisor reads."""

from __future__ import annotations

import difflib
import time
import uuid
from typing import TYPE_CHECKING, Final

from aid.protocol import AidError, Goal, GoalState, Output, history_line

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aid.protocol import HistoryEntry, HistoryItem, HistoryPage, PermissionRequest

SUPERVISION_ITEMS: Final = 10
"""History entries a supervision firing carries: the supervised session's recent tail, newest last."""
SUPERVISION_ITEM_CHARS: Final = 400
"""Characters of one tail line; longer bodies clip with their excess counted."""
SUPERVISION_SCORE_OUTPUTS: Final = 2
"""Latest output pair the loop score compares: old repetition must not conceal new progress."""
SUPERVISION_LOOP_THRESHOLD: Final = 0.85
"""Loop score at or above this reads as stuck; advisory, the supervisor judges."""
SEARCH_PAGE: Final = 1000
"""History entries one search step scans: page-sized reads, newest first."""


def _supervision_line(item: HistoryItem) -> str | None:
    """One clipped line of a supervised session's tail."""
    if (body := history_line(item)) is None:
        return None
    if len(body) > SUPERVISION_ITEM_CHARS:
        body = f"{body[:SUPERVISION_ITEM_CHARS]}\n… {len(body) - SUPERVISION_ITEM_CHARS} more characters"
    return body


def _normalized(body: str) -> str:
    """One line, one spacing, one case: what similarity compares."""
    return " ".join(body.split()).casefold()


def loop_score(items: Sequence[HistoryItem]) -> float | None:
    """Similarity of the latest two answers, never an older repeated pair. Missing/cancelled answers reset
    the signal rather than scoring the string 'None'. Advisory — the supervisor judges."""
    outputs = [item for item in items if isinstance(item, Output)][-SUPERVISION_SCORE_OUTPUTS:]
    if len(outputs) < SUPERVISION_SCORE_OUTPUTS:
        return None
    if any(item.output is None or item.stop_reason != "end_turn" or not str(item.output).strip() for item in outputs):
        return None
    first, second = (_normalized(str(item.output)) for item in outputs)
    return difflib.SequenceMatcher(None, first, second).ratio()


def supervision_text(
    name: str, reason: str, page: HistoryPage, goals: list[Goal], *, outputs: list[Output] | None = None
) -> str:
    """What the supervisor is told when a supervised turn ends: the session, how the turn closed, its open
    goals, its recent history clipped small, and the loop score over recent outputs. Cheap by design: the
    supervisor reads this inline, and reaches for more history only when it needs it. Bounded by
    construction: open goals plus at most `SUPERVISION_ITEMS` lines of `SUPERVISION_ITEM_CHARS` characters."""
    lines = [f"Supervised session {name!r} ended a turn ({reason})."]
    if goals:
        lines.append(f"Open goals ({len(goals)}):")
        lines.extend(f"- {goal.id[:8]}: {goal.text}" for goal in goals)
    else:
        lines.append("No open goals.")
    if page.total > len(page.entries):
        lines.append(f"Showing the newest {len(page.entries)} of {page.total} history entries:")
    rendered = [line for entry in page.entries if (line := _supervision_line(entry.item)) is not None]
    if rendered:
        lines.extend(rendered)
    else:
        lines.append("(Its history holds nothing readable yet.)")
    score = loop_score(outputs if outputs is not None else [entry.item for entry in page.entries])
    if score is None:
        lines.append("Loop score: n/a (fewer than two recent outputs).")
    else:
        lines.append(f"Loop score: {score:.2f} over recent outputs (stuck at {SUPERVISION_LOOP_THRESHOLD:.2f}).")
    return "\n".join(lines)


def permission_backstop_text(name: str, request: PermissionRequest) -> str:
    """What the supervisor is told when a supervised session's permission request outwaits the person: the
    session, the tool, the question, and every option with its id. Small by design: the supervisor answers
    inline with the answer_permission tool, and stays silent to leave the request for the person."""
    lines = [
        f"Supervised session {name!r} waits on permission (request {request.request_id}"
        f", tool {request.tool_name or 'a question'}): {request.title or 'no question given'}."
    ]
    options = [
        f"- {option.option_id}: {option.name}{' (recommended)' if option.recommended else ''}"
        for option in request.options
    ]
    lines.extend(options or ["(It offers no options: answer only with words, or not at all.)"])
    lines.append(
        f"Answer with the answer_permission tool, naming {name!r} and the request id. "
        "A person answering first wins: then the request is gone, and silence is the right answer."
    )
    return "\n".join(lines)


def nudge_repeats(last: str | None, text: str) -> bool:
    """The nudge says what the last one said: the loop-score check turned around, over normalized texts.
    With no last nudge, anything goes."""
    if last is None:
        return False
    return difflib.SequenceMatcher(None, _normalized(last), _normalized(text)).ratio() >= SUPERVISION_LOOP_THRESHOLD


def supervision_stuck(
    name: str, supervisor: str, stuck_turns: int, nudges: int, budget: int, score: float | None
) -> tuple[int, int, str | None]:
    """The stuck episode after a supervised turn end scoring `score`: the next stuck count, the nudges kept,
    and attention for a person. Evidence of progress ends the episode; missing evidence does not renew budgets.
    At `budget` stuck turns after `budget` nudges the session needs a person instead of another nudge."""
    if score is not None and score < SUPERVISION_LOOP_THRESHOLD:
        return 0, 0, None
    stuck = stuck_turns + int(score is not None)
    if stuck > 0 and stuck >= budget and nudges >= budget:
        return (
            stuck,
            nudges,
            f"{name} is stuck for {stuck} turns after {nudges} nudges from {supervisor}; needs a person",
        )
    return stuck, nudges, None


def add_goal(goals: list[Goal], text: str) -> Goal:
    """Name a goal. Pure: the caller saves."""
    goal = Goal(id=uuid.uuid4().hex, text=text, updated_at=time.time())
    goals.append(goal)
    return goal


def set_goal(goals: list[Goal], goal_id: str, state: GoalState) -> Goal:
    """Mark a goal done or dropped. Pure: the caller saves."""
    for i, goal in enumerate(goals):
        if goal.id == goal_id:
            updated = goal.model_copy(update={"state": state, "updated_at": time.time()})
            goals[i] = updated
            return updated
    raise AidError("unknown_goal", f"no goal {goal_id!r}")


def matches(entries: Sequence[HistoryEntry], query: str) -> list[HistoryEntry]:
    """Entries whose rendered line holds `query`, case-insensitively, in the order given."""
    needle = query.casefold()
    return [e for e in entries if (line := history_line(e.item)) is not None and needle in line.casefold()]
