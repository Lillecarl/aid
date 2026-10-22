"""Tools for a supervisor session: read a supervised session's history, track its goals, answer the
permission requests its allow-list names when the person does not.

    import aid
    from aid.agent import Agent

    boss = Agent("...", toolsets=[aid.supervisor_tools])

A supervisor names its subjects in their specs (`supervisor`); the tools take the supervised session's name and
call the bundled `supervision` plugin, which refuses any session its caller does not supervise. The daemon names
the caller from the session's own connection, so these tools work only inside an aid session.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Final

from aid.agent import ModelRetry, Toolset
from aid.client import connect
from aid.protocol import AidError, GoalState, HistoryPage, SearchResults, history_line

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import JsonValue

    from aid.protocol import HistoryEntry

PLUGIN: Final = "supervision"
SUPERVISION_LINE_CHARS: Final = 400
"""Characters of one history line a supervision tool returns; longer lines clip with their excess counted."""


async def _call(method: str, **args: JsonValue) -> JsonValue:
    async with connect() as client:
        return await client.call_plugin(PLUGIN, method, **args)


def _lines(entries: list[HistoryEntry]) -> str:
    rendered: list[str] = []
    for entry in entries:
        if (line := history_line(entry.item)) is None:
            continue
        if len(line) > SUPERVISION_LINE_CHARS:
            line = f"{line[:SUPERVISION_LINE_CHARS]}… {len(line) - SUPERVISION_LINE_CHARS} more"
        rendered.append(f"{entry.seq}: {line}")
    return "\n".join(rendered)


async def supervised_history(session: str, limit: int = 50) -> str:
    """The supervised session's recent history, newest last, one line per entry. Read this when a firing's
    clipped tail leaves questions open."""
    page = HistoryPage.model_validate(await _call("history", session=session, limit=limit))
    lines = _lines(page.entries)
    if page.total > len(page.entries):
        lines += f"\n… {page.total - len(page.entries)} older entries"
    return lines or "(no readable history)"


async def supervised_search(session: str, query: str, limit: int = 20) -> str:
    """History entries of the supervised session holding `query`, newest first, one line per entry."""
    results = SearchResults.model_validate(await _call("search", session=session, query=query, limit=limit))
    lines = _lines(results.entries)
    if results.total > len(results.entries):
        lines += f"\n… {results.total - len(results.entries)} older matches"
    return lines or "(no matches)"


async def goals(session: str) -> str:
    """Every goal of the supervised session, with ids and states, as JSON."""
    return json.dumps(await _call("goals", session=session), indent=2)


async def interrupt(session: str, turn: str) -> str:
    """Stop exactly the supervised turn you reviewed, retaining its completed work. Use this only after
    judging a suspected loop; send_message can give the next turn corrective instructions. A stale review
    does not stop newer work. Only your own supervised sessions are eligible."""
    try:
        await _call("interrupt", session=session, turn=turn)
    except AidError as error:
        if error.code == "stale_turn":
            return "The reviewed turn is no longer running; newer work was left alone."
        raise
    return f"Interruption requested for {session!r} turn {turn!r}."


async def goal_add(session: str, text: str) -> str:
    """Name a goal for the supervised session. Returns the goal, with its id, as JSON."""
    return json.dumps(await _call("goal_add", session=session, text=text), indent=2)


async def goal_done(session: str, goal_id: str) -> str:
    """Mark the supervised session's goal done. Returns the goal as JSON."""
    return json.dumps(await _call("goal_set", session=session, goal_id=goal_id, state=GoalState.DONE), indent=2)


async def goal_drop(session: str, goal_id: str) -> str:
    """Drop the supervised session's goal. Returns the goal as JSON."""
    return json.dumps(await _call("goal_set", session=session, goal_id=goal_id, state=GoalState.DROPPED), indent=2)


async def answer_permission(
    session: str, request_id: str, option_id: str | None = None, text: str | None = None
) -> str:
    """Answer a supervised session's waiting permission request: one of its options, or nothing to cancel
    it. `text` carries your own words: with an option, alongside the pick; alone, the whole answer, which
    only an `ask_user` question takes. The first resolution wins: when a person answered first, the
    request is gone, and this says so instead of failing."""
    try:
        await _call("answer_permission", session=session, request_id=request_id, option_id=option_id, text=text)
    except AidError as error:
        if error.code == "no_pending":
            return "The request is gone: a person answered it first, or the turn moved on."
        raise
    return f"Answered {request_id} with {option_id if option_id is not None else 'a cancel'}."


def _retrying[**P, R](tool: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """A tool's failure goes back to the model as a retry with the reason: an exception would end the run."""

    @functools.wraps(tool)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await tool(*args, **kwargs)
        except ModelRetry:
            raise
        except Exception as error:
            reason = error.message if isinstance(error, AidError) else str(error)
            raise ModelRetry(f"{type(error).__name__}: {reason}") from error

    return wrapper


supervisor_tools: Toolset = Toolset(
    [
        _retrying(supervised_history),
        _retrying(supervised_search),
        _retrying(goals),
        _retrying(goal_add),
        _retrying(goal_done),
        _retrying(goal_drop),
        _retrying(answer_permission),
        _retrying(interrupt),
    ]
)
