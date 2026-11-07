"""Which sessions a list shows (#59): why a session is hidden, what a query's terms match, the order and the pages.

Pure: the daemon passes the sessions and the clock. `web/src/lib/tags.ts` reads the same terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from aid.protocol import SessionInfo, SessionPage

if TYPE_CHECKING:
    from collections.abc import Sequence

DAY: Final = 86400.0
STALE_AFTER: Final = 7 * DAY
STATES: Final = ("hidden", "running", "working", "stopped")
LINKED: Final = "linked"


@dataclass(frozen=True)
class Term:
    kind: Literal["word", "tag", "is"]
    key: str
    value: str = ""


def parse(query: str) -> list[Term]:
    """Whitespace-separated terms: `is:STATE`, `key=value`, or a bare word. `is:` is reserved: a tag key `is:x` is
    still found as `is:x=value`."""
    terms: list[Term] = []
    for word in query.split():
        key, sep, value = word.partition("=")
        if sep:
            terms.append(Term("tag", key, value))
        elif word.startswith("is:"):
            if (state := word[3:]) not in STATES:
                raise ValueError(f"unknown state {state!r}: is: takes {', '.join(STATES)}")
            terms.append(Term("is", state))
        else:
            terms.append(Term("word", word))
    return terms


def matches(info: SessionInfo, terms: Sequence[Term]) -> bool:
    """Every term holds: a word is a tag key or a part of the name (any case), `key=value` a tag."""
    name = info.name.lower()
    for term in terms:
        match term.kind:
            case "word":
                held = term.key in info.tags or term.key.lower() in name
            case "tag":
                held = info.tags.get(term.key) == term.value
            case "is":
                held = {
                    "hidden": info.hidden is not None,
                    "running": info.running,
                    "working": info.working,
                    "stopped": not info.running,
                }[term.key]
        if not held:
            return False
    return True


def annotate(infos: Sequence[SessionInfo], *, now: float, stale_after: float = STALE_AFTER) -> list[SessionInfo]:
    """`infos` with `hidden` set where a session's own reason leaves it unset: `linked` (a supervisor, or working
    for one), `tab of <parent>` (a child with no children of its own), or stale (stopped and unused for
    `stale_after`). The same rules as `web/src/lib/links.ts`."""
    names = {i.name for i in infos}
    parents = {i.name: i.parent for i in infos}
    supervisors = {i.supervisor for i in infos if i.supervisor}
    holders = {i.parent for i in infos if i.parent}

    def linked(name: str) -> bool:
        # Bounded by the list's length: parent links come from specs, and nothing here trusts them to end.
        for _ in range(len(infos) + 1):
            if name in supervisors:
                return True
            parent = parents.get(name)
            if parent is None:
                return False
            name = parent
        return False

    def reason(info: SessionInfo) -> str | None:
        if info.hidden is not None:
            return info.hidden
        if linked(info.name):
            return LINKED
        if info.parent is not None and info.parent in names and info.name not in holders:
            return f"tab of {info.parent}"
        if not info.running and not info.working and (idle := now - info.active_at) >= stale_after:
            return f"stale: unused {int(idle // DAY)} days"
        return None

    return [info.model_copy(update={"hidden": reason(info)}) for info in infos]


def _order(info: SessionInfo) -> tuple[float, str]:
    return (-info.active_at, info.name)


def ordered(infos: Sequence[SessionInfo]) -> list[SessionInfo]:
    """The most recently active first (#54); names break ties."""
    return sorted(infos, key=_order)


def cursor(info: SessionInfo) -> str:
    return f"{info.active_at!r} {info.name}"


def _after(text: str) -> tuple[float, str]:
    at, sep, name = text.partition(" ")
    if not sep or not name:
        raise ValueError(f"not a cursor: {text!r}")
    return (-float(at), name)


def find(infos: Sequence[SessionInfo], query: str, *, limit: int, after: str | None = None) -> SessionPage:
    """The sessions `query` matches that are not hidden, in order, `limit` at a time after the cursor `after`.
    `is:hidden` or a word naming a session whole shows hidden matches too. A working session is on every first
    page, matched or not, beyond `limit`; later pages leave it out."""
    terms = parse(query)
    reveal = any(t.kind == "is" and t.key == "hidden" for t in terms)
    named = {t.key for t in terms if t.kind == "word"}
    candidates = [i for i in ordered(infos) if not i.working and matches(i, terms)]
    shown = [i for i in candidates if reveal or i.hidden is None or i.name in named]
    hidden = len(candidates) - len(shown)
    if after is not None:
        start = _after(after)
        shown = [i for i in shown if _order(i) > start]
    page = shown[:limit]
    more = len(shown) > limit
    working = [i for i in infos if i.working] if after is None else []
    return SessionPage(
        sessions=ordered([*working, *page]),
        hidden=hidden,
        next=cursor(page[-1]) if more and page else None,
    )
