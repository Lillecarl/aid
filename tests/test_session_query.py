from __future__ import annotations

import pytest

from aid.protocol import SessionInfo
from aid.session_query import DAY, annotate, find, matches, ordered, parse
from aid.spec import AgentKind

NOW = 100 * DAY


def info(name: str, *, at: float = NOW, **fields: object) -> SessionInfo:
    return SessionInfo.model_validate(
        {"name": name, "kind": AgentKind.NATIVE, "running": False, "active_at": at, **fields}
    )


@pytest.mark.parametrize(
    ("query", "shown"),
    [
        ("", True),
        ("team", True),
        ("team=web", True),
        ("team=api", False),
        ("REV", True),
        ("q", False),
        ("is:stopped team", True),
        ("is:running", False),
    ],
)
def test_terms_match_names_tags_and_states(query: str, shown: bool) -> None:
    assert matches(info("nixrev", tags={"team": "web"}), parse(query)) is shown


def test_an_unknown_state_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown state 'busy'"):
        parse("is:busy")


def test_hiding_names_its_reason() -> None:
    infos = [
        info("own", hidden="PR reviewed: COMMENT"),
        info("worker", supervisor="boss"),
        info("boss"),
        info("run"),
        info("run.7", parent="run"),
        info("run.7.judge", parent="run.7"),
        info("old", at=NOW - 9 * DAY),
        info("old-but-running", at=NOW - 9 * DAY, running=True),
        info("recent", at=NOW - 6 * DAY),
        # At work, so in view, whatever would hide it.
        info("run.7.security", parent="run.7", working=True),
        info("busy-own", hidden="done", working=True),
        info("busy-supervisor", working=True),
        info("busy-worker", supervisor="busy-supervisor"),
    ]
    reasons = {i.name: i.hidden for i in annotate(infos, now=NOW)}
    assert reasons == {
        "own": "PR reviewed: COMMENT",
        "worker": None,
        "boss": "linked",
        "run": None,
        "run.7": None,
        "run.7.judge": "tab of run.7",
        "old": "stale: unused 9 days",
        "old-but-running": None,
        "recent": None,
        "run.7.security": None,
        "busy-own": None,
        "busy-supervisor": None,
        "busy-worker": None,
    }


def test_working_sessions_come_first() -> None:
    infos = [info("new", at=NOW), info("old-busy", at=NOW - DAY, working=True), info("older", at=NOW - 2 * DAY)]
    assert [i.name for i in ordered(infos)] == ["old-busy", "new", "older"]


def test_a_find_leaves_hidden_sessions_out_and_counts_them() -> None:
    infos = annotate([info("a"), info("b", hidden="done"), info("c", at=NOW - 30 * DAY)], now=NOW)
    page = find(infos, "", limit=10)
    assert ([i.name for i in page.sessions], page.hidden) == (["a"], 2)
    revealed = find(infos, "is:hidden", limit=10)
    assert ([(i.name, i.hidden) for i in revealed.sessions], revealed.hidden) == (
        [("b", "done"), ("c", "stale: unused 30 days")],
        0,
    )
    assert [i.name for i in find(infos, "b", limit=10).sessions] == ["b"]


def test_a_working_session_is_on_every_first_page() -> None:
    infos = [info("idle", at=NOW), info("busy", at=NOW - DAY * 30, working=True, hidden="done")]
    assert [i.name for i in find(infos, "idle", limit=1).sessions] == ["busy", "idle"]
    assert [i.name for i in find(infos, "nothing", limit=1).sessions] == ["busy"]


def test_pages_follow_the_most_recently_active_first() -> None:
    infos = [info(f"s{n}", at=NOW - n) for n in range(5)] + [info("w", working=True)]
    first = find(infos, "s", limit=2)
    assert [i.name for i in first.sessions] == ["w", "s0", "s1"]
    names: list[str] = [i.name for i in first.sessions if not i.working]
    after = first.next
    while after is not None:
        page = find(infos, "s", limit=2, after=after)
        assert not any(i.working for i in page.sessions)
        names += [i.name for i in page.sessions]
        after = page.next
    assert names == ["s0", "s1", "s2", "s3", "s4"]
    with pytest.raises(ValueError, match="not a cursor"):
        find(infos, "", limit=2, after="nonsense")
