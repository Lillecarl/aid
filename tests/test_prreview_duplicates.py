from __future__ import annotations

import pytest

from plugins.prreview.duplicates import (
    Candidate,
    DuplicateProblem,
    Verdict,
    beaten_by,
    duplicates_prompt,
    parse_verdict,
    search_terms,
)
from plugins.prreview.review import Pr

PR = Pr(number=7, head="h", base="master", title="hello: 1 -> 2", body="", author="a", url="u", updated="2026-11-03")


def candidate(number: int, state: str = "OPEN", updated: str = "2026-11-02") -> Candidate:
    return Candidate(number=number, title="t", state="OPEN" if state == "OPEN" else "MERGED", updated=updated, diff="d")


def test_search_terms_are_the_title_s_package_then_the_rebuilds() -> None:
    assert search_terms(PR, ["hello", "hello.tests", "a", "b"]) == ["hello", "hello.tests", "a"]
    assert search_terms(Pr(7, "h", "m", "Update things", "", "a", "u"), ["x"]) == ["x"]


def test_a_merged_duplicate_always_wins() -> None:
    older_merged = candidate(5, "MERGED", "2026-10-01")
    verdict = Verdict(duplicates=(5, 8), winner=7)
    assert beaten_by(PR, verdict, [older_merged, candidate(8)]) == older_merged


def test_the_named_winner_wins_and_equals_go_to_the_latest() -> None:
    earlier, later = candidate(8, updated="2026-11-01"), candidate(9, updated="2026-11-05")
    assert beaten_by(PR, Verdict(duplicates=(8,), winner=7), [earlier]) is None
    assert beaten_by(PR, Verdict(duplicates=(8,), winner=8), [earlier]) == earlier
    assert beaten_by(PR, Verdict(duplicates=(8,), winner="equal"), [earlier]) is None
    assert beaten_by(PR, Verdict(duplicates=(8, 9), winner="equal"), [earlier, later]) == later
    assert beaten_by(PR, Verdict(duplicates=(), winner=None), [later]) is None


def test_an_answer_is_its_last_json_block_and_names_only_candidates() -> None:
    candidates = [candidate(8)]
    answer = 'First thought: ```json\n{"duplicates": []}\n``` then\n```json\n{"duplicates": [8], "winner": 8}\n```'
    assert parse_verdict(answer, PR, candidates) == Verdict(duplicates=(8,), winner=8)
    with pytest.raises(DuplicateProblem, match="not among the candidates"):
        parse_verdict('```json\n{"duplicates": [3], "winner": 3}\n```', PR, candidates)
    with pytest.raises(DuplicateProblem, match="The winner is"):
        parse_verdict('```json\n{"duplicates": [8], "winner": 4}\n```', PR, candidates)
    with pytest.raises(DuplicateProblem, match="no JSON block"):
        parse_verdict("They look alike.", PR, candidates)


def test_the_prompt_shows_each_candidate_with_its_state() -> None:
    prompt = duplicates_prompt(PR, "diff", [candidate(8), candidate(9, "MERGED")], 60_000)
    assert '<candidate number="8" state="OPEN"' in prompt
    assert '<candidate number="9" state="MERGED"' in prompt
