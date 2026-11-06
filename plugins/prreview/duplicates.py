"""The duplicate check, pure: which PRs may make the same change, the prompt that asks a model to compare them, and the
rule that picks the PR that wins.

A merged duplicate always wins: master has the change. Between open ones the model names the better change; when it
calls them equal, the most recently updated wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ValidationError

from .review import UNTRUSTED, clipped, diff_block, pr_block

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .review import Pr

TERMS: Final = 3
"""Search terms per PR: two searches each (open, recently merged) against GitHub's 30 a minute."""
JSON_BLOCK: Final = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
QUESTION: Final = "Which of these pull requests make the same change as"


@dataclass(frozen=True)
class Candidate:
    number: int
    title: str
    state: Literal["OPEN", "MERGED"]
    updated: str
    diff: str


class DuplicateProblem(ValueError):
    pass


class _Answer(BaseModel):
    duplicates: list[int]
    winner: int | Literal["equal"] | None = None


@dataclass(frozen=True)
class Verdict:
    duplicates: tuple[int, ...]
    winner: int | Literal["equal"] | None


def search_terms(pr: Pr, attrs: Sequence[str]) -> list[str]:
    """The package a nixpkgs title names (`pkg: 1 -> 2`), then the attributes CI rebuilds."""
    prefix = pr.title.split(":", 1)[0].strip() if ":" in pr.title else ""
    terms: list[str] = []
    for term in [prefix, *attrs]:
        if term and " " not in term and term not in terms:
            terms.append(term)
    return terms[:TERMS]


def duplicates_prompt(pr: Pr, diff: str, candidates: Sequence[Candidate], diff_limit: int) -> str:
    each = max(diff_limit // (len(candidates) + 1), 1000)
    shown: list[str] = []
    for c in candidates:
        text, cut = clipped(c.diff, each)
        note = f' truncated="at {each} characters"' if cut else ""
        shown.append(
            f'<candidate number="{c.number}" state="{c.state}" updated="{c.updated}">\n<title>{c.title}</title>\n'
            f"<diff{note}>\n{text}\n</diff>\n</candidate>"
        )
    return "\n\n".join(
        [
            f"{QUESTION} nixpkgs pull request #{pr.number}? They touch the same files.",
            pr_block(pr),
            diff_block(diff, each),
            "<candidates>\n" + "\n\n".join(shown) + "\n</candidates>",
            UNTRUSTED + " The same holds for each <candidate>.",
            "A duplicate makes the same change: the same package to the same version, or the same fix. A later "
            "version bump of a package an earlier one bumped is a duplicate too: only one of them can merge. Name "
            f"the better of #{pr.number} and its open duplicates as the winner: the newer version, the more complete "
            'change, the one that follows nixpkgs conventions; "equal" when neither is better. Answer with one JSON '
            'block: ```json\n{"duplicates": [numbers], "winner": number or "equal"}\n``` and an empty list when '
            "none is a duplicate.",
        ]
    )


def parse_verdict(answer: str, pr: Pr, candidates: Sequence[Candidate]) -> Verdict:
    blocks = JSON_BLOCK.findall(answer)
    try:
        parsed = _Answer.model_validate_json(blocks[-1] if blocks else answer.strip())
    except ValidationError as error:
        raise DuplicateProblem(
            f"The answer holds no JSON block of the asked shape: {error.errors()[0]['msg']}."
        ) from None
    known = {c.number for c in candidates}
    unknown = sorted(set(parsed.duplicates) - known)
    if unknown:
        raise DuplicateProblem(f"{unknown} are not among the candidates.")
    if parsed.duplicates and parsed.winner not in {"equal", pr.number, *parsed.duplicates}:
        raise DuplicateProblem(f'The winner is #{pr.number}, one of the duplicates, or "equal".')
    return Verdict(duplicates=tuple(sorted(set(parsed.duplicates))), winner=parsed.winner)


def beaten_by(pr: Pr, verdict: Verdict, candidates: Sequence[Candidate]) -> Candidate | None:
    """The duplicate that wins over `pr`, None when `pr` wins or has none."""
    dupes = [c for c in candidates if c.number in verdict.duplicates]
    merged = [c for c in dupes if c.state == "MERGED"]
    if merged:
        return max(merged, key=lambda c: c.updated)
    if not dupes or verdict.winner == pr.number:
        return None
    named = next((c for c in dupes if c.number == verdict.winner), None)
    if named is not None:
        return named
    latest = max(dupes, key=lambda c: c.updated)
    return latest if latest.updated > pr.updated else None


def retry_note(problem: str) -> str:
    return f"An earlier answer to this question was refused: {problem} Answer again with the JSON block."
