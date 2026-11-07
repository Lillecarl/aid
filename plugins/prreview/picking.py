"""Which PR to review next (pure). GitHub's search answers at most 1000 hits in one order, and more match than that,
so one order hides most PRs for good. A pick draws from several orders (User, 2026-11-08: "mix and match a little",
upvotes as a metric, "a bit of rng"): a source at random by its weight, then a random one of its first unreviewed
hits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

HEAD_START: Final = 10
"""A pick is uniform among a source's first this many unreviewed hits: its order still counts, loosely."""


class Source(BaseModel):
    """One `gh search prs --sort SORT --order ORDER` and how often a pick draws from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sort: Literal["updated", "created", "reactions-+1", "reactions", "comments", "interactions"]
    order: Literal["asc", "desc"] = "desc"
    weight: Annotated[float, Field(gt=0)] = 1.0


SOURCES: Final = (
    Source(sort="updated", order="desc", weight=0.5),
    Source(sort="reactions-+1", order="desc", weight=0.3),
    Source(sort="updated", order="asc", weight=0.2),
)
"""Active PRs most, upvoted ones next, the longest waiting now and then."""


def pick_order(
    hits: Sequence[tuple[Source, Sequence[int]]],
    reviewed: set[int],
    rng: random.Random,
    *,
    head_start: int = HEAD_START,
) -> list[int]:
    """Every hit once: unreviewed ones in picking order, then reviewed ones (a head may have moved) in the first
    source's order. `hits`: each source with its numbers, in its order."""
    left = [(source.weight, [n for n in numbers if n not in reviewed]) for source, numbers in hits]
    order: list[int] = []
    taken: set[int] = set()
    while live := [entry for entry in left if entry[1]]:
        numbers = rng.choices(live, weights=[weight for weight, _ in live])[0][1]
        number = rng.choice(numbers[:head_start])
        order.append(number)
        taken.add(number)
        left = [(weight, [n for n in rest if n not in taken]) for weight, rest in live]
    for _, numbers in hits:
        for number in numbers:
            if number not in taken:
                order.append(number)
                taken.add(number)
    return order
