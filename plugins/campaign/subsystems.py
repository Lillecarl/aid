"""Splitting a big repository, pure: the overview the model maps from, the map it answers with, and the assignment
of every file to exactly one subsystem.

The model proposes; this module decides. A file no subsystem's globs match goes to `OTHER`, and a map leaving too
many there is asked for again with what it missed: the model never decides what goes unread.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import PurePath
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .units import FileInfo

OVERVIEW_LINES: Final = 400
"""Lines of the directory overview at most: a deeper tree is summed into its shallower directories."""
MAX_SUBSYSTEMS: Final = 12
OTHER: Final = "other"
OTHER_SHARE: Final = 0.25
"""The share of files `OTHER` may hold before the map is asked for again."""
UNMATCHED_SAMPLE: Final = 40
ANSWER_CHARS: Final = 1500
"""Characters of one unit's answer the integration turn gets."""
INTEGRATION_CHARS: Final = 60_000


class Subsystem(BaseModel):
    """One part of a repository, as the model maps it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: Annotated[str, Field(min_length=1, max_length=60)]
    globs: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    purpose: Annotated[str, Field(max_length=1000)] = ""
    boundaries: Annotated[str, Field(max_length=1000)] = ""
    """What it takes from or gives to the outside and the other subsystems: where trust changes."""

    @field_validator("globs")
    @classmethod
    def _relative(cls, globs: list[str]) -> list[str]:
        return [glob.strip().removeprefix("./").strip("/") or "**" for glob in globs]


class SubsystemMap(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    subsystems: Annotated[list[Subsystem], Field(min_length=2, max_length=MAX_SUBSYSTEMS)]


def overview(files: Sequence[FileInfo]) -> str:
    """Directories with their file and line totals, deepest first summed upward until it fits `OVERVIEW_LINES`."""
    depth = max((len(PurePath(f.path).parts) for f in files), default=1)
    while True:
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for file in files:
            parts = PurePath(file.path).parts[:-1][:depth]
            key = "/".join(parts) + "/" if parts else "."
            totals[key][0] += 1
            totals[key][1] += file.lines
        if len(totals) <= OVERVIEW_LINES or depth == 0:
            break
        depth -= 1
    lines = [f"{key}  {count} files, {lines} lines" for key, (count, lines) in sorted(totals.items())]
    note = (
        ""
        if depth >= max((len(PurePath(f.path).parts) - 1 for f in files), default=0)
        else (f"(directories deeper than {depth} levels are summed into their parents)\n")
    )
    return note + "\n".join(lines[:OVERVIEW_LINES])


def _matches(path: str, glob: str) -> bool:
    """`glob` as a model writes one: a bare directory means everything under it."""
    pure = PurePath(path)
    if any(c in glob for c in "*?["):
        return pure.full_match(glob) or pure.full_match(f"{glob}/**")
    return path == glob or path.startswith(f"{glob}/")


def assign(files: Sequence[FileInfo], subsystems: Sequence[Subsystem]) -> list[list[FileInfo]]:
    """Each subsystem's files, the first whose globs match taking a file; one more list last, `OTHER`'s."""
    groups: list[list[FileInfo]] = [[] for _ in range(len(subsystems) + 1)]
    for file in files:
        index = next(
            (i for i, s in enumerate(subsystems) if any(_matches(file.path, g) for g in s.globs)), len(subsystems)
        )
        groups[index].append(file)
    return groups


class MapProblem(ValueError):
    """What to tell the model about its map before asking again."""


_FENCED: Final = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_map(answer: str, files: Sequence[FileInfo]) -> tuple[SubsystemMap, list[list[FileInfo]]]:
    """The map in `answer` and its assignment, or `MapProblem` saying what to fix."""
    found = _FENCED.search(answer)
    text = found.group(1) if found else answer[answer.find("{") : answer.rfind("}") + 1]
    try:
        mapped = SubsystemMap.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as error:
        raise MapProblem(f"The map is not valid: {error}") from None
    groups = assign(files, mapped.subsystems)
    if len(groups[-1]) > OTHER_SHARE * len(files):
        sample = "\n".join(f"- {f.path}" for f in groups[-1][:UNMATCHED_SAMPLE])
        raise MapProblem(
            f"{len(groups[-1])} of {len(files)} files match no subsystem's globs; cover them. Some of them:\n{sample}"
        )
    return mapped, groups


def map_prompt(brief: str, tree: str, files: int, problem: str | None = None) -> str:
    again = f"\n\nYour last map was refused. {problem}\n" if problem else ""
    return (
        f"Repository brief:\n{brief}\n\n"
        f"This repository has {files} files to audit, too many for one reader. Split it into 2 to {MAX_SUBSYSTEMS} "
        "subsystems that can each be audited on their own: group code that works together, such as a service, a "
        "library, deployment configuration or tests. You may ls, read and search to decide. Its directories:\n"
        f"{tree}\n{again}\n"
        'Answer with one JSON block: {"subsystems": [{"name": "...", "globs": ["dir/**", "file.py"], '
        '"purpose": "what it does", "boundaries": "what it takes from or gives to users, the network and the '
        'other subsystems"}]}. Globs are relative to the repository root; a file matching none is audited as '
        f'"{OTHER}". Report nothing yet.'
    )


def subsystem_context(mapped: SubsystemMap, index: int) -> str:
    """What a unit in subsystem `index` is told of its own subsystem and the others; `OTHER` is past the map."""
    if index == len(mapped.subsystems):
        own = f"Subsystem: {OTHER} (files no subsystem of the map covers)"
    else:
        part = mapped.subsystems[index]
        own = f"Subsystem: {part.name}\nPurpose: {part.purpose}\nBoundaries: {part.boundaries}"
    others = "\n".join(f"- {s.name}: {s.purpose}" for i, s in enumerate(mapped.subsystems) if i != index)
    return f"{own}\n\nThe other subsystems:\n{others}"


def integration_prompt(brief: str, mapped: SubsystemMap, answers: Sequence[tuple[str, str]]) -> str:
    """`answers`: each audited unit's subsystem name and final answer, clipped here."""
    parts = "\n".join(
        f"- {s.name} ({', '.join(s.globs)}): {s.purpose} Boundaries: {s.boundaries}" for s in mapped.subsystems
    )
    notes: list[str] = []
    used = 0
    for name, answer in answers:
        clipped = answer[:ANSWER_CHARS] + ("…" if len(answer) > ANSWER_CHARS else "")
        if used + len(clipped) > INTEGRATION_CHARS:
            notes.append(f"… {len(answers) - len(notes)} more unit answers left out")
            break
        notes.append(f"[{name}]\n{clipped}")
        used += len(clipped)
    return (
        f"Repository brief:\n{brief}\n\nIts subsystems, each audited on its own:\n{parts}\n\n"
        "What each audit answered, unit by unit:\n" + "\n\n".join(notes) + "\n\n"
        "Look for issues that cross subsystems, which no single audit could see: trust one subsystem places in "
        "another, authentication or authorization spanning them, secrets or data passed between them, "
        "configuration that weakens another part. Read and search the files you need. Record each real issue with "
        "report_finding; leave out what the audits above already found. End with a short summary."
    )
