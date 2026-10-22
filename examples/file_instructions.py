"""File-based instructions: a global file, the nearest AGENTS.md, and skills.

Copy this module onto AID_AGENTS_PATH and it offers the `file_instructions` provider and the `skill` tool. For
the session's directory the provider loads, in this order:

- the global instructions file (`AID_INSTRUCTIONS_FILE`, default `~/.config/aid/instructions.md`) as `global`,
- the nearest `AGENTS.md` walking up from the session's directory (the directory itself, then each parent)
  as `agents`; a directory without one offers its `CLAUDE.md` instead,
- an index of the skills: a `skills` header, then each skill as `skill:<name>` with only its description.

Skills live in the skills directories (`AID_SKILLS_DIR`, several joined with `:`, default `~/.config/aid/skills`),
one directory per skill holding a `SKILL.md`: the Agent Skills layout Claude Code and opencode use, so
`~/.claude/skills` works as one. The description is the frontmatter's `description`, else the first line of the
body. A directory earlier in `AID_SKILLS_DIR` wins a name. The agent loads a skill's body, and its other files,
with the `skill` tool when the work calls for it: the index costs a line per skill, the bodies nothing until used.

A file that is absent (or empty) is skipped: no instruction. A skill directory without a `SKILL.md` is ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import aid

ENV_INSTRUCTIONS_FILE: Final = "AID_INSTRUCTIONS_FILE"
ENV_SKILLS_DIR: Final = "AID_SKILLS_DIR"
# Per directory, in this order: a repository written for Claude Code keeps its rules in CLAUDE.md.
PROJECT_FILES: Final = ("AGENTS.md", "CLAUDE.md")
SKILLS_HEADER: Final = (
    "Skills hold instructions for particular work, listed below as `skill:<name>` with what each covers. "
    "Before you start work a skill covers, load it with the `skill` tool and follow it."
)


def _global_file() -> Path:
    if value := os.environ.get(ENV_INSTRUCTIONS_FILE):
        return Path(value).expanduser()
    return Path.home() / ".config" / "aid" / "instructions.md"


def _skills_dirs() -> list[Path]:
    if value := os.environ.get(ENV_SKILLS_DIR):
        return [Path(part).expanduser() for part in value.split(os.pathsep) if part]
    return [Path.home() / ".config" / "aid" / "skills"]


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def _nearest_agents(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        for name in PROJECT_FILES:
            if (candidate := directory / name).is_file():
                return candidate
    return None


def _skills() -> dict[str, Path]:
    """Each skill's SKILL.md by name, resolved: skill directories are often symlinks into a store."""
    found: dict[str, Path] = {}
    for directory in _skills_dirs():
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(("_", ".")) or not entry.is_dir() or entry.name in found:
                continue
            if (skill := entry / "SKILL.md").is_file():
                found[entry.name] = skill.resolve()
    return found


def _split(text: str) -> tuple[dict[str, str], str]:
    """Flat `key: value` frontmatter and the body after it. Enough for name and description, not YAML."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in (line.strip() for line in lines[1:]):
        return {}, text
    end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and not key.startswith((" ", "\t")):
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, "\n".join(lines[end + 1 :])


def _description(text: str) -> str:
    meta, body = _split(text)
    if description := meta.get("description"):
        return description
    return next((line.strip().lstrip("#").strip() for line in body.splitlines() if line.strip()), "")


@aid.mcptool
def skill(name: str, file: str | None = None) -> str:
    """Load a skill your instructions list as `skill:<name>`: its full instructions, and the names of its other
    files. `file` reads one of those files instead, by its path inside the skill."""
    skills = _skills()
    if (path := skills.get(name)) is None:
        raise aid.ModelRetry(f"no skill {name!r}; the skills are: {', '.join(skills) or 'none'}")
    directory = path.parent
    if file is not None:
        target = (directory / file).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise aid.ModelRetry(f"{file!r} is not a file of skill {name!r}")
        return target.read_text(encoding="utf-8", errors="replace")
    others = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != path)
    text = path.read_text(encoding="utf-8")
    if others:
        text += f"\n\n[Other files of this skill; read one with skill({name!r}, file=...): {', '.join(others)}]"
    return text


class FileInstructions(aid.InstructionsProvider):
    """Global instructions, the nearest AGENTS.md or CLAUDE.md, and an index of the skills."""

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        instructions: list[aid.Instruction] = []
        if (text := _read(_global_file())) is not None:
            instructions.append(aid.Instruction(name="global", text=text))
        agents = _nearest_agents(Path(context.cwd).expanduser())
        if agents is not None and (text := _read(agents)) is not None:
            instructions.append(aid.Instruction(name="agents", text=text))
        index = [
            aid.Instruction(name=f"skill:{name}", text=_description(text))
            for name, path in _skills().items()
            if (text := _read(path)) is not None
        ]
        if index:
            instructions.append(aid.Instruction(name="skills", text=SKILLS_HEADER))
            instructions.extend(index)
        return instructions
