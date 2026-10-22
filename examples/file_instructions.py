"""File-based instructions: a global file, the nearest AGENTS.md, and skills.

Copy this module onto AID_AGENTS_PATH and it offers the `file_instructions` provider. For the session's
directory it loads, in this order:

- the global instructions file (`AID_INSTRUCTIONS_FILE`, default `~/.config/aid/instructions.md`) as `global`,
- the nearest `AGENTS.md` walking up from the session's directory (the directory itself, then each parent)
  as `agents`; a directory without one offers its `CLAUDE.md` instead,
- every `SKILL.md` directly under a directory of the skills directory (`AID_SKILLS_DIR`, default
  `~/.config/aid/skills`) as `skill:<name>`: the skills directory holds one directory per skill, each with a
  `SKILL.md` whose first line describes it in one line. The whole file becomes the instruction's text.

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


def _global_file() -> Path:
    if value := os.environ.get(ENV_INSTRUCTIONS_FILE):
        return Path(value).expanduser()
    return Path.home() / ".config" / "aid" / "instructions.md"


def _skills_dir() -> Path:
    if value := os.environ.get(ENV_SKILLS_DIR):
        return Path(value).expanduser()
    return Path.home() / ".config" / "aid" / "skills"


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


def _skills() -> list[tuple[str, Path]]:
    directory = _skills_dir()
    if not directory.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith(("_", ".")) or not entry.is_dir():
            continue
        if (skill := entry / "SKILL.md").is_file():
            found.append((entry.name, skill))
    return found


class FileInstructions(aid.InstructionsProvider):
    """Global instructions, the nearest AGENTS.md, and every skill's SKILL.md."""

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        instructions: list[aid.Instruction] = []
        if (text := _read(_global_file())) is not None:
            instructions.append(aid.Instruction(name="global", text=text))
        agents = _nearest_agents(Path(context.cwd).expanduser())
        if agents is not None and (text := _read(agents)) is not None:
            instructions.append(aid.Instruction(name="agents", text=text))
        for name, skill in _skills():
            if (text := _read(skill)) is not None:
                instructions.append(aid.Instruction(name=f"skill:{name}", text=text))
        return instructions
