"""File-based instructions: a global file and the nearest AGENTS.md.

Copy this module onto AID_AGENTS_PATH and it offers the `file_instructions` provider. For the session's
directory it loads, in this order:

- the global instructions file (`AID_INSTRUCTIONS_FILE`, default `~/.config/aid/instructions.md`) as `global`,
- the nearest `AGENTS.md` walking up from the session's directory (the directory itself, then each parent)
  as `agents`; a directory without one offers its `CLAUDE.md` instead.

A file that is absent (or empty) is skipped: no instruction. Skills come from the bundled `skills` plugin.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import aid

ENV_INSTRUCTIONS_FILE: Final = "AID_INSTRUCTIONS_FILE"
# Per directory, in this order: a repository written for Claude Code keeps its rules in CLAUDE.md.
PROJECT_FILES: Final = ("AGENTS.md", "CLAUDE.md")


def _global_file() -> Path:
    if value := os.environ.get(ENV_INSTRUCTIONS_FILE):
        return Path(value).expanduser()
    return Path.home() / ".config" / "aid" / "instructions.md"


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


class FileInstructions(aid.InstructionsProvider):
    """Global instructions and the nearest AGENTS.md or CLAUDE.md."""

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        instructions: list[aid.Instruction] = []
        if (text := _read(_global_file())) is not None:
            instructions.append(aid.Instruction(name="global", text=text))
        agents = _nearest_agents(Path(context.cwd).expanduser())
        if agents is not None and (text := _read(agents)) is not None:
            instructions.append(aid.Instruction(name="agents", text=text))
        return instructions
