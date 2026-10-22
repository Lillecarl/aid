"""Agent Skills (agentskills.io), as Claude Code, opencode and codex read them: a directory per skill holding a
`SKILL.md` with YAML frontmatter. Sessions see each skill's name and description; the `skill` tool loads the body
when the work calls for it.

Where skills come from, the first directory with a name winning:

- the session's project: `.aid/skills`, `.agents/skills` and `.claude/skills` in the session's directory and each
  parent up to the git root, nearest first,
- `AID_SKILLS_PATH`, directories joined with `:`,
- `~/.config/aid/skills` (XDG_CONFIG_HOME), `~/.agents/skills`, `~/.claude/skills`.

Frontmatter follows the spec: `name` (1-64 lowercase letters, digits and single hyphens, equal to the directory's
name) and `description` (1-1024 characters) are required. A skill that breaks either is left out and logged.
Drop one skill from a session with `skip_instructions: ["skill:<name>"]`; drop them all by leaving the `skills`
provider out of `instruction_providers`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import structlog
import yaml

import aid

log = structlog.get_logger(__name__)

ENV_SKILLS_PATH: Final = "AID_SKILLS_PATH"
SKILL_FILE: Final = "SKILL.md"
PROJECT_DIRS: Final = (".aid/skills", ".agents/skills", ".claude/skills")
NAME: Final = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
HEADER: Final = (
    "Skills hold instructions for particular work, each listed below as `skill:<name>` with what it covers. "
    "Before you start work a skill covers, load it with the `skill` tool and follow it."
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    """The SKILL.md, resolved: skill directories are often symlinks into a store."""
    body: str


def search_dirs(cwd: Path, env: dict[str, str] | None = None) -> list[Path]:
    env = dict(os.environ) if env is None else env
    dirs: list[Path] = []
    for directory in (cwd, *cwd.parents):
        dirs.extend(directory / sub for sub in PROJECT_DIRS)
        if (directory / ".git").exists():
            break
    dirs.extend(Path(part).expanduser() for part in env.get(ENV_SKILLS_PATH, "").split(":") if part)
    config = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    home = Path(env.get("HOME") or Path.home())
    dirs.extend([config / "aid" / "skills", home / ".agents" / "skills", home / ".claude" / "skills"])
    return dirs


def parse(path: Path) -> Skill:
    """One SKILL.md, checked against the spec. ValueError says what is wrong."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n?(.*)", text, re.DOTALL)
    if match is None:
        raise ValueError("no YAML frontmatter between --- lines")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise ValueError(f"frontmatter is not YAML: {error}") from error
    if not isinstance(meta, dict):
        raise ValueError("frontmatter is not a mapping")
    fields = cast(dict[str, object], meta)
    name, description = fields.get("name"), fields.get("description")
    if not isinstance(name, str) or not NAME.match(name) or len(name) > 64:
        raise ValueError(f"name {name!r} is not 1-64 lowercase letters, digits and single hyphens")
    if name != path.parent.name:
        raise ValueError(f"name {name!r} differs from its directory {path.parent.name!r}")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
        raise ValueError("description is missing or longer than 1024 characters")
    return Skill(name=name, description=" ".join(description.split()), path=path.resolve(), body=match.group(2))


def discover(cwd: Path, env: dict[str, str] | None = None) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for directory in search_dirs(cwd, env):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            file = entry / SKILL_FILE
            if entry.name in found or not file.is_file():
                continue
            try:
                found[entry.name] = parse(file)
            except (OSError, UnicodeDecodeError, ValueError) as error:
                log.warning("skill_skipped", path=str(file), reason=str(error))
    return found


@aid.mcptool
def skill(name: str, file: str | None = None) -> str:
    """Load a skill your instructions list as `skill:<name>`: its instructions, and the other files in its
    directory. `file` reads one of those files instead, by its path inside the skill."""
    skills = discover(Path.cwd())
    if (found := skills.get(name)) is None:
        raise aid.ModelRetry(f"no skill {name!r}; the skills are: {', '.join(sorted(skills)) or 'none'}")
    directory = found.path.parent
    if file is not None:
        target = (directory / file).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise aid.ModelRetry(f"{file!r} is not a file of skill {name!r}")
        return target.read_text(encoding="utf-8", errors="replace")
    others = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != found.path)
    head = f"Skill {name}, from {directory}."
    if others:
        head += f" Its other files, read with skill({name!r}, file=...): {', '.join(others)}."
    return f"{head}\n\n{found.body.strip()}\n"


class Skills(aid.InstructionsProvider):
    """Agent Skills: each skill's name and description; the `skill` tool loads one."""

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        skills = discover(Path(context.cwd).expanduser())
        if not skills:
            return []
        return [
            aid.Instruction(name="skills", text=HEADER),
            *(aid.Instruction(name=f"skill:{s.name}", text=s.description) for s in skills.values()),
        ]
