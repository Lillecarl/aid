"""Compose session instructions from instruction providers.

`compose` runs every selected provider and concatenates what they load under each instruction's name. A provider
that fails is skipped loudly, never fatally: one broken provider must not keep a session from starting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import structlog

from aid.agents import InstructionsContext
from aid.discovery import Catalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aid.agents import InstructionsProvider

log = structlog.get_logger(__name__)


USERTEE: Final = (
    "`usertee` is on PATH: pipe a command's output through it before any filter (`make 2>&1 | usertee | tail -n 20`)."
    " The person watching sees all of it live; you get only what the filter keeps. Use it for long or slow commands."
)
"""How agents learn of `aid/usertee.py`: in the environment section of native sessions, and in the instructions of
aid's MCP server for the rest."""


def environment(cwd: Path | str, session_name: str) -> str:
    """Facts every session needs and no provider owns. Static, so the prompt cache holds: the time travels in
    each turn's metadata instead.

    Measured 2026-10 on DeepSeek V4.1 Flash: an agent that `cd`ed its shell elsewhere did not know its own
    directory, searched `/` for a pyproject.toml, and worked in another checkout of the same repository.
    """
    return "\n".join(
        [
            "# environment",
            "",
            f"- Working directory: {Path(cwd).resolve()}. Relative paths in file tools resolve against it, and the"
            " shell starts there; return to it with this absolute path. The session can move (`change_directory`, or a"
            " person's /cd): each turn's metadata names the directory as it is now.",
            f"- aid session: {session_name}",
            f"- Platform: {sys.platform}",
            f"- {USERTEE}",
        ]
    )


def compose(
    cwd: Path | str,
    providers: Catalog | Mapping[str, type[InstructionsProvider]],
    *,
    select: list[str] | None = None,
    skip: frozenset[str] = frozenset(),
    session_name: str = "",
) -> str:
    """The selected providers' instructions for `cwd`, each under a `# name` header.

    None selects every provider in catalog order, otherwise only the named ones; unknown names log and are
    ignored. Instructions whose name is in `skip` are dropped. A provider that fails to construct or load is
    skipped loudly, never fatally.
    """
    table = providers.providers if isinstance(providers, Catalog) else providers
    names = list(table) if select is None else list(select)
    context = InstructionsContext(cwd=Path(cwd), session_name=session_name)
    parts: list[str] = []
    for name in names:
        if (cls := table.get(name)) is None:
            log.warning("instructions_provider_missing", provider=name)
            continue
        try:
            loaded = cls().load(context)
        except Exception as error:
            log.exception("instructions_provider_failed", provider=name, error=f"{type(error).__name__}: {error}")
            continue
        for instruction in loaded:
            if instruction.name not in skip:
                parts.append(f"# {instruction.name}\n\n{instruction.text}")
    return "\n\n".join(parts)
