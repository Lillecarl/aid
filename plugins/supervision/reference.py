"""Read-only references a supervisor judges a command by: manual pages and tldr pages. Neither starts a program the
model names: `man` runs with a fixed argv, and a name must look like a command, never an option."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import anyio

from aid.agent import ModelRetry

COMMAND: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
SECTION: Final = re.compile(r"^[1-9][a-z]*$")
OVERSTRIKE: Final = re.compile(r".\x08")
"""groff's bold and underline for a terminal, which `man` keeps when its pager is `cat`."""
MAN_CHARS: Final = 20_000
"""Characters of a manual page one call returns; longer pages come in parts."""
SEARCH_CONTEXT: Final = 2
"""Lines kept around each line a search finds."""
ENV_TLDR_PAGES: Final = "AID_TLDR_PAGES"
"""The tldr-pages `pages` directory (`nix/tldr-pages.nix`); unset, tldr pages are unavailable."""
TLDR_PLATFORMS: Final = ("common", "linux")


def _command(name: str) -> str:
    if not COMMAND.match(name):
        raise ModelRetry(f"{name!r} is not a command name: give the program's name alone, such as 'rm'")
    return name


def _around(text: str, search: str) -> str:
    lines = text.splitlines()
    needle = search.casefold()
    keep: set[int] = set()
    for index, line in enumerate(lines):
        if needle in line.casefold():
            keep.update(range(max(0, index - SEARCH_CONTEXT), min(len(lines), index + SEARCH_CONTEXT + 1)))
    if not keep:
        return f"No line holds {search!r}."
    blocks: list[str] = []
    previous = -2
    for index in sorted(keep):
        if index != previous + 1:
            blocks.append("…")
        blocks.append(lines[index])
        previous = index
    return "\n".join(blocks)


async def man_page(command: str, section: str | None = None, search: str | None = None, offset: int = 0) -> str:
    """The manual page of `command` as plain text: what its options do. `search` returns only the lines holding
    that text, with two lines around each: use it for one option of a long page. Without it, a page longer than
    the limit comes in parts; pass the `offset` a part ends with."""
    argv = ["man", "-P", "cat"]
    if section is not None:
        if not SECTION.match(section):
            raise ModelRetry(f"{section!r} is not a manual section, such as '1' or '8'")
        argv.append(section)
    argv.append(_command(command))
    try:
        result = await anyio.run_process(argv, check=False, env={**os.environ, "MANWIDTH": "100"})
    except FileNotFoundError:
        return "`man` is not installed where this session runs."
    if result.returncode != 0:
        return f"No manual page for {command}: {result.stderr.decode(errors='replace').strip()}"
    text = OVERSTRIKE.sub("", result.stdout.decode(errors="replace"))
    if search:
        return _around(text, search)
    part = text[offset : offset + MAN_CHARS]
    if offset + MAN_CHARS < len(text):
        part += f"\n… {len(text) - offset - MAN_CHARS} more characters; offset={offset + MAN_CHARS} continues"
    return part


async def tldr_page(command: str) -> str:
    """The tldr page of `command`: its common uses, each with an example. Shorter than its manual page."""
    pages = os.environ.get(ENV_TLDR_PAGES)
    if not pages:
        return "tldr pages are not installed here."
    name = _command(command).lower()
    for platform in TLDR_PLATFORMS:
        page = anyio.Path(Path(pages) / platform / f"{name}.md")
        if await page.exists():
            return await page.read_text()
    return f"No tldr page for {command}."
