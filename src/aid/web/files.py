"""A session's working directory, for the web UI's file tree and viewer.

aid web reads the files itself, as the user the daemon runs as. Every path resolves, symlinks and `..` included,
to somewhere inside the session's cwd; anything else is refused. A worker on another host would need these
through the worker instead.
"""

from __future__ import annotations

import os
import stat
from typing import Final

import anyio.to_thread
from pydantic import BaseModel, ConfigDict

from aid.confine import inside
from aid.protocol import AidError

MAX_FILE: Final = 1024 * 1024
"""Bytes of a file the viewer gets; the rest is cut off."""
_SNIFF: Final = 8192


class Entry(BaseModel):
    name: str
    dir: bool
    size: int | None
    """None for a directory, or an entry that could not be read."""


class FileView(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    path: str
    size: int
    text: str | None
    """None for a binary file."""
    truncated: bool
    highlights: list[tuple[int, int, str]] | None = None
    """UTF-16 start, end and Pygments class (`web.highlight`); None when no grammar knows the file."""


def _list(root: str, rel: str) -> list[Entry]:
    target = inside(root, rel)
    try:
        with os.scandir(target) as found:
            entries = list(found)
    except FileNotFoundError:
        raise AidError("not_found", f"no directory {rel!r}") from None
    except NotADirectoryError:
        raise AidError("not_a_directory", f"{rel!r} is not a directory") from None
    listed: list[Entry] = []
    for entry in entries:
        try:
            info = entry.stat()
        except OSError:
            listed.append(Entry(name=entry.name, dir=False, size=None))
            continue
        is_dir = stat.S_ISDIR(info.st_mode)
        listed.append(Entry(name=entry.name, dir=is_dir, size=None if is_dir else info.st_size))
    return sorted(listed, key=lambda e: (not e.dir, e.name.casefold()))


def _read(root: str, rel: str) -> FileView:
    target = inside(root, rel)
    try:
        with target.open("rb") as file:
            size = os.fstat(file.fileno()).st_size
            data = file.read(MAX_FILE)
    except FileNotFoundError:
        raise AidError("not_found", f"no file {rel!r}") from None
    except IsADirectoryError:
        raise AidError("not_a_file", f"{rel!r} is a directory") from None
    truncated = size > len(data)
    return FileView(path=rel, size=size, text=_text(data, truncated), truncated=truncated)


def _text(data: bytes, truncated: bool) -> str | None:
    """The data as UTF-8, or None if it is binary."""
    if b"\0" in data[:_SNIFF]:
        return None
    try:
        return data.decode()
    except UnicodeDecodeError as error:
        # The cut can split a character, which leaves up to three bytes that do not decode.
        if truncated and error.start >= len(data) - 3:
            return data[: error.start].decode()
        return None


MAX_COMPLETIONS: Final = 50


class DirCompletions(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    prefix: str
    """What a new session's working directory starts as (`aid web --cwd-prefix`), ending in `/`."""
    dirs: list[str]
    """Directories whose path starts with what was typed, each ending in `/`."""


def _complete(typed: str, home: str) -> list[str]:
    """Directories a partly typed absolute path can go on to. `~` is the home directory. Hidden ones only once
    the typed name starts with a dot. Any directory on the host: a session there runs commands as this user."""
    if typed == "~" or typed.startswith("~/"):
        typed = home + typed[1:]
    if not typed.startswith("/"):
        return []
    parent, _, start = typed.rpartition("/")
    try:
        with os.scandir(parent or "/") as found:
            names = sorted(
                entry.name
                for entry in found
                if entry.name.startswith(start)
                and (start.startswith(".") or not entry.name.startswith("."))
                and entry.is_dir()
            )
    except OSError:
        return []
    return [f"{parent}/{name}/" for name in names[:MAX_COMPLETIONS]]


async def complete_dirs(typed: str, home: str) -> list[str]:
    return await anyio.to_thread.run_sync(_complete, typed, home)


async def list_dir(root: str, rel: str) -> list[Entry]:
    return await anyio.to_thread.run_sync(_list, root, rel)


async def read_file(root: str, rel: str) -> FileView:
    return await anyio.to_thread.run_sync(_read, root, rel)
