"""File access a script makes, seen through `sys.addaudithook` in the script's own process (`pyrun.script --files`).

An audit hook sees every Python-level file operation, `os.open` included; C code that calls libc itself does not
raise events. This is a guard against mistakes, not a sandbox.

Measured on Python 3.14.7: `open` (path, mode, flags; mode is None for `os.open`), `os.rename` (also `replace`),
`os.remove` (also `unlink`), `os.rmdir`, `os.mkdir`, `os.chmod`, `os.symlink`, `os.truncate`, `os.utime`,
`os.listdir`, `os.scandir`. `shutil` raises its own events and then the `os` ones, which are what count here, but
for `shutil.rmtree`. The `open` event carries no `dir_fd`: `os.open(name, dir_fd=…)` resolves against the cwd.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

type Access = Literal["read", "write"]

WRITE_FLAGS: Final = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# Event → (access, (path argument, its dir_fd argument or None)...). Positions measured on 3.14.7; a dir_fd of -1
# is none. `shutil.rmtree` removes by dir_fd, so a relative path there is relative to that directory, not the cwd.
EVENTS: Final[dict[str, tuple[Access, tuple[tuple[int, int | None], ...]]]] = {
    "os.rename": ("write", ((0, 2), (1, 3))),
    "os.remove": ("write", ((0, 1),)),
    "os.rmdir": ("write", ((0, 1),)),
    "os.mkdir": ("write", ((0, 2),)),
    "os.chmod": ("write", ((0, 2),)),
    "os.chown": ("write", ((0, 3),)),
    "os.symlink": ("write", ((1, 2),)),
    "os.link": ("write", ((1, 3),)),
    "os.truncate": ("write", ((0, None),)),
    "os.utime": ("write", ((0, 3),)),
    "os.listdir": ("read", ((0, None),)),
    "os.scandir": ("read", ((0, None),)),
    # Before any of its own os events: one answer for the whole tree.
    "shutil.rmtree": ("write", ((0, 1),)),
}
SYSTEM_READS: Final = ("/proc", "/dev", "/sys")
"""Read without asking: pyrun's scope finds its processes in /proc, and the interpreter reads the rest."""
QUIET_WRITES: Final = ("/dev/null", "/dev/tty")


@dataclass(frozen=True)
class FileAccess:
    access: Access
    path: str
    """Absolute, symlinks resolved."""
    event: str
    """The audit event: `open`, `os.rename`, …"""


type FilePolicy = Callable[[FileAccess], Awaitable[bool]]


def _inside(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _path(value: object, dir_fd: object = None) -> str | None:
    """None for a file descriptor: its opening was seen already."""
    if not isinstance(value, str | bytes | os.PathLike):
        return None
    try:
        path = Path(os.fsdecode(os.fspath(cast("str | bytes | os.PathLike[str]", value))))
        if isinstance(dir_fd, int) and dir_fd >= 0 and not path.is_absolute():
            path = Path(f"/proc/self/fd/{dir_fd}").readlink() / path
        return str(path.resolve())
    except TypeError, OSError:
        return None


def accesses(event: str, args: tuple[Any, ...]) -> list[FileAccess]:
    """What one audit event does to the filesystem; empty for events that touch no file."""
    if event == "open":
        path, mode, flags = (*args, None, None, None)[:3]
        if (resolved := _path(path)) is None:
            return []
        if isinstance(mode, str):
            write = any(c in mode for c in "wax+")
        else:
            write = isinstance(flags, int) and bool(flags & WRITE_FLAGS)
        return [FileAccess("write" if write else "read", resolved, event)]
    if (known := EVENTS.get(event)) is None:
        return []
    access, positions = known
    return [
        FileAccess(access, resolved, event)
        for i, fd in positions
        if i < len(args) and (resolved := _path(args[i], args[fd] if fd is not None and fd < len(args) else None))
    ]


class FileGuard:
    """Decides each access in the script's process: quiet inside `roots` (and reads of the interpreter's own files),
    otherwise `ask`, which blocks until the host answers. An approved path stays approved for the run, a directory
    with everything under it: one answer covers `shutil.rmtree`."""

    def __init__(self, roots: Iterable[str], ask: Callable[[FileAccess], tuple[bool, str | None]]) -> None:
        self._roots = [str(Path(r).resolve()) for r in roots]
        interpreter = [Path(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path) if p]
        self._reads = [str(p.resolve()) for p in interpreter if p.is_absolute()] + list(SYSTEM_READS)
        self._ask = ask
        self._approved: dict[Access, list[str]] = {"read": [], "write": []}

    def quiet(self, access: FileAccess) -> bool:
        if _inside(access.path, self._roots) or _inside(access.path, self._approved[access.access]):
            return True
        if access.access == "read":
            return _inside(access.path, self._reads) or _inside(access.path, self._approved["write"])
        return access.path in QUIET_WRITES

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        # Asking raises no file event of its own: `os.write` on an open descriptor has no audit event.
        for access in accesses(event, args):
            if self.quiet(access):
                continue
            allowed, reason = self._ask(access)
            if not allowed:
                raise PermissionError(f"pyrun policy refused to {access.access} {access.path}: {reason or 'denied'}")
            self._approved[access.access].append(access.path)
