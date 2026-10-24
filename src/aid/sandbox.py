"""bubblewrap around what a native session's tools start (`NativeSpec.sandbox`): the `shell` tool's bash, `background`
tasks, and `python` scripts but `in-loop`, which runs in the worker and is refused.

Measured with bubblewrap 0.12.0: in `--ro-bind / /` writes outside the binds fail with EROFS, `/dev/null` takes
writes under `--dev /dev`, and `--unshare-net` refuses connections. Unix sockets on the read-only root still
connect: the nix daemon (so `nix build` still writes the store through it) and aid's own sockets. `--tmpfs /tmp`
must come before the binds, or it hides a session directory under /tmp.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Final

import anyio

from aid.protocol import AidError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aid.spec import Sandbox

BWRAP: Final = "bwrap"


def bwrap_argv(sandbox: Sandbox, cwd: str, *, chdir: str | None = None, writable: Sequence[str] = ()) -> list[str]:
    """The bwrap prefix for a command of the session in `cwd`, started in `chdir` (default `cwd`). `writable` adds
    to the spec's: a script runner's own store."""
    binds = [arg for path in (cwd, *sandbox.writable, *writable) for arg in ("--bind", path, path)]
    return [
        BWRAP,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *binds,
        *([] if sandbox.network else ["--unshare-net"]),
        "--die-with-parent",
        "--chdir",
        chdir or cwd,
        "--",
    ]


async def check(sandbox: Sandbox, cwd: str) -> None:
    """Raise unless the sandbox runs here: a session that asked for one never runs without it."""
    if shutil.which(BWRAP) is None:
        raise AidError("sandbox_unavailable", f"the session asks for a sandbox, and {BWRAP} is not on PATH")
    result = await anyio.run_process([*bwrap_argv(sandbox, cwd), "true"], check=False)
    if result.returncode != 0:
        reason = result.stderr.decode(errors="replace").strip() or f"exit {result.returncode}"
        raise AidError("sandbox_unavailable", f"the session asks for a sandbox, and {BWRAP} fails here: {reason}")
