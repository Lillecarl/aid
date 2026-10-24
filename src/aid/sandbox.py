"""bubblewrap around the native `shell` tool's bash (`NativeSpec.sandbox`).

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
    from aid.spec import Sandbox

BWRAP: Final = "bwrap"


def bwrap_argv(sandbox: Sandbox, cwd: str) -> list[str]:
    """The bwrap prefix for a command that runs in `cwd`."""
    binds = [arg for path in (cwd, *sandbox.writable) for arg in ("--bind", path, path)]
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
        cwd,
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
