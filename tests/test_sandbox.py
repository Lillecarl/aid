from __future__ import annotations

import shutil
import subprocess  # noqa: TID251 -- a synchronous probe at collection, to skip where bwrap cannot run
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.protocol import AidError
from aid.sandbox import bwrap_argv
from aid.spec import Sandbox
from tests.conftest import py_spec
from tests.test_coding import plan

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


def bwrap_runs() -> bool:
    if shutil.which("bwrap") is None:
        return False
    probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "true"], check=False, capture_output=True)
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(not bwrap_runs(), reason="bwrap is missing or cannot make namespaces here")


def test_the_tmpfs_comes_before_the_binds_and_the_network_is_optional() -> None:
    argv = bwrap_argv(Sandbox(network=False, writable=["/var/cache/x"]), "/tmp/work")
    assert argv.index("--tmpfs") < argv.index("--bind")
    assert argv[argv.index("--bind") : argv.index("--bind") + 6] == [
        "--bind",
        "/tmp/work",
        "/tmp/work",
        "--bind",
        "/var/cache/x",
        "/var/cache/x",
    ]
    assert "--unshare-net" in argv
    assert "--unshare-net" not in bwrap_argv(Sandbox(), "/tmp/work")
    assert argv[-1] == "--"


def test_writable_paths_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        Sandbox(writable=["cache"])


@needs_bwrap
async def test_the_shell_writes_only_inside_its_directory(daemon: Paths, tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    spec = py_spec(work, "agents:coder", aid.PermissionMode.ALLOW).model_copy(update={"sandbox": Sandbox()})
    steps = plan(
        ("shell", {"command": "echo inside > made.txt && cd sub"}),
        # The session's parent is under /tmp, which is the sandbox's own tmpfs; /usr is the read-only root.
        ("shell", {"command": f"pwd; echo outside > {tmp_path / 'escaped.txt'}; echo root > /usr/aid-escape"}),
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("boxed", spec)
            result = await session.run(steps)
    assert (work / "made.txt").read_text() == "inside\n"
    assert not (tmp_path / "escaped.txt").exists()
    assert str(work / "sub") in str(result.output)
    assert "Read-only file system" in str(result.output)


async def test_a_session_that_asks_for_a_sandbox_never_runs_without_one(daemon: Paths, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:coder", PATH="/var/empty").model_copy(update={"sandbox": Sandbox()})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            with pytest.raises(AidError, match="bwrap is not on PATH"):
                await (await client.create("boxed", spec)).run(plan(("shell", {"command": "true"})))
