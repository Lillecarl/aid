from __future__ import annotations

import shutil
import subprocess  # noqa: TID251 -- a synchronous probe at collection, to skip where bwrap cannot run
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.protocol import AidError, Lifecycle, ToolCall
from aid.sandbox import bwrap_argv
from aid.spec import Sandbox
from tests.conftest import acp_spec, py_spec
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


@needs_bwrap
async def test_scripts_and_background_tasks_run_in_the_sandbox_too(daemon: Paths, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:coder", aid.PermissionMode.ALLOW).model_copy(update={"sandbox": Sandbox()})
    script = """
Path("by-script.txt").write_text("inside")
try:
    Path("/usr/aid-escape").write_text("root")
except OSError as error:
    print(type(error).__name__, error.errno)
await run("sh", "-c", "echo child > by-child.txt; echo x > /usr/aid-escape || echo child refused")
"""
    steps = plan(
        ("python", {"script": script}),
        ("background", {"argv": ["sh", "-c", "echo bg > by-bg.txt; echo x > /usr/aid-escape || echo bg refused"]}),
        ("task_output", {"task_id": "bg1", "wait": 10}),
        ("python", {"script": "print('hi')", "mode": "in-loop"}),
    )
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("boxed", spec)
            result = str((await session.run(steps)).output)
    assert [(tmp_path / f).read_text().strip() for f in ("by-script.txt", "by-child.txt", "by-bg.txt")] == [
        "inside",
        "child",
        "bg",
    ]
    assert "OSError 30" in result  # EROFS
    assert "child refused" in result
    assert "bg refused" in result
    assert "in-loop runs outside it" in result


@needs_bwrap
async def test_a_sandbox_change_restarts_the_worker_when_no_turn_runs(daemon: Paths, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:coder", aid.PermissionMode.ALLOW)
    sleeper = plan(("python", {"script": "import time; time.sleep(1.5)", "mode": "sync"}))
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("boxed", spec)
            await session.start()
            first = (await session.status()).pid
            await session.set_sandbox(Sandbox(network=False))
            idle = await session.status()
            during = None
            async for event in session.stream(sleeper):
                if isinstance(event, ToolCall) and during is None:
                    await session.set_sandbox(None)
                    during = await session.status()
            while (after := await session.status()).pid in (None, idle.pid):  # noqa: ASYNC110 -- the restart follows the turn's end
                await anyio.sleep(0.05)
            entries = [e.item for e in (await session.history(limit=1000)).entries]
    assert idle.pid != first and idle.sandbox == Sandbox(network=False)
    assert during is not None and during.pid == idle.pid and during.sandbox is None
    assert after.sandbox is None
    assert [e.detail for e in entries if isinstance(e, Lifecycle)] == ['{"network":false}', "off"]


async def test_only_native_sessions_take_a_sandbox(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("acp", acp_spec(tmp_path))
            with pytest.raises(AidError, match="only native sessions"):
                await session.set_sandbox(Sandbox())


async def test_the_daemon_checks_the_sandbox_on_the_workers_path(daemon: Paths, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:coder", PATH="/var/empty")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            session = await client.create("boxed", spec)
            with pytest.raises(AidError, match="bwrap is not on PATH"):
                await session.set_sandbox(Sandbox())
            assert (await session.status()).sandbox is None
