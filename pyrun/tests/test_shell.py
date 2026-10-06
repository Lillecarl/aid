from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest

import pyrun

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

TIMEOUT = 20


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "runs"


async def test_cd_and_env_persist_between_runs(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            (tmp_path / "sub").mkdir()
            cd = await shell.run("cd sub && pwd")
            still = await shell.run("pwd")
            export = await shell.run("export AID_TEST_MARK=here")
            marked = await shell.run('echo "in $(basename $(pwd)):$AID_TEST_MARK"')
    finally:
        await shell.aclose()
    assert (cd.output.strip(), cd.code) == (str(tmp_path / "sub"), 0)
    assert still.output.strip() == str(tmp_path / "sub")
    assert (export.output, export.code) == ("", 0)
    assert marked.output.strip() == "in sub:here"


async def test_exit_codes_and_merged_streams(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            missing = await shell.run("ls does-not-exist")
            ordered = await shell.run("echo out; echo err >&2; echo out2")
            stdin_empty = await shell.run("cat")
            empty = await shell.run("  ")
    finally:
        await shell.aclose()
    assert missing.code != 0 and "does-not-exist" in missing.output
    assert ordered.output.splitlines() == ["out", "err", "out2"]
    # stdin is empty: cat meets EOF at once instead of eating later commands.
    assert (stdin_empty.output, stdin_empty.code) == ("", 0)
    assert (empty.output, empty.code) == ("", 0)


async def test_timeout_kills_and_the_next_run_starts_fresh(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            before = await shell.run("echo $$")
            await shell.run("cd /tmp")
            hung = await shell.run("sleep 30", time_limit=0.5)
            after = await shell.run("echo $$; pwd")
    finally:
        await shell.aclose()
    assert (hung.code, hung.timed_out, hung.reset) == (None, True, True)
    assert before.output.strip() != after.output.splitlines()[0].strip()
    assert after.output.splitlines()[1].strip() == str(tmp_path)


async def test_a_broken_command_eats_its_mark_and_times_out(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            broken = await shell.run("echo 'unclosed", time_limit=1.0)
            alive = await shell.run("echo recovered")
    finally:
        await shell.aclose()
    assert (broken.timed_out, broken.reset) == (True, True)
    assert (alive.output.strip(), alive.code, alive.reset) == ("recovered", 0, True)


async def test_exit_ends_the_shell_and_the_next_run_respawns(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            ended = await shell.run("exit 3")
            assert shell.pid is None
            revived = await shell.run("echo back")
    finally:
        await shell.aclose()
    assert (ended.code, ended.reset) == (None, True)
    assert (revived.output.strip(), revived.code) == ("back", 0)


async def test_concurrent_runs_serialize(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):

            async def slow() -> pyrun.ShellResult:
                return await shell.run("sleep 0.5; echo slow")

            async def quick() -> pyrun.ShellResult:
                return await shell.run("echo quick")

            async with anyio.create_task_group() as tg:
                slow_result: pyrun.ShellResult | None = None
                quick_result: pyrun.ShellResult | None = None

                async def run_slow() -> None:
                    nonlocal slow_result
                    slow_result = await slow()

                async def run_quick() -> None:
                    nonlocal quick_result
                    quick_result = await quick()

                tg.start_soon(run_slow)
                tg.start_soon(run_quick)
    finally:
        await shell.aclose()
    assert slow_result is not None and quick_result is not None
    assert (slow_result.output.strip(), slow_result.code) == ("slow", 0)
    assert (quick_result.output.strip(), quick_result.code) == ("quick", 0)


async def test_each_incarnation_is_recorded(store: Path, tmp_path: Path) -> None:
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            await shell.run("echo one")
            await shell.restart()
            await shell.run("echo two")
    finally:
        await shell.aclose()
    first = json.loads((store / shell.id / "1" / "command.json").read_text())
    second = json.loads((store / shell.id / "2" / "command.json").read_text())
    assert first["argv"] == ["bash", "--noprofile", "--norc"]
    assert first["cwd"] == str(tmp_path)
    assert second["argv"] == first["argv"]
    assert (store / shell.id / "1" / "stdout").read_text() != ""


async def test_the_session_survives_tasks_and_scopes(store: Path, tmp_path: Path) -> None:
    """The load-bearing property: the held handle is data, not scopes, so timeout scopes open after the
    spawn and other tasks can run and close around it."""
    (tmp_path / "sub").mkdir()
    shell = pyrun.ShellSession(cwd=tmp_path, store=store)
    try:
        with anyio.fail_after(TIMEOUT):
            await shell.run("cd sub")
        results: list[pyrun.ShellResult] = []

        async def pwd() -> None:
            results.append(await shell.run("pwd"))

        with anyio.fail_after(TIMEOUT):
            async with anyio.create_task_group() as tg:
                tg.start_soon(pwd)
        with anyio.fail_after(TIMEOUT):
            await shell.run("export AID_HELD=1")
    finally:
        await shell.aclose()
    (only,) = results
    assert only.output.strip() == str(tmp_path / "sub")
