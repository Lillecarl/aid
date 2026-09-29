from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import TYPE_CHECKING

import anyio
import pytest

import pyrun
from pyrun import cmd, run, sh
from pyrun import scope as scope_module

if TYPE_CHECKING:
    from pathlib import Path

    from pyrun import Command

pytestmark = pytest.mark.anyio

TIMEOUT = 20


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def alive(pid: int) -> bool:
    try:
        state = (anyio.Path(f"/proc/{pid}/stat")._path.read_text()).split(") ", 1)[1][0]  # pyright: ignore[reportPrivateUsage]
    except OSError, IndexError:
        return False
    return state != "Z"


async def test_a_command_is_a_value_and_awaiting_runs_it(store: Path, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with pyrun.Scope(cwd=tmp_path, store=store) as scope:
            command = cmd("sh", "-c", 'printf "a\\nb\\n"; echo oops >&2')
            first, second = await command, await command
            listed = await run("printf", '{"n": 1}')
    assert str(command) == "sh -c 'printf \"a\\nb\\n\"; echo oops >&2'"
    assert (first.text, first.lines, first.code, first.stderr.text) == ("a\nb\n", ["a", "b"], 0, "oops\n")
    assert (first.id, second.id) == (f"{scope.id}.1", f"{scope.id}.2")
    assert listed.json() == {"n": 1}
    record = json.loads((store / scope.id / "1" / "command.json").read_text())
    assert (record["argv"], record["code"], record["cwd"]) == (list(command.argv), 0, str(tmp_path))
    assert (store / scope.id / "1" / "stdout").read_text() == "a\nb\n"


async def test_failure_raises_unless_asked_not_to(store: Path) -> None:
    async with pyrun.Scope(store=store):
        with pytest.raises(pyrun.Failed) as failed:
            await sh("echo broken >&2; exit 3")
        kept = await sh("exit 4", check=False)
    assert failed.value.result.code == 3
    assert "exit 3" in str(failed.value)
    assert "broken" in str(failed.value)
    assert (kept.code, kept.ok) == (4, False)


async def test_input_env_and_cwd(store: Path, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    os.environ["PYRUN_TEST_DROPPED"] = "here"
    try:
        async with pyrun.Scope(cwd=tmp_path, store=store, env={"FROM_SCOPE": "s"}):
            upper = await run("tr", "a-z", "A-Z", input="shout")
            seen = await sh(
                'echo "$FROM_SCOPE $FROM_CMD ${PYRUN_TEST_DROPPED:-gone} $(pwd)"',
                cwd="sub",
                env={"FROM_CMD": "c", "PYRUN_TEST_DROPPED": None},
            )
    finally:
        del os.environ["PYRUN_TEST_DROPPED"]
    assert upper.text == "SHOUT"
    assert seen.text == f"s c gone {tmp_path / 'sub'}\n"


async def test_pipelines_fail_when_a_stage_fails(store: Path) -> None:
    async with pyrun.Scope(store=store):
        counted = await (cmd("printf", "b\\na\\nb\\n") | cmd("sort") | cmd("uniq", "-c"))
        # yes writes forever; head's exit kills it with SIGPIPE, which is not a failure.
        headed = await (cmd("yes") | cmd("head", "-2"))
        with pytest.raises(pyrun.Failed) as failed:
            await (sh("echo x; exit 2") | cmd("cat"))
    assert [line.split() for line in counted.lines] == [["1", "a"], ["2", "b"]]
    assert headed.lines == ["y", "y"]
    assert failed.value.result.code == 2


async def test_lines_arrive_as_written(store: Path) -> None:
    seen: list[tuple[str, float]] = []
    started = time.monotonic()
    async with pyrun.Scope(store=store):
        async for line in sh("echo one; sleep 0.5; echo two").lines():
            seen.append((line, time.monotonic() - started))
    assert [line for line, _ in seen] == ["one", "two"]
    assert seen[0][1] < 0.4 < seen[1][1]


async def test_a_started_process_talks_both_ways(store: Path) -> None:
    async with pyrun.Scope(store=store), cmd("cat").start() as process:
        await process.write("ping\n")
        echoed = await process.readline()
        await process.close_stdin()
        result = await process.wait()
    assert echoed == "ping"
    assert result.ok


async def test_all_runs_together_and_the_first_failure_stops_the_rest(store: Path) -> None:
    started = time.monotonic()
    async with pyrun.Scope(store=store):
        a, b = await pyrun.all(sh("sleep 0.5; echo a"), sh("sleep 0.5; echo b"))
        together = time.monotonic() - started
        with pytest.raises(pyrun.Failed):
            await pyrun.all(sh("sleep 30"), sh("exit 1"))
    assert (a.text, b.text) == ("a\n", "b\n")
    assert together < 0.9
    assert time.monotonic() - started < 10


async def test_a_timeout_names_what_was_running(store: Path) -> None:
    async with pyrun.Scope(store=store):
        with pytest.raises(pyrun.TimedOut) as timed_out:
            await sh("echo started; sleep 30; echo never", timeout=0.5)  # A last command, or sh execs into it.
    [running] = [r for r in timed_out.value.running if r.id is not None]
    assert running.argv[:2] == ["sh", "-c"]
    assert "started" in running.last_lines
    assert "timed out after 0.5s" in str(timed_out.value)


async def test_a_scope_timeout_kills_everything(store: Path) -> None:
    with pytest.raises(pyrun.TimedOut) as timed_out:
        async with pyrun.Scope(store=store, timeout=0.5):
            await pyrun.all(sh("sleep 30"), sh("sleep 31"))
    assert sorted(r.argv for r in timed_out.value.running if r.id) == [["sleep", "30"], ["sleep", "31"]]


async def test_the_policy_sees_each_command(store: Path) -> None:
    asked: list[Command] = []

    async def only_echo(command: Command) -> bool:
        asked.append(command)
        return command.argv[0] == "echo"

    async with pyrun.Scope(store=store, policy=only_echo):
        await run("echo", "fine")
        with pytest.raises(pyrun.Denied):
            await run("rm", "-rf", "nothing")
        with pytest.raises(pyrun.Denied):
            await (cmd("echo", "x") | cmd("rm", "y"))
    assert [c.argv[0] for c in asked] == ["echo", "rm", "echo", "rm"]


async def test_descendants_that_leave_their_group_die_with_the_scope(store: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    daemonize = f"setsid sh -c 'echo $$ > {pid_file}; exec sleep 100' </dev/null >/dev/null 2>&1 &"
    async with pyrun.Scope(store=store):
        async with sh(f"{daemonize} sleep 100").start():
            with anyio.fail_after(TIMEOUT):
                while not pid_file.exists() or not pid_file.read_text().strip():  # noqa: ASYNC110 -- a file, no event
                    await anyio.sleep(0.05)
            escaped = int(pid_file.read_text())
            assert alive(escaped)
            assert os.getpgid(escaped) != os.getpgid(0)
        with anyio.fail_after(TIMEOUT):
            while alive(escaped):  # noqa: ASYNC110 -- as above
                await anyio.sleep(0.05)


async def test_output_past_the_memory_limit_is_read_back(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scope_module, "MEMORY_LIMIT", 10)
    async with pyrun.Scope(store=store):
        result = await run(sys.executable, "-c", "print('x' * 100)")
    assert result.stdout.clipped
    assert len(result.stdout.head) == 10
    assert result.text == "x" * 100 + "\n"


async def test_a_signal_is_not_an_exit_code(store: Path) -> None:
    async with pyrun.Scope(store=store):
        result = await sh("kill -TERM $$", check=False)
    assert (result.code, result.signal) == (None, signal.SIGTERM)
