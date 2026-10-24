from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from pyrun.host import run_script

if TYPE_CHECKING:
    from pathlib import Path

    from pyrun import Command
    from pyrun.files import FileAccess

pytestmark = pytest.mark.anyio

TIMEOUT = 30


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_a_script_runs_with_top_level_await(tmp_path: Path) -> None:
    script = """
status = await run("sh", "-c", "echo hello; echo warn >&2")
print("got", status.text.strip())
lines = await (cmd("printf", "b\\na\\n") | cmd("sort"))
print(lines.lines)
"""
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=tmp_path, store=tmp_path / "runs")
    assert report.ok, report.error
    assert report.printed == "got hello\n['a', 'b']\n"
    assert [p.argv[0] for p in report.processes] == ["sh", "printf", "sort"]
    rendered = report.render()
    [sh] = [p for p in report.processes if p.argv[0] == "sh"]
    assert "$ sh -c 'echo hello; echo warn >&2'  (exit 0, " in rendered
    assert f", {sh.id})\n  hello" in rendered
    assert "  stderr: warn" in rendered
    assert "printed:\n  got hello" in rendered


async def test_the_error_shows_the_scripts_own_lines(tmp_path: Path) -> None:
    script = 'x = 1\nawait run("sh", "-c", "exit 3")\n'
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=tmp_path, store=tmp_path / "runs")
    assert report.error is not None
    assert 'File "<script>", line 2' in report.error
    assert 'await run("sh", "-c", "exit 3")' in report.error
    assert "Failed: sh -c 'exit 3' failed (exit 3" in report.error
    assert "pyrun/command.py" not in report.error


async def test_the_host_decides_each_command(tmp_path: Path) -> None:
    asked: list[Command] = []

    async def no_rm(command: Command) -> bool:
        asked.append(command)
        return command.argv[0] != "rm"

    script = """
a, b = await pyrun.all(cmd("echo", "one"), cmd("echo", "two"))
print(a.text + b.text, end="")
await run("rm", "-rf", "/tmp/nothing-here")
"""
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=tmp_path, store=tmp_path / "runs", policy=no_rm)
    assert sorted(c.argv[1] for c in asked if c.argv[0] == "echo") == ["one", "two"]
    assert asked[-1].argv[0] == "rm"
    assert report.printed == "one\ntwo\n"
    assert report.error is not None and "Denied" in report.error
    assert [p.argv[0] for p in report.processes] == ["echo", "echo"]


async def test_the_host_decides_file_access_outside_the_working_directory(tmp_path: Path) -> None:
    work, outside = tmp_path / "work", tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "secret").write_text("s3cret\n")
    (outside / "tree").mkdir()
    (outside / "tree" / "leaf").write_text("x")
    asked: list[FileAccess] = []

    async def reads_only(access: FileAccess) -> bool:
        asked.append(access)
        return access.access == "read" or access.path == str(outside / "tree")

    script = f"""
import os, shutil
Path("inside.txt").write_text("quiet")
os.rename("inside.txt", "moved.txt")
print(Path({str(outside / "secret")!r}).read_text(), end="")
shutil.rmtree({str(outside / "tree")!r})
try:
    os.close(os.open({str(outside / "made")!r}, os.O_WRONLY | os.O_CREAT))
except PermissionError as error:
    print("refused:", error)
"""
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=work, store=tmp_path / "runs", files=reads_only)
    assert report.error is None, report.error
    assert (work / "moved.txt").read_text() == "quiet"
    assert not (outside / "tree").exists()
    assert not (outside / "made").exists()
    assert report.printed.startswith("s3cret\nrefused: pyrun policy refused to write")
    # One question each: the read, the tree (its contents follow its approval), the refused file.
    assert [(a.access, a.path, a.event) for a in asked] == [
        ("read", str(outside / "secret"), "open"),
        ("write", str(outside / "tree"), "shutil.rmtree"),
        ("write", str(outside / "made"), "open"),
    ]


async def test_long_output_is_clipped_in_the_report(tmp_path: Path) -> None:
    script = 'await run("seq", "1000")\n'
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=tmp_path, store=tmp_path / "runs")
    rendered = report.render()
    [ran] = report.processes
    assert "  1\n" in rendered
    assert "  1000" in rendered
    assert "  500\n" not in rendered
    assert f"… 980 more lines of stdout (3893 bytes in all): read {ran.id}/stdout" in rendered
    assert ran.stdout.read_text().count("\n") == 1000


async def test_a_script_timeout_kills_what_it_started(tmp_path: Path) -> None:
    script = 'await run("sh", "-c", "echo before; sleep 60; echo after")\n'
    with anyio.fail_after(TIMEOUT):
        report = await run_script(script, cwd=tmp_path, store=tmp_path / "runs", time_limit=1)
    assert report.error is not None
    assert "timed out after 1s" in report.error
    assert "before" in report.error
