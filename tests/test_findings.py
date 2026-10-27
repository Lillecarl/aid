from __future__ import annotations

import json
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from pydantic import ValidationError

import aid
from aid.daemon import Daemon
from aid.paths import ENV_STATE_DIR
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import AidError, QueryResult
from aid.spec import NativeSpec, PermissionMode
from plugins.findings.store import FINDINGS_FILE, Report, check, dedup_key
from tests.conftest import py_spec
from tests.test_spawn_plugin import SCRIPTED

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))
PLUGINS = Path(__file__).parent.parent / "plugins"


def report(**fields: object) -> Report:
    return Report.model_validate(
        {"path": "app.py", "category": "bug", "severity": "low", "title": "T", "detail": "D", **fields}
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "lib").mkdir(parents=True)
    (root / "app.py").write_text("import os\n\nPASSWORD = 'hunter2'\nprint(os.environ)\n")
    (tmp_path / "secret").write_text("outside\n")
    return root


def test_a_report_names_a_place_inside_the_directory(repo: Path) -> None:
    assert check(repo, report(line_start=3, evidence="PASSWORD =  'hunter2'")) == (str(repo.resolve()), "app.py")
    assert check(repo, report(path="lib")) == (str(repo.resolve()), "lib")
    assert check(repo, report(path="./lib/../app.py"))[1] == "app.py"
    assert check(repo, report(path=str(repo / "app.py")))[1] == "app.py"


@pytest.mark.parametrize(
    ("fields", "code"),
    [
        ({"path": "../secret"}, "outside"),
        ({"path": "/etc/passwd"}, "outside"),
        ({"path": "missing.py"}, "no_such_file"),
        ({"line_start": 5}, "bad_args"),
        ({"evidence": "PASSWORD = 'swordfish'"}, "bad_args"),
        ({"path": "lib", "line_start": 1}, "bad_args"),
    ],
)
def test_a_report_the_file_does_not_bear_out_is_refused(repo: Path, fields: dict[str, object], code: str) -> None:
    with pytest.raises(AidError) as raised:
        check(repo, report(**fields))
    assert raised.value.code == code


def test_a_report_is_typed() -> None:
    for fields in ({"category": "style"}, {"severity": "urgent"}, {"line_end": 2}, {"line_start": 3, "line_end": 2}):
        with pytest.raises(ValidationError):
            report(**fields)


def test_duplicates_share_a_key_however_the_title_is_cased() -> None:
    assert dedup_key("/r", report(title="Hard-coded password!")) == dedup_key("/r", report(title="hard coded PASSWORD"))
    assert dedup_key("/r", report()) != dedup_key("/r", report(line_start=1))


@asynccontextmanager
async def serving(paths: Paths, launcher: ForkserverLauncher) -> AsyncGenerator[aid.Client]:
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[PLUGINS]).serve)
        async with aid.connect(paths) as client:
            with anyio.fail_after(TIMEOUT):
                while True:
                    try:
                        await client.call_plugin("findings", "probe")
                    except aid.AidError as error:
                        if error.code != "no_plugin":
                            break
                    await anyio.sleep(0.05)
            yield client
        tg.cancel_scope.cancel()


def auditor_spec(tmp_path: Path, repo: Path) -> NativeSpec:
    """An instructions spec on a read-only ceiling plus `report_finding`, as a campaign gives one."""
    path = tmp_path / "plugins"
    (path / "scripted").mkdir(parents=True)
    (path / "scripted" / "__init__.py").write_text(SCRIPTED)
    (path / "findings").symlink_to(PLUGINS / "findings")
    spec = py_spec(repo, "agents:coder", PermissionMode.DENY)
    return NativeSpec.model_validate(
        {
            **spec.model_dump(),
            "target": None,
            "instructions": "Audit.",
            "env": {ENV_PLUGINS_PATH: str(path)},
            "model": "scripted/planned",
            "tools": ["ls", "read", "outline", "report_finding"],
        }
    )


def calls(*steps: tuple[str, dict[str, JsonValue]]) -> str:
    return json.dumps([list(step) for step in steps])


@pytest.mark.anyio
async def test_a_session_on_a_read_only_ceiling_records_findings(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    finding: dict[str, JsonValue] = {
        "path": "app.py",
        "category": "secret",
        "severity": "high",
        "title": "Hard-coded password",
        "detail": "A password sits in the source.",
        "line_start": 3,
        "evidence": "PASSWORD = 'hunter2'",
    }
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            session = await client.create("auditor", auditor_spec(tmp_path, repo))
            offered = {tool.name for tool in (await session.status()).tools}
            assert offered == {"ls", "read", "outline", "report_finding"}
            output = (
                await session.run(
                    calls(
                        ("report_finding", finding),
                        ("report_finding", {**finding, "title": "hard-coded PASSWORD"}),
                        ("report_finding", {**finding, "path": "../secret"}),
                    )
                )
            ).output
            assert isinstance(output, str)
            assert "Recorded finding 1." in output
            assert "Recorded finding 2; finding 1 has the same place and title." in output
            assert "outside" in output
            answer = await client.call_plugin(
                "findings", "query", sql="SELECT id, repo, path, severity, session, model FROM findings ORDER BY id"
            )
            rows = QueryResult.model_validate(answer).rows
            where = str(await anyio.Path(repo).resolve())
            assert rows == [
                [1, where, "app.py", "high", "auditor", "scripted/planned"],
                [2, where, "app.py", "high", "auditor", "scripted/planned"],
            ]
            database = paths.state_dir / "plugins" / "findings" / FINDINGS_FILE
            assert stat.S_IMODE(database.stat().st_mode) == 0o600
            with pytest.raises(AidError, match="session reports"):
                await client.call_plugin("findings", "report", **finding)
