from __future__ import annotations

import contextlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import Daemon
from aid.paths import ENV_STATE_DIR
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import AidError, GetSpec, PermissionRequest, QueryResult
from aid.spec import AgentSpecAdapter, NativeSpec, PermissionMode
from plugins.reverse import ENV_TOOLS, InvestigationConfig, lead_name, lead_spec
from plugins.reverse.probes import ENV_SSH
from plugins.reverse.store import Evidence, Kind, Recorded, ReverseStore, Settlement, Status, artifact
from tests.conftest import TESTS
from tests.models import HOST
from tests.test_sandbox import needs_bwrap

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "60"))
PLUGINS = Path(__file__).parent.parent / "plugins"

SCRIPTED = """\
import agents
import aid


class Scripted(aid.ModelProvider):
    async def transport(self, model: str) -> "_Model":
        return _Model(model, agents.investigator.model)


class _Model:
    def __init__(self, name, model) -> None:
        self.model_name = f"scripted/{name}"
        self._model = model

    def host(self, settings) -> str:
        return "model.test"

    def stream(self, messages, info):
        return self._model.stream(messages, info)

    async def request(self, messages, info):
        return await self._model.request(messages, info)

    async def aclose(self) -> None:
        pass
"""


def config(tmp_path: Path, workspace: Path, **fields: JsonValue) -> dict[str, JsonValue]:
    plugins = tmp_path / "worker-plugins"
    if not plugins.exists():
        (plugins / "scripted").mkdir(parents=True)
        (plugins / "scripted" / "__init__.py").write_text(SCRIPTED)
        for name in ("reverse", "supervision"):
            (plugins / name).symlink_to(PLUGINS / name)
    return {
        "name": "fans",
        "workspace": str(workspace),
        "goal": "How fast does the fan spin?",
        "model": "scripted/investigator",
        "private": [HOST],
        "env": {ENV_PLUGINS_PATH: str(plugins)},
        "python_path": [str(TESTS)],
        "supervise": False,
        **fields,
    }


@asynccontextmanager
async def serving(paths: Paths, launcher: ForkserverLauncher) -> AsyncGenerator[aid.Client]:
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[PLUGINS]).serve)
        async with aid.connect(paths) as client:
            with anyio.fail_after(TIMEOUT):
                while True:
                    try:
                        await client.call_plugin("reverse", "status")
                        break
                    except aid.AidError as error:
                        if error.code != "no_plugin":
                            raise
                    await anyio.sleep(0.05)
            yield client
        tg.cancel_scope.cancel()


def test_the_lead_writes_only_its_workspace_and_asks_before_touching_a_machine(tmp_path: Path) -> None:
    spec = lead_spec(InvestigationConfig.model_validate(config(tmp_path, tmp_path, supervise=True)))
    assert spec.sandbox is not None and not spec.sandbox.network
    assert spec.tool_permissions["remote_command"] == PermissionMode.ASK
    assert spec.tool_permissions["shell"] == PermissionMode.ALLOW
    assert spec.offers("ghidra_decompile") and spec.offers("machine_read") and not spec.offers("remove")
    assert (spec.parent, spec.supervisor, spec.tags) == ("fans", "fans", {"investigation": "fans"})


async def test_evidence_keeps_kinds_apart_and_settles_by_kind(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "dsdt.dsl").write_text("Method (_FST)\n")
    store = await ReverseStore.open(tmp_path / "reverse.db")
    try:
        await store.create("fans", "{}")
        path, sha = artifact(tmp_path / "work", "dsdt.dsl")
        who = Recorded("fans", "fans.lead", "m/x", path, sha)
        seen = await store.record(Evidence(kind=Kind.OBSERVATION, claim="a _FST method", artifact=path), who)
        plain = Recorded("fans", "fans.lead", "m/x", None, None)
        guess = await store.record(Evidence(kind=Kind.HYPOTHESIS, claim="_FST reads the fan", supports=[seen]), plain)
        with pytest.raises(ValueError, match="supports"):
            Evidence(kind=Kind.INFERENCE, claim="unfounded")
        with pytest.raises(AidError, match="no evidence 99"):
            await store.record(Evidence(kind=Kind.INFERENCE, claim="x", supports=[99]), who)
        with pytest.raises(AidError, match="settles as retracted"):
            await store.settle("fans", Settlement(evidence=seen, status=Status.CONFIRMED, note="yes"), "fans.lead")
        with pytest.raises(AidError, match="no evidence"):
            await store.settle("other", Settlement(evidence=guess, status=Status.REFUTED, note="no"), "x")
        await store.settle(
            "fans", Settlement(evidence=guess, status=Status.CONFIRMED, note="read it", answer=seen), "p"
        )
        listing = await store.listing("fans", None)
        assert await store.counts("fans") == {"hypothesis": 1, "observation": 1}
    finally:
        await store.aclose()
    assert sha is not None and len(sha) == 64
    assert listing.splitlines() == [
        "#1 observation a _FST method (dsdt.dsl)",
        "#2 hypothesis [confirmed] _FST reads the fan from #1",
    ]


def test_an_artifact_is_a_file_in_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "outside").write_text("x")
    with pytest.raises(AidError):
        artifact(tmp_path / "work", "../outside")
    with pytest.raises(AidError, match="no file"):
        artifact(tmp_path / "work", ".")


FAKE_SSH = """#!/bin/sh
# ssh OPTIONS... DEST -- COMMAND: the command runs here, as ssh would run it there.
while [ "$1" != "--" ]; do shift; done
shift
exec sh -c "$*"
"""


@needs_bwrap
async def test_probes_read_freely_and_a_remote_command_waits_for_a_person(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ssh = tmp_path / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    steps: list[JsonValue] = [
        ["machine_read", {"machine": "spark", "path": "/proc/version"}],
        ["machine_read", {"machine": "spark", "path": "/proc/../etc/shadow"}],
        ["machine_probe", {"machine": "other", "probe": "identity"}],
        ["remote_command", {"machine": "spark", "command": "echo written", "why": "test a write"}],
    ]
    fields: dict[str, JsonValue] = {"machines": {"spark": "root@spark"}}
    settings = config(tmp_path, workspace, **fields)
    env = settings["env"]
    assert isinstance(env, dict)
    settings["env"] = {**env, "AID_TEST_STEPS": json.dumps(steps), ENV_SSH: str(ssh)}
    requests: list[PermissionRequest] = []
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("reverse", "start", config=settings)
            session = client.session(lead_name("fans"))
            async with contextlib.aclosing(session.follow_history()) as entries:
                async for entry in entries:
                    if isinstance(entry.item, PermissionRequest):
                        requests.append(entry.item)
                        [allow] = [o for o in entry.item.options if o.kind == "allow_once"]
                        await session.answer(entry.item.request_id, allow.option_id)
                    if entry.item.type == "output":
                        break
            history = (await session.history(limit=100)).entries
    output = str(history[-1].item.output)  # type: ignore[union-attr]
    assert output.startswith("[read_proc_version on spark: exit 0, saved to machines/spark/read_proc_version-")
    assert "Linux version" in output
    assert "is no plain path under /sys or /proc" in output
    assert "no machine 'other' in this investigation (machines: spark)" in output
    assert "[command on spark: exit 0, saved to machines/spark/command-" in output and "\nwritten\n" in output
    [request] = requests
    assert request.tool_name == "remote_command"
    assert request.input == {"machine": "spark", "command": "echo written", "why": "test a write"}
    saved = sorted(p.name.split("-")[0] for p in (workspace / "machines" / "spark").iterdir())
    assert saved == ["command", "read_proc_version"]


needs_ghidra = pytest.mark.skipif(not os.environ.get(ENV_TOOLS), reason=f"{ENV_TOOLS} is unset: no Ghidra")


@needs_bwrap
@needs_ghidra
async def test_the_lead_decompiles_a_raw_image_in_its_sandbox(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = "fw.bin"
    steps: list[JsonValue] = [
        # mov r0, #42; bx lr: an embedded controller's whole firmware, as far as this test goes.
        ["shell", {"command": rf"printf '\x2a\x00\xa0\xe3\x1e\xff\x2f\xe1' > {image}"}],
        ["ghidra_import", {"path": image, "language": "ARM:LE:32:v7", "base_address": "0x8000000"}],
        ["ghidra_functions", {"path": image}],
        ["ghidra_decompile", {"path": image, "function": "0x8000004"}],
        ["ghidra_disassemble", {"path": image, "address": "0x8000000", "count": 2}],
        ["ghidra_decompile", {"path": "../outside", "function": "entry"}],
    ]
    settings = config(tmp_path, workspace)
    env = settings["env"]
    assert isinstance(env, dict)
    settings["env"] = {**env, "AID_TEST_STEPS": json.dumps(steps)}
    with anyio.fail_after(max(TIMEOUT, 300)):
        async with serving(paths, launcher) as client:
            await client.call_plugin("reverse", "start", config=settings)
            session = client.session(lead_name("fans"))
            async with contextlib.aclosing(session.follow_history()) as entries:
                async for entry in entries:
                    if entry.item.type == "output":
                        break
            history = (await session.history(limit=100)).entries
    output = str(history[-1].item.output)  # type: ignore[union-attr]
    assert '"lowest_address": "08000000"' in output
    assert "1 functions; rows 0-1\n08000000 entry (8 bytes)" in output
    assert "// entry at 08000000\n" in output and "return 0x2a;" in output
    assert "08000000  2a00a0e3          mov r0,#0x2a\n08000004  1eff2fe1          bx lr" in output
    assert "../outside is no file in the workspace" in output
    # The sandbox let Ghidra write its project and settings in the workspace, and nowhere it could see besides.
    assert (workspace / "ghidra" / "project" / "aid.gpr").is_file()


@needs_bwrap
async def test_an_investigation_records_what_its_lead_finds(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("reverse", "start", config=config(tmp_path, workspace))
            session = client.session(lead_name("fans"))
            async with contextlib.aclosing(session.follow_history()) as entries:
                async for entry in entries:
                    if entry.item.type == "output":
                        break
            history = (await session.history(limit=100)).entries
            spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session=lead_name("fans"))))
            status = await client.call_plugin("reverse", "status", name="fans")
            rows = await client.call_plugin("reverse", "query", sql="SELECT kind, artifact, sha256 FROM evidence")
            with pytest.raises(AidError, match="exist"):
                await client.call_plugin("reverse", "start", config=config(tmp_path, workspace))
    output = str(history[-1].item.output)  # type: ignore[union-attr]
    assert "fan: 2400 rpm" in output
    assert "#1 observation the fan spins at 2400 rpm (probe.txt @ line 1)" in output
    assert "#2 hypothesis the controller sets the speed from #1" in output
    assert isinstance(spec, NativeSpec) and spec.private is not None
    assert status == [
        {
            "name": "fans",
            "workspace": str(workspace),
            "model": "scripted/investigator",
            "lead": "fans.lead",
            "working": False,
            "evidence": {"hypothesis": 1, "observation": 1},
        }
    ]
    [seen, guess] = QueryResult.model_validate(rows).rows
    assert seen[:2] == ["observation", "probe.txt"] and isinstance(seen[2], str)
    assert guess == ["hypothesis", None, None]
