from __future__ import annotations

import contextlib
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
from aid.protocol import AidError, GetSpec, QueryResult
from aid.spec import AgentSpecAdapter, NativeSpec, PermissionMode
from plugins.reverse import InvestigationConfig, lead_name, lead_spec
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
