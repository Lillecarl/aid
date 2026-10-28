from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import apsw
import pytest

import aid
from aid.daemon import Daemon
from aid.paths import ENV_STATE_DIR
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import CallPlugin, GetSpec, PromptEntry, QueryResult, ToolCall
from aid.spec import AgentSpecAdapter, NativeSpec
from plugins.campaign import CampaignConfig, session_names, subsystem_session, survey_spec
from plugins.campaign.store import CAMPAIGN_FILE
from tests.conftest import TESTS, py_spec
from tests.models import HOST
from tests.test_campaign_units import git_repo

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
from aid.messages import TextPart
from aid.model import StreamDelta

MODELS = {
    "auditor": agents.auditor,
    "lazy": agents.lazy_auditor,
    "flaky": agents.flaky_auditor,
    "hanging": agents.hanging_auditor,
    "echo": agents.echo,
}


CUT = []


class Scripted(aid.ModelProvider):
    async def transport(self, model: str) -> "_Model":
        if model == "cutoff":
            return _Cutoff(model, agents.auditor.model)
        return _Model(model, MODELS[model].model)


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


# The auditor, its first unit answer lost to the output limit, as GLM spent max_tokens on reasoning: nothing grew,
# so the engine stops at max_tokens instead of asking to continue.
class _Cutoff(_Model):
    async def stream(self, messages, info):
        async for delta in self._model.stream(messages, info):
            if isinstance(delta.part, TextPart) and ": read" in delta.part.content and not CUT:
                CUT.append(1)
                yield StreamDelta(truncated=True)
                return
            yield delta
"""


def test_session_names_fit_and_never_collide() -> None:
    names = session_names("c" * 24, ["My Repo", "my_repo", "x" * 80, "x" * 80 + "y"])
    assert names[:2] == ["c" * 24 + ".my-repo", "c" * 24 + ".my-repo-2"]
    assert all(len(name) <= 64 for name in names) and len(set(names)) == 4


def test_the_survey_session_cannot_report(tmp_path: Path) -> None:
    config = CampaignConfig.model_validate({"name": "c", "root": "/r", "model": "m/x", "hosts": ["h"]})
    spec = survey_spec(config, tmp_path, "c.repo")
    assert (spec.tools, spec.parent) == (["ls", "read", "outline", "search"], "c.repo")


@asynccontextmanager
async def serving(paths: Paths, launcher: ForkserverLauncher) -> AsyncGenerator[aid.Client]:
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[PLUGINS]).serve)
        async with aid.connect(paths) as client:
            with anyio.fail_after(TIMEOUT):
                while True:
                    try:
                        await client.call_plugin("campaign", "status")
                        break
                    except aid.AidError as error:
                        if error.code != "no_plugin":
                            raise
                    await anyio.sleep(0.05)
            yield client
        tg.cancel_scope.cancel()


def config(tmp_path: Path, root: Path, **fields: JsonValue) -> dict[str, JsonValue]:
    plugins = tmp_path / "worker-plugins"
    if not plugins.exists():
        (plugins / "scripted").mkdir(parents=True)
        (plugins / "scripted" / "__init__.py").write_text(SCRIPTED)
        for name in ("findings", "supervision"):
            (plugins / name).symlink_to(PLUGINS / name)
    return {
        "name": "audit",
        "root": str(root),
        "model": "scripted/auditor",
        "hosts": [HOST],
        "env": {ENV_PLUGINS_PATH: str(plugins)},
        "python_path": [str(TESTS)],
        "supervise": False,
        "backoff": 0,
        "limits": {"unit_files": 2},
        **fields,
    }


async def repos(root: Path) -> None:
    long = "".join(f"line {n}\n" for n in range(4500)).encode()
    await git_repo(root / "alpha", {"a.py": b"PASSWORD = 'hunter2'\n", "b.py": b"x = 1\n", "c.py": b"y\n"})
    await git_repo(root / "beta", {"long.py": long, "blob.bin": b"\0\1"})
    (root / "notes").mkdir()


async def status(client: aid.Client, name: str) -> dict[str, JsonValue]:
    answer = await client.call_plugin("campaign", "status", name=name)
    assert isinstance(answer, list)
    campaign = answer[0]
    assert isinstance(campaign, dict)
    return campaign


async def settled(client: aid.Client, name: str) -> list[dict[str, JsonValue]]:
    """The campaign's repositories once it runs no more: polled, as the plugin publishes no event. The stored state
    leaves `running` only after every repository's rows are written."""
    campaign = await status(client, name)
    while campaign["state"] == "running":
        await anyio.sleep(0.1)
        campaign = await status(client, name)
    repos = campaign["repos"]
    assert isinstance(repos, list)
    return [repo for repo in repos if isinstance(repo, dict)]


async def read_paths(client: aid.Client, session: str) -> list[str]:
    entries = (await client.session(session).history(limit=1000)).entries
    return [
        str(e.item.input["path"])
        for e in entries
        if isinstance(e.item, ToolCall) and e.item.tool == "read" and isinstance(e.item.input, dict)
    ]


async def test_a_campaign_reads_every_line_of_every_tracked_file(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("campaign", "start", config=config(tmp_path, root))
            repos_done = await settled(client, "audit")
            findings = await client.call_plugin("findings", "query", sql="SELECT session, path, category FROM findings")
            alpha = await read_paths(client, "audit.alpha")
            briefs = [
                e
                for e in (await client.session("audit.alpha").history(limit=1000)).entries
                if isinstance(e.item, PromptEntry) and e.item.text.startswith("Look over this repository")
            ]
            assert briefs == []
            beta = await read_paths(client, "audit.beta")
            spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session="audit.alpha")))
            listed = {info.name: info for info in await client.sessions()}
    assert [(r["path"], r["state"], r["units"], r["skipped"]) for r in repos_done] == [
        ("alpha", "done", {"done": 2}, 0),
        ("beta", "done", {"done": 1}, 1),
    ]
    assert sorted(alpha) == ["a.py", "b.py", "c.py"]
    assert beta == ["long.py", "long.py", "long.py"]
    assert QueryResult.model_validate(findings).rows == [["audit.alpha", "a.py", "secret"]]
    assert isinstance(spec, NativeSpec)
    assert (spec.parent, spec.fresh_turns, spec.instruction_providers, spec.private is not None) == (
        "audit",
        True,
        [],
        True,
    )
    assert spec.tools == ["ls", "read", "outline", "search", "report_finding"]
    # The brief ran in a survey session with no tool to report with, deleted once the brief was in.
    assert not any(name.endswith(".survey") for name in listed)
    assert all(listed[name].private for name in ("audit", "audit.alpha", "audit.beta"))


async def test_a_unit_is_prompted_again_with_what_was_left_unread(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin(
                "campaign", "start", config=config(tmp_path, root, model="scripted/lazy", repos=["alpha"])
            )
            [alpha] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
    assert (alpha["state"], alpha["units"]) == ("done", {"done": 2})
    # Unit one: b.py, then a.py when prompted again alone; unit two: c.py alone, which the lazy model reads.
    assert reads == ["b.py", "a.py", "c.py"]


async def test_a_turn_that_fails_after_its_reads_is_audited_again(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            fields: dict[str, JsonValue] = {"model": "scripted/flaky", "repos": ["alpha"], "limits": {"unit_files": 3}}
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, **fields))
            [alpha] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
            findings = await client.call_plugin("findings", "query", sql="SELECT path FROM findings")
    # The unit's reads and report, an answer that fails, then all three read, reported and answered again.
    assert (alpha["state"], alpha["units"]) == ("done", {"done": 1})
    assert reads.count("a.py") == 2
    assert QueryResult.model_validate(findings).rows == [["a.py"], ["a.py"]]


async def test_an_answer_cut_off_at_the_output_limit_is_audited_again(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            fields: dict[str, JsonValue] = {"model": "scripted/cutoff", "repos": ["alpha"], "limits": {"unit_files": 3}}
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, **fields))
            [alpha] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
    assert (alpha["state"], alpha["units"]) == ("done", {"done": 1})
    assert reads.count("a.py") == 2


async def test_attempts_run_out_on_a_file_never_read(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin(
                "campaign", "start", config=config(tmp_path, root, model="scripted/echo", repos=["alpha"], attempts=2)
            )
            [alpha] = await settled(client, "audit")
    assert (alpha["state"], alpha["units"]) == ("done", {"incomplete": 2})


async def test_a_campaign_off_its_hosts_stops_with_the_reason(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, hosts=["vllm.example"]))
            await settled(client, "audit")
            campaign = await status(client, "audit")
    assert campaign["state"] == "paused"
    assert "private_host" in str(campaign["error"])


async def test_a_restart_resumes_without_reading_again(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, repos=["alpha"]))
            [alpha] = await settled(client, "audit")
        assert alpha["state"] == "done"
        # As a daemon stopped between a unit's answer and its row leaves it: running, reads and answer recorded.
        db = apsw.Connection(str(paths.plugin_dir("campaign") / CAMPAIGN_FILE))
        with db:
            db.execute("UPDATE units SET state = 'running' WHERE idx = 0")
            db.execute("UPDATE repos SET state = 'briefed'")
            db.execute("UPDATE campaigns SET state = 'running'")
        db.close()
        async with serving(paths, launcher) as client:
            [again] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
    assert again["state"] == "done"
    assert sorted(reads) == ["a.py", "b.py", "c.py"]


async def test_the_root_session_supervises_each_repository(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            fields: dict[str, JsonValue] = {"repos": ["alpha"], "supervise": True}
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, **fields))
            [alpha] = await settled(client, "audit")
            infos = {info.name: info for info in await client.sessions()}
            boss = AgentSpecAdapter.validate_python(await client.call(GetSpec(session="audit")))
    assert alpha["state"] == "done"
    assert infos["audit.alpha"].supervisor == "audit"
    assert isinstance(boss, NativeSpec) and boss.agent == "supervisor" and boss.private is not None
    assert boss.tools is not None and not {"man_page", "tldr_page"} & set(boss.tools)


async def test_a_unit_stopped_before_its_answer_is_audited_again(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            fields: dict[str, JsonValue] = {
                "model": "scripted/hanging",
                "repos": ["alpha"],
                "limits": {"unit_files": 3},
                "env": {ENV_PLUGINS_PATH: str(tmp_path / "worker-plugins"), "AID_HANG_GATE": str(tmp_path / "hung")},
            }
            config(tmp_path, root)  # The worker plugins directory, before its path goes into `fields`.
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, **fields))
            # The daemon stops while the model hangs over its answer: the reads are recorded, the answer is not.
            while not (tmp_path / "hung").exists():  # noqa: ASYNC110 -- the worker drops a file; no event crosses to here
                await anyio.sleep(0.1)
            assert sorted(await read_paths(client, "audit.alpha")) == ["a.py", "b.py", "c.py"]
        async with serving(paths, launcher) as client:
            [alpha] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
    assert alpha["units"] == {"done": 1}
    assert reads.count("a.py") == 2


async def test_a_recreated_session_is_audited_afresh(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, repos=["alpha"]))
            await settled(client, "audit")
            # Its attempt cut short, as the stored rows say below, in a session then made again: the attempt's seq
            # belongs to a history that is gone, and earns nothing.
            await client.session("audit.alpha").delete()
        db = apsw.Connection(str(paths.plugin_dir("campaign") / CAMPAIGN_FILE))
        with db:
            db.execute("UPDATE units SET state = 'running', covered = NULL")
            db.execute("UPDATE repos SET state = 'briefed'")
            db.execute("UPDATE campaigns SET state = 'running'")
        db.close()
        async with serving(paths, launcher) as client:
            [alpha] = await settled(client, "audit")
            reads = await read_paths(client, "audit.alpha")
    assert alpha["units"] == {"done": 2}
    assert sorted(reads) == ["a.py", "b.py", "c.py"]


async def test_a_campaign_never_adopts_a_session_it_did_not_make(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    await repos(root)
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            await client.create("audit", py_spec(tmp_path, "agents:echo"))
            await client.call_plugin("campaign", "start", config=config(tmp_path, root))
            await settled(client, "audit")
            campaign = await status(client, "audit")
    assert campaign["state"] == "paused"
    assert "did not make" in str(campaign["error"])


async def test_a_big_repository_is_split_into_subsystems_each_with_a_session(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    root = tmp_path / "tree"
    files = {f"{d}/f{n}.py": b"x = 1\n" for d in ("api", "auth", "deploy") for n in range(4)}
    await git_repo(root / "gamma", files | {"README.md": b"# gamma\n"})
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            fields: dict[str, JsonValue] = {"repos": ["gamma"], "split_units": 2, "limits": {"unit_files": 2}}
            await client.call_plugin("campaign", "start", config=config(tmp_path, root, **fields))
            [gamma] = await settled(client, "audit")
            sessions = {info.name: info for info in await client.sessions()}
            reads: list[str] = []
            for name in sessions:
                if name.startswith(subsystem_session("audit", "gamma", 0)[:-1]):
                    reads += await read_paths(client, name)
            prompts = [
                e.item.text
                for e in (await client.session("audit.gamma").history(limit=1000)).entries
                if isinstance(e.item, PromptEntry)
            ]
    assert gamma["state"] == "done"
    parts = [subsystem_session("audit", "gamma", idx) for idx in range(3)]
    assert all(sessions[name].parent == "audit.gamma" for name in parts)
    # README.md matched no subsystem: it is the "other" subsystem's, in a session of its own.
    assert sorted(reads) == sorted([*files, "README.md"])
    assert any("cross subsystems" in prompt for prompt in prompts)


async def test_only_the_operator_runs_campaigns(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher) as client:
            with pytest.raises(aid.AidError, match="operator"):
                await client.call(CallPlugin(plugin="campaign", method="status", caller="someone"))
            with pytest.raises(aid.AidError, match="bad_args"):
                await client.call_plugin("campaign", "start", config={"name": "Bad Name"})
    assert CampaignConfig.model_validate(config(tmp_path, tmp_path)).concurrency == 2
