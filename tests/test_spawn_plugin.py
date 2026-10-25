from __future__ import annotations

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
from aid.protocol import CallPlugin, GetSpec, MessageEntry, Usage
from aid.spec import AgentSpecAdapter, NativeSpec, PermissionMode
from tests.conftest import py_spec

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))
PLUGINS = Path(__file__).parent.parent / "plugins"

SCRIPTED = '''\
import agents
import aid


class Scripted(aid.ModelProvider):
    """`scripted/planned` runs the test coder's plan; `scripted/echo` echoes the prompt."""

    async def transport(self, model: str) -> "_Model":
        return _Model(model, agents.coder.model if model == "planned" else agents.echo.model)


class _Model:
    def __init__(self, name, model) -> None:
        self.model_name = f"scripted/{name}"
        self._model = model

    def stream(self, messages, info):
        return self._model.stream(messages, info)

    async def request(self, messages, info):
        return await self._model.request(messages, info)

    async def aclose(self) -> None:
        pass
'''


@asynccontextmanager
async def serving(paths: Paths, launcher: ForkserverLauncher, plugin_dirs: list[Path]) -> AsyncGenerator[Paths]:
    """A daemon on `paths`; with the bundled plugins, spawn serving before it yields."""
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=plugin_dirs).serve)
        if plugin_dirs:
            async with aid.connect(paths) as client:
                with anyio.fail_after(TIMEOUT):
                    while True:
                        try:
                            await client.call_plugin("spawn", "probe")
                        except aid.AidError as error:
                            if error.code != "no_plugin":
                                break
                        await anyio.sleep(0.05)
        yield paths
        tg.cancel_scope.cancel()


@pytest.fixture
async def daemon(paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Paths]:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    async with serving(paths, launcher, [PLUGINS]) as served:
        yield served


def parent_spec(tmp_path: Path, **fields: object) -> NativeSpec:
    """The test coder on `scripted/planned`, whose workers see the spawn plugin and the scripted models."""
    path = tmp_path / "plugins"
    (path / "scripted").mkdir(parents=True, exist_ok=True)
    (path / "scripted" / "__init__.py").write_text(SCRIPTED)
    if not (path / "spawn").exists():
        (path / "spawn").symlink_to(PLUGINS / "spawn")
    spec = py_spec(tmp_path, "agents:coder", PermissionMode.ALLOW)
    return NativeSpec.model_validate(
        {
            **spec.model_dump(),
            "env": {**spec.env, ENV_PLUGINS_PATH: str(path)},
            "model": "scripted/planned",
            "spawn_models": ["scripted/echo"],
            **fields,
        }
    )


def spawning(**args: JsonValue) -> str:
    return json.dumps([["spawn", {"label": "helper", "instructions": "Echo.", "model": "scripted/echo", **args}]])


async def relayed(session: aid.Session, count: int, sender: str = "parent.helper") -> list[MessageEntry]:
    while True:
        entries = (await session.history(limit=1000)).entries
        messages = [e.item for e in entries if isinstance(e.item, MessageEntry) and e.item.sender == sender]
        if len(messages) >= count and not (await session.status()).busy:
            return messages
        await anyio.sleep(0.05)


async def test_a_child_works_on_a_cut_ceiling_and_its_turns_reach_the_parent(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            parent = await client.create("parent", parent_spec(tmp_path))
            answer = (await parent.run(spawning(tools=["ls|read", "send_message"], task="hello"))).output
            assert isinstance(answer, str)
            started = json.loads(answer)
            assert started == {"session": "parent.helper", "tools": ["ls", "read", "send_message"]}
            child = client.session("parent.helper")
            spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session=child.name)))
            assert isinstance(spec, NativeSpec)
            assert (spec.parent, spec.model, spec.tools) == ("parent", "scripted/echo", ["ls", "read", "send_message"])
            [message] = await relayed(parent, 1)
            assert message.text.startswith("turn 1: echo ") and "hello" in message.text
            entries = (await parent.history(limit=1000)).entries
            assert any(isinstance(e.item, Usage) and e.item.agent == "spawned parent.helper" for e in entries)
            await parent.delete()
            assert await client.sessions() == []


async def test_a_child_offered_spawn_spawns_its_own(daemon: Paths, tmp_path: Path) -> None:
    """The child's catalog has the plugin's tool through the inherited plugins path; its call names it the caller, so
    the grandchild's turns reach the child, and the child's reach the parent."""
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            parent = await client.create("parent", parent_spec(tmp_path))
            await parent.run(spawning(tools=["spawn"], model="scripted/planned"))
            child = client.session("parent.helper")
            grand: dict[str, JsonValue] = {
                "label": "deep",
                "instructions": "Echo.",
                "model": "scripted/echo",
                "tools": [],
                "task": "hi",
            }
            await child.run(json.dumps([["spawn", grand]]))
            spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session="parent.helper.deep")))
            assert isinstance(spec, NativeSpec)
            assert (spec.parent, spec.model) == ("parent.helper", "scripted/echo")
            assert "hi" in (await relayed(child, 1, "parent.helper.deep"))[0].text
            assert "parent.helper.deep" in (await relayed(parent, 1))[0].text
            await parent.delete()
            assert await client.sessions() == []


async def test_spawn_refuses_what_the_parent_may_not_give(daemon: Paths, tmp_path: Path) -> None:
    async with aid.connect(daemon) as client:
        parent = await client.create("parent", parent_spec(tmp_path))
        await parent.run("[]")

        async def spawn(caller: str | None, **args: JsonValue) -> None:
            given: dict[str, JsonValue] = {"label": "helper", "instructions": "Echo.", "tools": [], **args}
            await client.call(CallPlugin(plugin="spawn", method="spawn", args=given, caller=caller))

        with pytest.raises(aid.AidError, match="only a session"):
            await spawn(None)
        with pytest.raises(aid.AidError, match="match none"):
            await spawn("parent", tools=["no_such_tool"])
        with pytest.raises(aid.AidError, match="may not run on"):
            await spawn("parent", model="vllm/glm-5.3-flash")
        with pytest.raises(aid.AidError, match="label"):
            await spawn("parent", label="Not A Label")
        assert {info.name for info in await client.sessions()} == {"parent"}


async def test_only_the_parent_dismisses_its_child(daemon: Paths, tmp_path: Path) -> None:
    async with aid.connect(daemon) as client:
        parent = await client.create("parent", parent_spec(tmp_path))
        await parent.run(spawning(tools=[]))
        await client.create("other", py_spec(tmp_path, "agents:steady"))

        async def dismiss(caller: str) -> None:
            args: dict[str, JsonValue] = {"session": "parent.helper"}
            await client.call(CallPlugin(plugin="spawn", method="dismiss", args=args, caller=caller))

        with pytest.raises(aid.AidError, match="no session you spawned"):
            await dismiss("other")
        await dismiss("parent")
        assert {info.name for info in await client.sessions()} == {"parent", "other"}


async def test_a_restart_relays_a_turn_that_ended_meanwhile_once(
    paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    with anyio.fail_after(TIMEOUT):
        async with serving(paths, launcher, [PLUGINS]), aid.connect(paths) as client:
            parent = await client.create("parent", parent_spec(tmp_path))
            await parent.run(spawning(tools=[], task="first"))
            await relayed(parent, 1)
        async with serving(paths, launcher, []), aid.connect(paths) as client:
            await client.session("parent.helper").run("second")
        async with serving(paths, launcher, [PLUGINS]), aid.connect(paths) as client:
            messages = await relayed(client.session("parent"), 2)
            await anyio.sleep(0.5)
            assert len(await relayed(client.session("parent"), 2)) == 2
    assert "first" in messages[0].text
    assert messages[1].text == "turn 2: echo second"
