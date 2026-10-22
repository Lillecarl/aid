"""In-process plugins speak the plugin protocol over inproc ZMQ, through the same grant checks."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq.asyncio

import aid
from aid import daemon as daemon_module
from aid import plugin_host
from aid.client import local_plugin_endpoints, register_plugin
from aid.daemon import Daemon
from aid.discovery import Catalog
from aid.plugins import INPROC_PLUGIN_EVENTS, INPROC_PLUGINS, Grant, PluginSpec
from aid.protocol import AidError, CallPlugin, CallResult, GetSpec, MessageEntry, ServeCalls, SetAttention
from tests.conftest import py_spec

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


@asynccontextmanager
async def _serving(
    paths: Paths, launcher: ForkserverLauncher, plugin_dirs: list[Path] | None = None
) -> AsyncGenerator[Daemon]:
    async with anyio.create_task_group() as tg:
        daemon = Daemon(paths, launcher, plugin_dirs=[] if plugin_dirs is None else plugin_dirs)
        await tg.start(daemon.serve)
        yield daemon
        tg.cancel_scope.cancel()


def test_local_endpoints_are_the_inproc_pair_without_keys() -> None:
    endpoints = local_plugin_endpoints()
    assert (endpoints.control, endpoints.events, endpoints.curve) == (
        INPROC_PLUGINS,
        INPROC_PLUGIN_EVENTS,
        None,
    )


async def test_a_local_plugin_passes_the_same_grant_checks(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, paths, "reader", frozenset({Grant.READ}))
            # The daemon names the task: identity by construction, no keys.
            async with daemon.plugin_client("reader") as reader:
                listed = [s.name for s in await reader.sessions()]
                with pytest.raises(AidError, match="may not send 'prompt'"):
                    await reader.session("echo").run("hi")
                with pytest.raises(AidError, match="may not send 'add_plugin'"):
                    await register_plugin(reader, paths, "more", frozenset(Grant))
                await control.remove_plugin("reader")
                with pytest.raises(AidError, match="may not send 'list'"):
                    await reader.sessions()
    assert listed == ["echo"]


async def test_a_local_plugins_message_names_the_plugin(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, paths, "notes", frozenset({Grant.MESSAGE}))
            async with daemon.plugin_client("notes") as notes:
                await notes.send_message("echo", "hello", sender="echo")  # A claim the daemon replaces.
            history = await control.session("echo").history()
    [message] = [e.item for e in history.entries if isinstance(e.item, MessageEntry)]
    assert message.sender == "plugin:notes"


async def test_a_local_plugin_granted_read_follows_what_is_published(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await register_plugin(control, paths, "watcher", frozenset({Grant.READ}))
            async with daemon.plugin_client("watcher") as watcher:
                seen: list[list[str]] = []
                async for listed in watcher.follow_sessions():
                    seen.append([s.name for s in listed])
                    if len(seen) == 1:
                        await control.create("echo", py_spec(tmp_path, "agents:echo"))
                    elif "echo" in seen[-1]:
                        break
    assert seen[0] == []
    assert seen[-1] == ["echo"]


class Answering(aid.Plugin):
    async def run(self, client: aid.Client) -> None:
        await anyio.sleep_forever()

    async def call(self, caller: str | None, method: str, args: dict[str, JsonValue]) -> JsonValue:
        match method:
            case "echo":
                return {"caller": caller, "args": args}
            case "refuse":
                raise AidError("not_yours", f"{caller} may not")
            case "crash":
                raise RuntimeError("bug")
            case "hang":
                await anyio.sleep_forever()
            case _:
                pass
        raise AidError("unknown_method", method)


@asynccontextmanager
async def _answering(daemon: Daemon, name: str) -> AsyncGenerator[aid.Client]:
    async with daemon.plugin_client(name) as client, anyio.create_task_group() as tg:
        tg.start_soon(plugin_host.run, Answering(), client)
        yield client
        tg.cancel_scope.cancel()


async def _called(control: aid.Client, plugin: str, method: str, **args: JsonValue) -> JsonValue:
    """Call once the plugin serves: its first `ServeCalls` races the test's first call."""
    while True:
        try:
            return await control.call_plugin(plugin, method, **args)
        except AidError as error:
            if error.code != "no_plugin":
                raise
            await anyio.sleep(0.01)


async def test_a_plugin_answers_calls_for_the_caller_the_daemon_names(
    paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_module, "PLUGIN_CALL_TIMEOUT", 0.2)
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await register_plugin(control, paths, "answering", frozenset())
            async with _answering(daemon, "answering"):
                assert await _called(control, "answering", "echo", n=1) == {"caller": None, "args": {"n": 1}}
                as_boss = CallPlugin(plugin="answering", method="echo", caller="boss")
                assert await control.call(as_boss) == {"caller": "boss", "args": {}}
                with pytest.raises(AidError, match="not_yours: boss may not"):
                    await control.call(CallPlugin(plugin="answering", method="refuse", caller="boss"))
                with pytest.raises(AidError, match="plugin_failed: RuntimeError: bug"):
                    await control.call_plugin("answering", "crash")
                with pytest.raises(AidError, match="plugin_timeout"):
                    await control.call_plugin("answering", "hang")
                await control.remove_plugin("answering")
                with pytest.raises(AidError, match="no_plugin"):
                    await control.call_plugin("answering", "echo")


async def test_only_the_called_plugin_answers_a_call(paths: Paths, launcher: ForkserverLauncher) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await register_plugin(control, paths, "answering", frozenset())
            await register_plugin(control, paths, "other", frozenset())
            async with _answering(daemon, "answering"), daemon.plugin_client("other") as other:
                await _called(control, "answering", "echo")
                async with anyio.create_task_group() as tg:
                    tg.start_soon(control.call_plugin, "answering", "hang")
                    await anyio.sleep(0.1)
                    [call] = daemon._calls  # pyright: ignore[reportPrivateUsage]
                    with pytest.raises(AidError, match="unknown_call"):
                        await other.call(CallResult(call=call, data="forged"))
                    with pytest.raises(AidError, match="only a plugin serves calls"):
                        await control.call(ServeCalls())
                    tg.cancel_scope.cancel()
                with pytest.raises(AidError, match="no_plugin"):
                    await control.call_plugin("other", "echo")


class Gate(aid.Plugin):
    async def run(self, client: aid.Client) -> None:
        await anyio.sleep_forever()

    async def on_message(self, to: str, sender: str | None, text: str) -> None:
        if text == "refuse me":
            raise AidError("gated", f"{sender} may not tell {to} that")


async def test_a_supervising_plugin_gates_messages_and_watches_sessions(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, paths, "gate", frozenset({Grant.SUPERVISE, Grant.MESSAGE}))
            async with daemon.plugin_client("gate") as gate, anyio.create_task_group() as tg:
                tg.start_soon(plugin_host.run, Gate(), gate)
                while not daemon._hooks.get("gate"):  # pyright: ignore[reportPrivateUsage]  # noqa: ASYNC110
                    await anyio.sleep(0.01)
                with pytest.raises(AidError, match="gated: boss may not tell echo that"):
                    await control.send_message("echo", "refuse me", sender="boss")
                await control.send_message("echo", "fine", sender="boss")
                await gate.send_message("echo", "as the boss", sender="boss")
                spec = await gate.call(GetSpec(session="echo"))
                assert isinstance(spec, dict)
                assert spec["target"] == "agents:echo"
                await gate.call(SetAttention(session="echo", text="look here"))
                assert (await control.session("echo").status()).attention == "look here"
                tg.cancel_scope.cancel()
            history = await control.session("echo").history()
    messages = [(e.item.sender, e.item.text) for e in history.entries if isinstance(e.item, MessageEntry)]
    assert messages == [("boss", "fine"), ("boss", "as the boss")]


async def test_only_supervise_hooks_messages_or_reads_specs(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path
) -> None:
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher) as daemon, aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, paths, "reader", frozenset({Grant.READ, Grant.MESSAGE}))
            async with daemon.plugin_client("reader") as reader:
                with pytest.raises(AidError, match="hooks nothing without supervise"):
                    await reader.call(ServeCalls(hooks=["message"]))
                with pytest.raises(AidError, match="may not send 'spec'"):
                    await reader.call(GetSpec(session="echo"))
                with pytest.raises(AidError, match="may not send 'attention'"):
                    await reader.call(SetAttention(session="echo", text="x"))


STUB_PLUGIN = '''"""A stub hosted plugin: pings, then falls over."""

from __future__ import annotations

import aid


class Stub(aid.Plugin):
    """A stub that pings and falls over."""

    grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

    async def run(self, client: aid.Client) -> None:
        await client.send_message("echo", "ping")
        raise RuntimeError("boom")
'''


AGENT_PLUGIN = '''"""An agent plugin, with no task: the daemon must not import it."""

import os

import aid

with open("{marker}", "a") as marker:
    marker.write(f"{{os.getpid()}}\\n")


class Agentish(aid.AgentDefinition):
    def build(self):
        raise NotImplementedError
'''


async def test_hosted_plugins_start_on_registration_and_stop_on_removal(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plugin_host, "RESTART_DELAY", 0.05)
    plugindir = tmp_path / "plugindir"
    (plugindir / "stub").mkdir(parents=True)
    (plugindir / "stub" / "__init__.py").write_text(STUB_PLUGIN)
    (plugindir / "agentish").mkdir()
    (plugindir / "agentish" / "__init__.py").write_text(AGENT_PLUGIN.format(marker=tmp_path / "imported-by"))
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher, [plugindir]), aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, paths, "stub", frozenset({Grant.READ, Grant.MESSAGE}))

            async def pings() -> int:
                history = await control.session("echo").history()
                return sum(1 for e in history.entries if isinstance(e.item, MessageEntry))

            await anyio.sleep(1)  # Falls over every RESTART_DELAY, pinging each time it starts.
            assert await pings() >= 2
            await control.remove_plugin("stub")
            before = await pings()
            await anyio.sleep(0.3)
            assert await pings() == before
            history = await control.session("echo").history()
    assert {e.item.sender for e in history.entries if isinstance(e.item, MessageEntry)} == {"plugin:stub"}
    # The catalog subprocess imported the agent plugin; the daemon, in this process, never did.
    importers = (tmp_path / "imported-by").read_text().split()
    assert importers and str(os.getpid()) not in importers


async def test_an_autostart_plugin_runs_unregistered_until_removed(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plugin_host, "RESTART_DELAY", 0.05)
    plugindir = tmp_path / "plugindir"
    (plugindir / "stub").mkdir(parents=True)
    (plugindir / "stub" / "__init__.py").write_text(STUB_PLUGIN)
    (plugindir / "stub" / "plugin.toml").write_text("autostart = true\n")
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher, [plugindir]), aid.connect(paths) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))

            async def pings() -> int:
                history = await control.session("echo").history()
                return sum(1 for e in history.entries if isinstance(e.item, MessageEntry))

            await anyio.sleep(1)  # Falls over every RESTART_DELAY, pinging each time it starts.
            assert await pings() >= 2
            [spec] = await control.plugins()
            assert (spec.name, spec.grants) == ("stub", frozenset({Grant.READ, Grant.MESSAGE}))
            await control.remove_plugin("stub")
            await register_plugin(control, paths, "other", frozenset())  # Reconciles again.
            before = await pings()
            await anyio.sleep(0.3)
            assert await pings() == before
            assert [s.name for s in await control.plugins()] == ["other"]
    assert not (paths.plugin_dir("stub")).exists()


async def test_supervise_gives_up_after_repeated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_host, "RESTART_DELAY", 0.01)
    monkeypatch.setattr(plugin_host, "MAX_FAILURES", 3)
    runs = 0

    @asynccontextmanager
    async def fake_spawn(_ctx: zmq.asyncio.Context, _name: str) -> AsyncGenerator[object]:
        yield object()

    monkeypatch.setattr(plugin_host, "spawn_plugin_client", fake_spawn)

    class Flaky(aid.Plugin):
        """Falls over at once."""

        async def run(self, client: aid.Client) -> None:
            nonlocal runs
            runs += 1
            raise RuntimeError("boom")

    ctx = zmq.asyncio.Context()
    try:
        with anyio.fail_after(TIMEOUT):
            await plugin_host.supervise(ctx, "flaky", Flaky())
    finally:
        ctx.term()
    assert runs == 3


async def test_supervise_ends_on_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def fake_spawn(_ctx: zmq.asyncio.Context, _name: str) -> AsyncGenerator[object]:
        yield object()

    monkeypatch.setattr(plugin_host, "spawn_plugin_client", fake_spawn)

    class Slow(aid.Plugin):
        """Never returns."""

        async def run(self, client: aid.Client) -> None:
            await anyio.sleep_forever()

    ctx = zmq.asyncio.Context()
    try:
        with anyio.fail_after(TIMEOUT):
            async with anyio.create_task_group() as tg:
                tg.start_soon(plugin_host.supervise, ctx, "slow", Slow())
                await anyio.sleep(0.1)
                tg.cancel_scope.cancel()
    finally:
        ctx.term()


def test_enabled_runs_registered_and_covered_only() -> None:
    class Full(aid.Plugin):
        """Covered."""

        grants = frozenset({Grant.READ})

        async def run(self, client: aid.Client) -> None: ...

    class Short(aid.Plugin):
        """Under-granted."""

        grants = frozenset({Grant.READ, Grant.MESSAGE})

        async def run(self, client: aid.Client) -> None: ...

    class Stranger(aid.Plugin):
        """Unregistered."""

        async def run(self, client: aid.Client) -> None: ...

    catalog = Catalog(plugins={"full": Full, "short": Short, "stranger": Stranger})
    registered = {
        "full": PluginSpec(name="full", public_key="x" * 40, grants=frozenset({Grant.READ})),
        "short": PluginSpec(name="short", public_key="y" * 40, grants=frozenset({Grant.READ})),
    }
    ready = plugin_host.enabled(catalog, registered)
    assert [name for name, _ in ready] == ["full"]
    assert ready[0][1] is Full


async def test_reconciliation_does_not_reconstruct_a_running_plugin(
    paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructions = 0
    started = anyio.Event()

    class Resident(aid.Plugin):
        def __init__(self) -> None:
            nonlocal constructions
            constructions += 1

        async def run(self, client: aid.Client) -> None:
            started.set()
            await anyio.sleep_forever()

    catalog = Catalog(plugins={"resident": Resident})

    async def hosted(_dirs: Sequence[Path] | None) -> Catalog:
        return catalog

    monkeypatch.setattr(plugin_host, "hosted", hosted)
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher), aid.connect(paths) as control:
            await register_plugin(control, paths, "resident", frozenset())
            await started.wait()
            await register_plugin(control, paths, "unrelated", frozenset({Grant.READ}))
            await control.remove_plugin("unrelated")
            assert constructions == 1


async def test_unrelated_registry_changes_do_not_restart_a_disabled_plugin(
    paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = 0
    failed = anyio.Event()
    monkeypatch.setattr(plugin_host, "MAX_FAILURES", 1)

    class Broken(aid.Plugin):
        async def run(self, client: aid.Client) -> None:
            nonlocal runs
            runs += 1
            failed.set()
            raise RuntimeError("broken")

    catalog = Catalog(plugins={"broken": Broken})

    async def hosted(_dirs: Sequence[Path] | None) -> Catalog:
        return catalog

    monkeypatch.setattr(plugin_host, "hosted", hosted)
    with anyio.fail_after(TIMEOUT):
        async with _serving(paths, launcher), aid.connect(paths) as control:
            await register_plugin(control, paths, "broken", frozenset())
            await failed.wait()
            await register_plugin(control, paths, "unrelated", frozenset({Grant.READ}))
            await control.remove_plugin("unrelated")
            assert runs == 1
