"""In-process plugins speak the plugin protocol over inproc ZMQ, through the same grant checks."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq.asyncio

import aid
from aid import plugin_host
from aid.client import local_plugin_endpoints, register_plugin
from aid.daemon import Daemon
from aid.plugins import INPROC_PLUGIN_EVENTS, INPROC_PLUGINS, Grant, PluginCatalog, PluginSpec
from aid.protocol import AidError, MessageEntry
from tests.conftest import py_spec

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

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


async def test_hosted_plugins_start_on_registration_and_stop_on_removal(
    paths: Paths, launcher: ForkserverLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plugin_host, "RESTART_DELAY", 0.05)
    plugindir = tmp_path / "plugindir"
    plugindir.mkdir()
    (plugindir / "stubbed.py").write_text(STUB_PLUGIN)
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

    catalog = PluginCatalog(plugins={"full": Full, "short": Short, "stranger": Stranger})
    registered = {
        "full": PluginSpec(name="full", public_key="x" * 40, grants=frozenset({Grant.READ})),
        "short": PluginSpec(name="short", public_key="y" * 40, grants=frozenset({Grant.READ})),
    }
    ready = plugin_host.enabled(catalog, registered)
    assert [name for name, _ in ready] == ["full"]
    assert isinstance(ready[0][1], Full)
