"""In-process plugins speak the plugin protocol over inproc ZMQ, through the same grant checks."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.client import local_plugin_endpoints, register_plugin
from aid.daemon import Daemon
from aid.plugins import INPROC_PLUGIN_EVENTS, INPROC_PLUGINS, Grant
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
async def _serving(paths: Paths, launcher: ForkserverLauncher) -> AsyncGenerator[Daemon]:
    async with anyio.create_task_group() as tg:
        daemon = Daemon(paths, launcher)
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
