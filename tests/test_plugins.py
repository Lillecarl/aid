from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.client import register_plugin
from aid.plugins import Grant, PluginSpec
from aid.protocol import AidError, MessageEntry
from aid.zap import Keypair
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


async def test_a_plugin_may_do_what_it_is_granted_and_nothing_more(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, daemon, "reader", frozenset({Grant.READ}))
            async with aid.connect(daemon, plugin="reader") as reader:
                listed = [s.name for s in await reader.sessions()]
                status = await reader.session("echo").status()
                with pytest.raises(AidError, match="may not send 'prompt'") as prompt:
                    await reader.session("echo").run("hi")
                with pytest.raises(AidError, match="may not send 'add_plugin'"):
                    await register_plugin(reader, daemon, "more", frozenset(Grant))
                await control.remove_plugin("reader")
                # The connection outlives the plugin; its requests do not.
                with pytest.raises(AidError, match="may not send 'list'"):
                    await reader.sessions()
            plugins = await control.plugins()
    assert listed == ["echo"]
    assert status.running
    assert prompt.value.code == "forbidden"
    assert plugins == []


async def test_a_plugins_message_names_the_plugin(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.create("echo", py_spec(tmp_path, "agents:echo"))
            await register_plugin(control, daemon, "notes", frozenset({Grant.MESSAGE}))
            async with aid.connect(daemon, plugin="notes") as notes:
                await notes.send_message("echo", "hello", sender="echo")  # A claim the daemon replaces.
            history = await control.session("echo").history()
    [message] = [e.item for e in history.entries if isinstance(e.item, MessageEntry)]
    assert message.sender == "plugin:notes"


async def test_a_plugin_granted_read_follows_what_is_published(daemon: Paths, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await register_plugin(control, daemon, "watcher", frozenset({Grant.READ}))
            async with aid.connect(daemon, plugin="watcher") as watcher:
                seen: list[list[str]] = []
                async for listed in watcher.follow_sessions():
                    seen.append([s.name for s in listed])
                    if len(seen) == 1:
                        await control.create("echo", py_spec(tmp_path, "agents:echo"))
                    elif "echo" in seen[-1]:
                        break
    assert seen[0] == []
    assert seen[-1] == ["echo"]


async def test_a_key_another_plugin_holds_is_refused(daemon: Paths) -> None:
    key = Keypair.new().public
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as control:
            await control.add_plugin(PluginSpec(name="one", public_key=key, grants=frozenset()))
            with pytest.raises(AidError, match="holds that key"):
                await control.add_plugin(PluginSpec(name="two", public_key=key, grants=frozenset()))
            # Its own name may take it again, with other grants.
            await control.add_plugin(PluginSpec(name="one", public_key=key, grants=frozenset({Grant.READ})))
            plugins = await control.plugins()
    assert [(p.name, p.grants) for p in plugins] == [("one", frozenset({Grant.READ}))]
