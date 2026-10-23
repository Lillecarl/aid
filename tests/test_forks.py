from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.daemon import Daemon
from aid.paths import ENV_STATE_DIR
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import CallPlugin, MessageEntry, Output, Usage
from plugins.forks import PREAMBLE, is_fork
from tests.conftest import py_spec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import JsonValue

    from aid.launcher import ForkserverLauncher
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))
PLUGINS = Path(__file__).parent.parent / "plugins"


@pytest.fixture
async def daemon(paths: Paths, launcher: ForkserverLauncher, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Paths]:
    """A daemon hosting the bundled plugins, forks serving before the test starts."""
    monkeypatch.setenv(ENV_STATE_DIR, str(paths.state_dir))
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher, plugin_dirs=[PLUGINS]).serve)
        async with aid.connect(paths) as client:
            with anyio.fail_after(TIMEOUT):
                while True:
                    try:
                        await client.call_plugin("forks", "probe")
                    except aid.AidError as error:
                        if error.code != "no_plugin":
                            break
                    await anyio.sleep(0.05)
        yield paths
        tg.cancel_scope.cancel()


def forks_only(tmp_path: Path) -> str:
    """A plugins path holding only the forks plugin, for the workers: its `fork` tool."""
    path = tmp_path / "plugins"
    path.mkdir()
    (path / "forks").symlink_to(PLUGINS / "forks")
    return str(path)


def test_fork_names() -> None:
    assert is_fork("parent.fork-0a1b2c")
    assert not is_fork("parent")
    assert not is_fork("parent.fork-0a1b2c.x")


async def test_a_fork_answers_its_parent_and_is_deleted(daemon: Paths, tmp_path: Path) -> None:
    """The parent reads a file, then forks: the fork's answer is the parent's turn as the fork saw it, so it
    proves the fork started from the running turn, not from history.json."""
    (tmp_path / "notes.txt").write_text("the answer is 42\n")
    spec = py_spec(tmp_path, "agents:coder")
    spec = spec.model_copy(update={"env": {**spec.env, ENV_PLUGINS_PATH: forks_only(tmp_path)}})
    plan = json.dumps([["read", {"path": "notes.txt"}], ["fork", {"instructions": "What is the answer?"}]])
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            parent = await client.create("parent", spec)
            started = (await parent.run(plan)).output
            assert isinstance(started, str) and "Fork parent.fork-" in started
            while True:
                entries = (await parent.history(limit=1000)).entries
                messages = [e for e in entries if isinstance(e.item, MessageEntry)]
                names = {info.name for info in await client.sessions()}
                woken = messages and any(isinstance(e.item, Output) for e in entries if e.seq > messages[0].seq)
                if woken and names == {"parent"}:
                    summary = await parent.summary()
                    break
                await anyio.sleep(0.05)
    [message] = [e.item for e in messages if isinstance(e.item, MessageEntry)]
    assert message.sender is not None and is_fork(message.sender)
    # The fork's turn is recorded in the parent right after its answer, and the parent's totals count it.
    after = next(e.item for e in entries if e.seq > messages[0].seq)
    assert isinstance(after, Usage) and after.agent == f"fork {message.sender}"
    assert after.requests == 1 and after.context_used is None
    assert summary.requests == 3 + 1 + 1  # The parent's turn, the fork's, the parent's wake.
    assert "the answer is 42" in message.text
    assert PREAMBLE in message.text


async def test_only_a_litelm_session_without_a_parent_may_fork(daemon: Paths, tmp_path: Path) -> None:
    async with aid.connect(daemon) as client:
        await client.create("plain", py_spec(tmp_path, "agents:steady"))
        child = py_spec(tmp_path, "agents:steady").model_copy(update={"parent": "plain"})
        await client.create("child", child)

        async def fork(caller: str | None) -> None:
            args: dict[str, JsonValue] = {"snapshot": str(tmp_path / "seed.json"), "instructions": "look"}
            await client.call(CallPlugin(plugin="forks", method="fork", args=args, caller=caller))

        with pytest.raises(aid.AidError, match="only a session"):
            await fork(None)
        with pytest.raises(aid.AidError, match="cannot fork"):
            await fork("child")
        assert {info.name for info in await client.sessions()} == {"plain", "child"}
