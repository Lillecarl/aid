from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.discovery import discover
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import AgentInfo
from aid.spec import NativeSpec

if TYPE_CHECKING:
    from aid.paths import Paths

AGENT_DIR = Path(__file__).parent / "agent_dir"
TIMEOUT = 30


def test_discover() -> None:
    catalog = discover([AGENT_DIR, AGENT_DIR / "missing"])
    assert sorted(catalog.agents) == ["custom-name", "shouter"]
    assert catalog.agents["shouter"].description() == "Repeats the prompt in capitals."
    assert catalog.agents["custom-name"].description() == ""
    assert len(catalog.problems) == 1
    assert "broken" in catalog.problems[0]
    assert "ModuleNotFoundError" in catalog.problems[0]


def test_spec_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        NativeSpec(cwd="/")
    with pytest.raises(ValueError, match="exactly one"):
        NativeSpec(cwd="/", agent="a", target="m:a")


@pytest.mark.anyio
async def test_daemon_lists_and_runs_agents(daemon: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The catalog runs in a subprocess of this process; workers take the path from the spec.
    monkeypatch.setenv(ENV_PLUGINS_PATH, str(AGENT_DIR))
    spec = NativeSpec(cwd=str(tmp_path), agent="shouter", env={ENV_PLUGINS_PATH: str(AGENT_DIR)})
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            catalog = await client.agents()
            session = await client.create("loud", spec)
            result = await session.run("hello")
            with pytest.raises(aid.AidError) as missing:
                await client.create("nope", spec.model_copy(update={"agent": "no-such-agent"}))
    assert catalog.agents == [
        AgentInfo(name="custom-name", description="", module="echoing"),
        AgentInfo(name="shouter", description="Repeats the prompt in capitals.", module="echoing"),
    ]
    assert [p for p in catalog.problems if "broken" in p]
    assert result.output == "HELLO"
    assert missing.value.code == "start_failed"
    assert "no-such-agent" in missing.value.message
    assert "shouter" in missing.value.message
