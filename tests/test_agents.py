from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.agents import ENV_AGENTS_PATH, agents_path, discover
from aid.protocol import AgentInfo
from aid.spec import PydanticAISpec

if TYPE_CHECKING:
    from aid.paths import Paths

AGENT_DIR = Path(__file__).parent / "agent_dir"
TIMEOUT = 30


def test_discover(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    catalog = discover([AGENT_DIR, AGENT_DIR / "missing"])
    assert sorted(catalog.agents) == ["custom-name", "shouter"]
    assert catalog.agents["shouter"].description() == "Repeats the prompt in capitals."
    assert catalog.agents["custom-name"].description() == ""
    assert len(catalog.problems) == 1
    assert "broken" in catalog.problems[0]
    assert "ModuleNotFoundError" in catalog.problems[0]


def test_agents_path() -> None:
    assert agents_path({ENV_AGENTS_PATH: "/a::/b"}) == [Path("/a"), Path("/b")]
    assert agents_path({"XDG_CONFIG_HOME": "/cfg"}) == [Path("/cfg/aid/agents")]


def test_spec_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAISpec(cwd="/")
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAISpec(cwd="/", agent="a", target="m:a")


@pytest.mark.anyio
async def test_daemon_lists_and_runs_agents(daemon: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The catalog runs in a subprocess of this process; workers take the path from the spec.
    monkeypatch.setenv(ENV_AGENTS_PATH, str(AGENT_DIR))
    spec = PydanticAISpec(cwd=str(tmp_path), agent="shouter", env={ENV_AGENTS_PATH: str(AGENT_DIR)})
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
