"""The bundled supervisor agent: what it may do, and its read-only reference tools."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

import aid
from aid.agent import ModelRetry, Tool
from aid.discovery import discover
from plugins.supervision.agent import MODEL
from plugins.supervision.reference import ENV_TLDR_PAGES, man_page, tldr_page

pytestmark = pytest.mark.anyio

PLUGINS = Path(__file__).parent.parent / "plugins"
FAKE_MAN = """#!/bin/sh
# Records its argv, then prints a page with groff's terminal bold, as man -P cat does.
printf '%s\\n' "$@" > "$0.argv"
printf 'N\\bNA\\bAM\\bME\\bE\\n    rm - remove files\\n\\nOPTIONS\\n  -f, --force\\n    never prompt\\n  -r\\n    recursive\\n'
"""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_man(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "man"
    script.write_text(FAKE_MAN)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/run/current-system/sw/bin:/usr/bin:/bin")
    return script


def test_the_supervisor_is_discovered_with_typed_tools_only() -> None:
    found = discover([PLUGINS], only=["supervision"])
    assert found.problems == []
    agent = found.agents["supervisor"]().build()
    assert agent.model == MODEL
    tools = [*agent.tools, *(tool for toolset in agent.toolsets for tool in toolset.tools)]
    names = {tool.name if isinstance(tool, Tool) else Tool(tool).name for tool in tools}
    assert names == {
        "man_page",
        "tldr_page",
        "supervised_history",
        "supervised_entry",
        "supervised_search",
        "goals",
        "goal_add",
        "goal_done",
        "goal_drop",
        "answer_permission",
        "forward_permission",
        "interrupt",
    }
    assert aid.coding_tools not in agent.toolsets


async def test_man_page_runs_man_with_a_fixed_argv_and_plain_text(fake_man: Path) -> None:
    page = await man_page("rm", section="1")
    assert (await anyio.Path(f"{fake_man}.argv").read_text()).split() == ["-P", "cat", "1", "rm"]
    assert page.startswith("NAME\n")
    assert "\b" not in page
    assert await man_page("rm", search="--force") == "…\n\nOPTIONS\n  -f, --force\n    never prompt\n  -r"


async def test_names_that_are_not_commands_are_refused(fake_man: Path) -> None:
    for name in ("-H", "rm; reboot", "../rm", ""):
        with pytest.raises(ModelRetry):
            await man_page(name)
    with pytest.raises(ModelRetry):
        await man_page("rm", section="-K")
    assert not await anyio.Path(f"{fake_man}.argv").exists()


async def test_tldr_page_reads_common_then_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "linux").mkdir()
    (tmp_path / "linux" / "apt.md").write_text("# apt\n")
    monkeypatch.setenv(ENV_TLDR_PAGES, str(tmp_path))
    assert await tldr_page("apt") == "# apt\n"
    assert await tldr_page("nothing") == "No tldr page for nothing."
    with pytest.raises(ModelRetry):
        await tldr_page("../apt")
    monkeypatch.delenv(ENV_TLDR_PAGES)
    assert await tldr_page("apt") == "tldr pages are not installed here."
