"""Instruction providers: discovery, compose, and the file-based default. Daemon-free."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

import aid
from aid.agent import Agent
from aid.agents import Instruction, InstructionsContext, InstructionsProvider
from aid.backends.native import NativeBackend, open_native
from aid.backends.permissions import PermissionWaits
from aid.coding import CODING, Coding, compact
from aid.discovery import Catalog, discover
from aid.instructions import compose
from aid.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import AgentCatalog, Started
from aid.spec import NativeSpec, PermissionMode
from tests.conftest import TESTS
from tests.models import TestModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from typing import Any

    from aid.protocol import SessionEvent

pytestmark = pytest.mark.anyio

PROVIDER_DIR = Path(__file__).parent / "provider_dir"
EXAMPLE_FILE = Path(__file__).parents[1] / "examples" / "file_instructions" / "__init__.py"


def file_instructions() -> Any:
    """The `examples/file_instructions` plugin, loaded from its file."""
    name = "aid_example_file_instructions"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXAMPLE_FILE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Ok(InstructionsProvider):
    """Two instructions."""

    def load(self, context: InstructionsContext) -> list[Instruction]:
        return [Instruction(name="first", text="First."), Instruction(name="second", text="Second.")]


class Failing(InstructionsProvider):
    """Always fails."""

    def load(self, context: InstructionsContext) -> list[Instruction]:
        raise RuntimeError("boom")


class Other(InstructionsProvider):
    """One more."""

    def load(self, context: InstructionsContext) -> list[Instruction]:
        return [Instruction(name="third", text="Third.")]


def catalog() -> Catalog:
    return Catalog(providers={"ok": Ok, "failing": Failing, "other": Other})


def test_naming_defaults_to_snake_case() -> None:
    assert Ok.provider_name() == "ok"
    assert Failing.provider_name() == "failing"
    assert Ok.description() == "Two instructions."


def test_discovers_providers_and_lists_problems() -> None:
    found = discover([PROVIDER_DIR, PROVIDER_DIR / "missing"])
    assert sorted(found.providers) == ["briefing", "custom-provider", "dup"]
    assert found.providers["briefing"].description() == "A short project brief."
    assert found.providers["custom-provider"].description() == ""
    assert found.providers["dup"].__module__ == "dup_a"
    assert len(found.problems) == 2
    assert any("broken_provider" in problem and "ModuleNotFoundError" in problem for problem in found.problems)
    assert any("dup_a" in problem and "dup_b" in problem for problem in found.problems)


def test_describe_rows_validate_as_agent_catalog() -> None:
    described = AgentCatalog.model_validate_json(discover([PROVIDER_DIR]).describe().model_dump_json())
    assert [(provider.name, provider.module) for provider in described.providers] == [
        ("briefing", "briefing"),
        ("custom-provider", "briefing"),
        ("dup", "dup_a"),
    ]


def test_compose_concatenates_with_headers(tmp_path: Path) -> None:
    assert compose(tmp_path, catalog()) == "# first\n\nFirst.\n\n# second\n\nSecond.\n\n# third\n\nThird."


def test_compose_select_and_skip(tmp_path: Path) -> None:
    assert compose(tmp_path, catalog(), select=["other", "ok"]) == (
        "# third\n\nThird.\n\n# first\n\nFirst.\n\n# second\n\nSecond."
    )
    assert compose(tmp_path, catalog(), skip=frozenset({"second"})) == "# first\n\nFirst.\n\n# third\n\nThird."
    assert compose(tmp_path, catalog(), select=["no-such-provider"]) == ""
    assert compose(tmp_path, catalog(), select=[]) == ""


def test_compose_skips_a_failing_provider(tmp_path: Path) -> None:
    assert "Third." in compose(tmp_path, catalog())


def test_default_provider_reads_global_and_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    (tmp_path / "instructions.md").write_text("Be kind.")
    project = tmp_path / "project"
    cwd = project / "sub"
    cwd.mkdir(parents=True)
    (project / "AGENTS.md").write_text("Use tabs.")
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "instructions.md"))
    loaded = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert [(item.name, item.text) for item in loaded] == [("global", "Be kind."), ("agents", "Use tabs.")]


def test_default_provider_prefers_the_nearest_agents_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    (tmp_path / "AGENTS.md").write_text("Root.")
    cwd = tmp_path / "sub"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("Local.")
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "missing.md"))
    [agents] = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert (agents.name, agents.text) == ("agents", "Local.")


def test_default_provider_skips_absent_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "missing.md"))
    assert module.FileInstructions().load(aid.InstructionsContext(cwd=tmp_path, session_name="demo")) == []


def test_spec_instruction_fields() -> None:
    assert NativeSpec(cwd="/", agent="briefing").instruction_providers is None
    assert NativeSpec(cwd="/", agent="briefing").skip_instructions == []
    spec = NativeSpec(
        cwd="/", agent="briefing", instruction_providers=["file_instructions"], skip_instructions=["skill:x"]
    )
    assert spec.instruction_providers == ["file_instructions"]
    assert spec.skip_instructions == ["skill:x"]


def test_compose_names_the_session(tmp_path: Path) -> None:
    class Who(InstructionsProvider):
        def load(self, context: InstructionsContext) -> list[Instruction]:
            return [Instruction(name="who", text=f"session {context.session_name}")]

    assert compose(tmp_path, {"who": Who}, session_name="s1") == "# who\n\nsession s1"
    assert compose(tmp_path, {"who": Who}) == "# who\n\nsession "


def write_wired_provider(agents: Path) -> None:
    """A provider that reads its brief from disk at every load, so a rewrite shows at the next compose."""
    (agents / "prov").mkdir(parents=True, exist_ok=True)
    (agents / "prov" / "__init__.py").write_text(
        "from __future__ import annotations\n\nfrom pathlib import Path\n\nimport aid\n\n\n"
        "class Prov(aid.InstructionsProvider):\n"
        '    """Test provider."""\n\n'
        "    def load(self, context):\n"
        "        brief = (Path(__file__).parent / 'brief.txt').read_text()\n"
        "        return [\n"
        "            aid.Instruction(name='brief', text=brief),\n"
        "            aid.Instruction(name='who', text='session ' + context.session_name),\n"
        "        ]\n"
    )
    (agents / "prov" / "brief.txt").write_text("Ship it.")


def emit_to(events: list[SessionEvent]) -> Any:
    async def emit(event: SessionEvent) -> None:
        events.append(event)

    return emit


@asynccontextmanager
async def wired_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str | None, **spec_kwargs: Any
) -> AsyncGenerator[NativeBackend]:
    """The test provider on the plugins path, and an open backend wired to it."""
    write_wired_provider(tmp_path / "agents")
    monkeypatch.setenv(ENV_PLUGINS_PATH, str(tmp_path / "agents"))
    monkeypatch.syspath_prepend(str(TESTS))  # pyright: ignore[reportUnknownMemberType] -- pytest leaves the path untyped
    spec = NativeSpec(cwd=str(tmp_path), target=target, permission=PermissionMode.DENY, **spec_kwargs)
    state = anyio.Path(tmp_path / "s1")
    await state.mkdir()
    async with open_native(spec, state) as backend:
        yield backend


async def open_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str | None, **spec_kwargs: Any) -> str:
    """A session wired to the test provider, one turn run: its stored history as text."""
    async with wired_backend(tmp_path, monkeypatch, target, **spec_kwargs) as backend:
        await backend.prompt("hi", emit_to([]))
    return (tmp_path / "s1" / "history.json").read_text()


def contents(stored: str) -> list[str]:
    """Every text content of a stored history, unescaped."""
    return [
        part["content"]
        for message in json.loads(stored)
        for part in message["parts"]
        if isinstance(part.get("content"), str)
    ]


def system_prompt(stored: str) -> str:
    return contents(stored)[0]


async def test_wiring_prepends_composed_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = await open_wired(tmp_path, monkeypatch, "agents:echo")
    assert system_prompt(stored).startswith("# brief\n\nShip it.\n\n# who\n\nsession s1\n\n# environment")


async def test_wiring_names_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it an agent that `cd`ed away cannot find its way back."""
    stored = await open_wired(tmp_path, monkeypatch, "agents:echo", instruction_providers=[])
    cwd = await anyio.Path(tmp_path).resolve()
    assert system_prompt(stored).startswith(f"# environment\n\n- Working directory: {cwd}.")
    assert "- aid session: s1" in system_prompt(stored)


async def test_wiring_skip_drops_named_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = await open_wired(tmp_path, monkeypatch, "agents:echo", skip_instructions=["brief"])
    assert system_prompt(stored).startswith("# who\n\nsession s1\n\n# environment")


def write_quiet_agent(agents: Path) -> None:
    """An agent definition that composes no instruction providers."""
    (agents / "quiet").mkdir(parents=True, exist_ok=True)
    (agents / "quiet" / "__init__.py").write_text(
        "import aid\nfrom agents import echo\n\n\n"
        "class Quiet(aid.AgentDefinition):\n"
        '    """Composes nothing."""\n\n'
        "    instruction_providers = ()\n\n"
        "    def build(self):\n"
        "        return echo\n"
    )


async def test_an_agent_declares_its_instruction_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_quiet_agent(tmp_path / "agents")
    stored = await open_wired(tmp_path, monkeypatch, None, agent="quiet")
    assert system_prompt(stored).startswith("# environment")
    assert not any("# brief" in content for content in contents(stored))


async def test_a_spec_overrides_the_agents_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_quiet_agent(tmp_path / "agents")
    stored = await open_wired(tmp_path, monkeypatch, None, agent="quiet", instruction_providers=["prov"])
    assert system_prompt(stored).startswith("# brief\n\nShip it.")


async def test_fresh_turns_start_from_the_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = tmp_path / "agents"
    (agents / "fresh").mkdir(parents=True)
    (agents / "fresh" / "__init__.py").write_text(
        "import aid\nfrom agents import echo\n\n\n"
        "class Fresh(aid.AgentDefinition):\n"
        '    """Forgets each turn."""\n\n'
        "    fresh_turns = True\n\n"
        "    def build(self):\n"
        "        return echo\n"
    )
    async with wired_backend(tmp_path, monkeypatch, None, agent="fresh") as backend:
        first = await backend.prompt("a", emit_to([]))
        second = await backend.prompt("b", emit_to([]))
    assert (first.output, second.output) == ("turn 1: echo a", "turn 1: echo b")
    stored = (tmp_path / "s1" / "history.json").read_text()
    assert system_prompt(stored).startswith("# brief")
    assert not any("echo a" in content for content in contents(stored))


async def test_wiring_unknown_provider_starts_bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = await open_wired(tmp_path, monkeypatch, "agents:echo", instruction_providers=["nope"])
    assert system_prompt(stored).startswith("# environment")
    assert not any("# brief" in content for content in contents(stored))
    assert any("echo hi" in content for content in contents(stored))


async def test_compact_recomposes_instructions_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with wired_backend(tmp_path, monkeypatch, "agents:steady") as backend:
        await backend.prompt("hi", emit_to([]))
        await backend.prompt("again", emit_to([]))
        (tmp_path / "agents" / "prov" / "brief.txt").write_text("Ship it, twice.")
        await backend.compact("the work")
    stored = (tmp_path / "s1" / "history.json").read_text()
    assert system_prompt(stored).startswith("# brief\n\nShip it, twice.\n\n# who\n\nsession s1")
    assert not any("Ship it.\n" in content for content in contents(stored))


def backend_with(tmp_path: Path, history: list[Any], refresh: Any, output: str = "kept") -> tuple[Coding, Any]:
    agent = Agent(TestModel(custom_output_text=output))
    coding = Coding(
        cwd=tmp_path,
        store=tmp_path / "runs",
        outputs=tmp_path / "outputs",
        mode=PermissionMode.ALLOW,
        timeout=60,
        waits=PermissionWaits(),
    )
    backend = NativeBackend(
        agent,
        anyio.Path(tmp_path / "history.json"),
        history,
        [],
        Started(pid=1, agent="t", model="m"),
        coding,
        refresh_instructions=refresh,
    )
    return coding, backend


def aged_history() -> list[Any]:
    return [
        ModelRequest(parts=[SystemPromptPart(content="# brief\n\nStale.")]),
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart(content="did one")]),
        ModelRequest(parts=[UserPromptPart(content="second")]),
        ModelResponse(parts=[TextPart(content="did two")]),
    ]


async def compacted(tmp_path: Path, history: list[Any], refresh: Any) -> list[Any]:
    coding, _ = backend_with(tmp_path, history, refresh)
    token = CODING.set(coding)
    try:
        await compact("the billing work")
    finally:
        CODING.reset(token)
    return json.loads((tmp_path / "history.json").read_bytes())


async def test_compact_replaces_stale_instructions(tmp_path: Path) -> None:
    stored = await compacted(tmp_path, aged_history(), lambda: "# brief\n\nFresh.")
    [head] = stored[0]["parts"]
    assert (head["part_kind"], head["content"]) == ("system-prompt", "# brief\n\nFresh.")
    assert stored[1]["parts"][0]["content"].startswith("[Summary of earlier work]\nkept")


async def test_compact_drops_instructions_when_providers_go_quiet(tmp_path: Path) -> None:
    stored = await compacted(tmp_path, aged_history(), lambda: "")
    assert stored[0]["parts"][0]["content"].startswith("[Summary of earlier work]\nkept")


async def test_compact_keeps_the_head_without_a_refresher(tmp_path: Path) -> None:
    stored = await compacted(tmp_path, aged_history(), None)
    [head] = stored[0]["parts"]
    assert (head["part_kind"], head["content"]) == ("system-prompt", "# brief\n\nStale.")
    assert stored[1]["parts"][0]["content"].startswith("[Summary of earlier work]\nkept")


def test_default_provider_falls_back_to_claude_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repository written for Claude Code keeps its rules in CLAUDE.md; AGENTS.md wins in the same directory."""
    module = file_instructions()
    (tmp_path / "AGENTS.md").write_text("Root.")
    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text("Claude rules.")
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "missing.md"))
    [agents] = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert (agents.name, agents.text) == ("agents", "Claude rules.")
    (cwd / "AGENTS.md").write_text("Agents rules.")
    [agents] = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert agents.text == "Agents rules."
