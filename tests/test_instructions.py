"""Instruction providers: discovery, compose, and the file-based default. Daemon-free."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import aid
from aid.agents import Catalog, Instruction, InstructionsContext, InstructionsProvider, discover
from aid.instructions import compose
from aid.protocol import AgentCatalog
from aid.spec import PydanticAISpec

if TYPE_CHECKING:
    from typing import Any

    import pytest

PROVIDER_DIR = Path(__file__).parent / "provider_dir"
EXAMPLE_FILE = Path(__file__).parents[1] / "examples" / "file_instructions.py"


def file_instructions() -> Any:
    """The `examples/file_instructions.py` module, loaded from its file."""
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


def test_discovers_providers_and_lists_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    found = discover([PROVIDER_DIR, PROVIDER_DIR / "missing"])
    assert sorted(found.providers) == ["briefing", "custom-provider", "dup"]
    assert found.providers["briefing"].description() == "A short project brief."
    assert found.providers["custom-provider"].description() == ""
    assert found.providers["dup"].__module__ == "dup_a"
    assert len(found.problems) == 2
    assert any("broken_provider" in problem and "ModuleNotFoundError" in problem for problem in found.problems)
    assert any("dup_a" in problem and "dup_b" in problem for problem in found.problems)


def test_describe_rows_validate_as_agent_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    described = AgentCatalog.model_validate(discover([PROVIDER_DIR]).describe())
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


def test_default_provider_reads_global_agents_and_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    (tmp_path / "instructions.md").write_text("Be kind.")
    project = tmp_path / "project"
    cwd = project / "sub"
    cwd.mkdir(parents=True)
    (project / "AGENTS.md").write_text("Use tabs.")
    (tmp_path / "skills" / "codereview").mkdir(parents=True)
    (tmp_path / "skills" / "codereview" / "SKILL.md").write_text("Review carefully.\nCheck every diff.")
    (tmp_path / "skills" / "empty").mkdir()
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "instructions.md"))
    monkeypatch.setenv(module.ENV_SKILLS_DIR, str(tmp_path / "skills"))
    loaded = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert [(item.name, item.text) for item in loaded] == [
        ("global", "Be kind."),
        ("agents", "Use tabs."),
        ("skill:codereview", "Review carefully.\nCheck every diff."),
    ]


def test_default_provider_prefers_the_nearest_agents_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    (tmp_path / "AGENTS.md").write_text("Root.")
    cwd = tmp_path / "sub"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("Local.")
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "missing.md"))
    monkeypatch.setenv(module.ENV_SKILLS_DIR, str(tmp_path / "missing-skills"))
    [agents] = module.FileInstructions().load(aid.InstructionsContext(cwd=cwd, session_name="demo"))
    assert (agents.name, agents.text) == ("agents", "Local.")


def test_default_provider_skips_absent_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = file_instructions()
    monkeypatch.setenv(module.ENV_INSTRUCTIONS_FILE, str(tmp_path / "missing.md"))
    monkeypatch.setenv(module.ENV_SKILLS_DIR, str(tmp_path / "missing-skills"))
    assert module.FileInstructions().load(aid.InstructionsContext(cwd=tmp_path, session_name="demo")) == []


def test_spec_instruction_fields() -> None:
    assert PydanticAISpec(cwd="/", agent="briefing").instruction_providers is None
    assert PydanticAISpec(cwd="/", agent="briefing").skip_instructions == []
    spec = PydanticAISpec(
        cwd="/", agent="briefing", instruction_providers=["file_instructions"], skip_instructions=["skill:x"]
    )
    assert spec.instruction_providers == ["file_instructions"]
    assert spec.skip_instructions == ["skill:x"]
