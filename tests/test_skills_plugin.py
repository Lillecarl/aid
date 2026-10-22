from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import aid
from aid.discovery import discover
from aid.plugins import ENV_PLUGINS_PATH

PLUGINS = Path(__file__).parents[1] / "plugins"


def skills() -> Any:
    """The bundled `plugins/skills` package, loaded from its file."""
    name = "aid_plugin_skills"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PLUGINS / "skills" / "__init__.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_skill(directory: Path, name: str, description: str = "Does a thing.", body: str = "Body.\n") -> Path:
    (directory / name).mkdir(parents=True, exist_ok=True)
    path = directory / name / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}")
    return path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("AID_SKILLS_PATH", raising=False)
    return home


@pytest.mark.parametrize(
    ("frontmatter", "reason"),
    [
        ("name: other\ndescription: x", "differs from its directory"),
        ("name: Bad_Name\ndescription: x", "lowercase"),
        ("name: bad--name\ndescription: x", "single hyphens"),
        ("name: tool", "description is missing"),
        ("name: tool\ndescription: " + "x" * 1025, "longer than 1024"),
        ("name: [tool", "not YAML"),
    ],
)
def test_parse_holds_skills_to_the_spec(tmp_path: Path, frontmatter: str, reason: str) -> None:
    (tmp_path / "tool").mkdir()
    path = tmp_path / "tool" / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\nBody.\n")
    with pytest.raises(ValueError, match=reason):
        skills().parse(path)


def test_parse_reads_yaml_frontmatter(tmp_path: Path) -> None:
    (tmp_path / "tool").mkdir()
    path = tmp_path / "tool" / "SKILL.md"
    path.write_text("---\nname: tool\ndescription: >\n  Folded over\n  two lines.\nmetadata:\n  a: b\n---\n# Tool\n")
    parsed = skills().parse(path)
    assert (parsed.name, parsed.description, parsed.body) == ("tool", "Folded over two lines.", "# Tool\n")


def test_nearest_directory_wins_a_name(tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "sub"
    cwd.mkdir(parents=True)
    (repo / ".git").mkdir()
    write_skill(tmp_path / ".claude" / "skills", "above-the-repo")
    write_skill(cwd / ".agents" / "skills", "shared", "Project, nearest.")
    write_skill(repo / ".claude" / "skills", "shared", "Project, at the root.")
    write_skill(repo / ".claude" / "skills", "project-only")
    extra = tmp_path / "extra"
    write_skill(extra, "shared", "From AID_SKILLS_PATH.")
    write_skill(extra, "extra-only")
    write_skill(home / ".config" / "aid" / "skills", "extra-only", "Shadowed by AID_SKILLS_PATH.")
    write_skill(home / ".claude" / "skills", "global-only")
    monkeypatch.setenv("AID_SKILLS_PATH", str(extra))
    found = skills().discover(cwd)
    assert {name: s.description for name, s in found.items()} == {
        "shared": "Project, nearest.",
        "project-only": "Does a thing.",
        "extra-only": "Does a thing.",
        "global-only": "Does a thing.",
    }


def test_plugins_offer_skills_after_the_global_dirs(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = tmp_path / "plugins"
    (tier / "kit").mkdir(parents=True)
    (tier / "kit" / "plugin.toml").write_text('description = "Skills only."\n')
    write_skill(tier / "kit" / "skills", "kit-skill", "From the kit plugin.")
    write_skill(tier / "kit" / "skills", "shared", "From the kit plugin.")
    (tier / "unresolved").mkdir()
    (tier / "unresolved" / "plugin.toml").write_text('requires = ["absent"]\n')
    write_skill(tier / "unresolved" / "skills", "orphaned")
    write_skill(home / ".claude" / "skills", "shared", "The user's own.")
    monkeypatch.setenv(ENV_PLUGINS_PATH, str(tier))
    found = skills().discover(tmp_path)
    assert {name: s.description for name, s in found.items()} == {
        "shared": "The user's own.",
        "kit-skill": "From the kit plugin.",
    }


def test_a_broken_skill_is_left_out(tmp_path: Path, home: Path) -> None:
    write_skill(home / ".claude" / "skills", "good")
    (home / ".claude" / "skills" / "bad").mkdir()
    (home / ".claude" / "skills" / "bad" / "SKILL.md").write_text("no frontmatter\n")
    assert list(skills().discover(tmp_path)) == ["good"]


def test_index_holds_descriptions_only(tmp_path: Path, home: Path) -> None:
    write_skill(home / ".claude" / "skills", "jj", "Jujutsu version control.", body="Secret body.\n")
    loaded = skills().Skills().load(aid.InstructionsContext(cwd=tmp_path, session_name="s"))
    assert [(item.name, item.text) for item in loaded] == [
        ("skills", skills().HEADER),
        ("skill:jj", "Jujutsu version control."),
    ]
    (home / ".claude" / "skills" / "jj" / "SKILL.md").unlink()
    assert skills().Skills().load(aid.InstructionsContext(cwd=tmp_path, session_name="s")) == []


def test_skill_tool_loads_body_and_files(tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skill directories are often symlinks into a store; the files resolve inside the real one."""
    store = tmp_path / "store"
    write_skill(store, "jj", body="# jj\n\nUse jj.\n")
    (store / "jj" / "references").mkdir()
    (store / "jj" / "references" / "split.md").write_text("Split by file.")
    (home / ".claude" / "skills").mkdir(parents=True)
    (home / ".claude" / "skills" / "jj").symlink_to(store / "jj")
    write_skill(home / ".claude" / "skills", "nix")
    monkeypatch.chdir(tmp_path)
    loaded = skills().skill("jj")
    assert loaded.startswith(f"Skill jj, from {(store / 'jj').resolve()}.")
    assert "references/split.md" in loaded and "# jj\n\nUse jj." in loaded and "name: jj" not in loaded
    assert skills().skill("jj", file="references/split.md") == "Split by file."
    with pytest.raises(aid.ModelRetry, match="not a file of skill"):
        skills().skill("jj", file="../nix/SKILL.md")
    with pytest.raises(aid.ModelRetry, match="the skills are: jj, nix"):
        skills().skill("absent")


def test_plugin_discovery_offers_the_provider_and_tool() -> None:
    catalog = discover([PLUGINS])
    assert catalog.providers["skills"].__module__ == "skills"
    assert catalog.tools["skill"].fn.__module__ == "skills"
