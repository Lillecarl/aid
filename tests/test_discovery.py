"""The plugins path: manifests, tiers, requirements, and what loads in which order. Daemon-free."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aid.discovery import discover, loaded, resolve

if TYPE_CHECKING:
    from pathlib import Path

TASK = "import aid\n\n\nclass {cls}(aid.Plugin):\n    async def run(self, client: aid.Client) -> None: ...\n"


def plugin(tier: Path, name: str, manifest: str | None = None, code: str | None = "") -> Path:
    directory = tier / name
    directory.mkdir(parents=True)
    if manifest is not None:
        (directory / "plugin.toml").write_text(manifest)
    if code is not None:
        (directory / "__init__.py").write_text(code)
    return directory


def names(paths: list[Path]) -> list[str]:
    return [root.name for root in resolve(paths).roots]


def test_first_tier_wins_a_name(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    plugin(first, "shared", 'version = "1"')
    plugin(second, "shared", 'version = "2"')
    plugin(second, "other")
    resolution = resolve([first, second, tmp_path / "missing"])
    assert [(root.name, root.manifest.version) for root in resolution.roots] == [("shared", "1"), ("other", "")]
    assert resolution.problems == []


def test_requirements_load_first_and_missing_ones_drop_dependents(tmp_path: Path) -> None:
    plugin(tmp_path, "app", 'requires = ["lib"]')
    plugin(tmp_path, "lib", 'requires = ["base"]')
    plugin(tmp_path, "base")
    plugin(tmp_path, "orphan", 'requires = ["absent"]')
    plugin(tmp_path, "needs_orphan", 'requires = ["orphan"]')
    resolution = resolve([tmp_path])
    assert [root.name for root in resolution.roots] == ["base", "lib", "app"]
    assert "plugin orphan: requires absent, which is not on the plugins path" in resolution.problems
    assert "plugin needs_orphan: requires orphan, which did not resolve" in resolution.problems


def test_a_cycle_drops_every_plugin_in_it(tmp_path: Path) -> None:
    plugin(tmp_path, "a", 'requires = ["b"]')
    plugin(tmp_path, "b", 'requires = ["a"]')
    plugin(tmp_path, "c")
    resolution = resolve([tmp_path])
    assert [root.name for root in resolution.roots] == ["c"]
    assert any("requirement cycle a -> b -> a" in problem for problem in resolution.problems)


def test_bad_directories_are_problems_and_others_are_ignored(tmp_path: Path) -> None:
    plugin(tmp_path, "renamed", 'name = "other"')
    plugin(tmp_path, "unknown_key", "colour = 1")
    plugin(tmp_path, "broken_toml", "requires = [")
    plugin(tmp_path, "has-dash")
    plugin(tmp_path, "_private")
    (tmp_path / "plain_dir").mkdir()
    (tmp_path / "loose.py").write_text("")
    resolution = resolve([tmp_path])
    assert resolution.roots == []
    problems = "\n".join(resolution.problems)
    assert "names it other" in problems
    assert "unknown_key" in problems and "broken_toml" in problems
    assert "has-dash: a plugin's directory name must be an import name" in problems
    assert f"{tmp_path / 'loose.py'}: a plugin is a directory; move it to {tmp_path / 'loose'}/__init__.py" in problems
    assert len(resolution.problems) == 5


def test_a_manifest_alone_makes_a_data_plugin(tmp_path: Path) -> None:
    plugin(tmp_path, "data_only", 'description = "Skills only."', code=None)
    catalog = discover([tmp_path])
    assert list(catalog.roots) == ["data_only"]
    assert catalog.describe()["plugins"][0]["description"] == "Skills only."
    assert "data_only" not in loaded()


def test_a_failed_import_drops_dependents(tmp_path: Path) -> None:
    plugin(tmp_path, "fails", code="raise RuntimeError('no')\n")
    plugin(tmp_path, "needs_it", 'requires = ["fails"]', code="import fails\n")
    catalog = discover([tmp_path])
    assert catalog.roots == {}
    assert any("fails: RuntimeError: no" in problem for problem in catalog.problems)
    assert "plugin needs_it: requires fails, which did not load" in catalog.problems


def test_a_plugin_imports_what_it_requires_by_name(tmp_path: Path) -> None:
    plugin(tmp_path, "lib", code="VALUE = 42\n")
    plugin(tmp_path, "app", 'requires = ["lib"]', code="from lib import VALUE\n")
    plugin(tmp_path, "hidden")
    plugin(tmp_path, "sneaky", code="import hidden\n")
    catalog = discover([tmp_path], only=["app", "sneaky"])
    assert sorted(catalog.roots) == ["app", "lib"]
    assert any("sneaky" in problem and "ModuleNotFoundError" in problem for problem in catalog.problems)


def test_only_imports_the_named_plugins_and_their_requirements(tmp_path: Path) -> None:
    plugin(tmp_path, "wanted", 'requires = ["dep"]')
    plugin(tmp_path, "dep")
    plugin(tmp_path, "unwanted", code="raise AssertionError('imported')\n")
    catalog = discover([tmp_path], only=["wanted"])
    assert list(catalog.roots) == ["dep", "wanted"]
    assert catalog.problems == []


def test_an_installed_module_name_is_refused(tmp_path: Path) -> None:
    plugin(tmp_path, "json")
    catalog = discover([tmp_path])
    assert catalog.roots == {}
    assert any("the name json is taken by" in problem for problem in catalog.problems)


def test_one_task_per_plugin_named_after_it(tmp_path: Path) -> None:
    plugin(tmp_path, "single", code=TASK.format(cls="Anything"))
    plugin(tmp_path, "double", code=TASK.format(cls="One") + "\n\n" + TASK.format(cls="Two").split("\n\n\n", 1)[1])
    catalog = discover([tmp_path])
    assert list(catalog.plugins) == ["single"]
    assert catalog.plugins["single"].plugin_name() == "single"
    assert "plugin double: defines more than one aid.Plugin (One, Two)" in catalog.problems


def test_classes_from_submodules_belong_to_their_plugin(tmp_path: Path) -> None:
    directory = plugin(tmp_path, "split", code="from .agents import Helper\nfrom .task import Task\n")
    (directory / "agents.py").write_text(
        "import aid\n\n\nclass Helper(aid.AgentDefinition):\n    def build(self):\n        raise NotImplementedError\n"
    )
    (directory / "task.py").write_text(TASK.format(cls="Task"))
    catalog = discover([tmp_path])
    assert list(catalog.agents) == ["helper"]
    assert catalog.plugins["split"].plugin_name() == "split"
