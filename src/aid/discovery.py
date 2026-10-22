"""Plugins on the plugins path: what they are, what they need, and what they offer.

A plugin is a directory on the plugins path (`aid.plugins.plugins_path`): a Python package, data such as a
`skills/` directory, or both. Its name is the directory's name, which must be an import name. An optional
`plugin.toml` describes it:

    name = "spark_coder"         # optional; must equal the directory's name
    version = "0.1.0"
    description = "Coding agents on the spark models."
    requires = ["skills"]        # plugins that must load first

`resolve` reads the manifests and imports nothing: the first tier holding a name wins, a missing requirement or a
cycle drops the plugin and everything that needs it, and the rest come back in dependency order. A directory needs
an `__init__.py` or a `plugin.toml` to count.

`discover` imports the resolved packages and sorts what each defines by the interfaces it implements:
`aid.AgentDefinition` (agents), `aid.InstructionsProvider` (session instructions), `@aid.mcptool` (tools) and
`aid.Plugin` (a daemon task, at most one per plugin, registered under the plugin's name). Importing runs the
plugins' code: call it in a process that may run user code, never the daemon, which imports only plugins with a
task, and their requirements, after a subprocess (`python -m aid.catalog`) names them.

A package loads under its own name, without its directory on `sys.path`: another plugin is importable once it has
loaded, which only a requirement guarantees. A name an installed module already holds is refused rather than
shadowed. A plugin's `skills/` directory is read by the bundled `skills` plugin.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aid import builtin_tools
from aid.agents import AgentDefinition, InstructionsProvider
from aid.plugins import Plugin
from aid.tools import McpTool, tool_of

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from types import ModuleType

MANIFEST: Final = "plugin.toml"
PACKAGE: Final = "__init__.py"
IMPORT_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_LOADED: dict[str, Path] = {}
"""Plugin packages this process imported, by name, with their directory."""


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    version: str = ""
    description: str = ""
    requires: list[str] = Field(default_factory=list[str])


@dataclass(frozen=True)
class PluginRoot:
    name: str
    path: Path
    manifest: Manifest

    @property
    def package(self) -> bool:
        return (self.path / PACKAGE).is_file()


@dataclass(frozen=True)
class Resolution:
    roots: list[PluginRoot]
    """Every plugin whose requirements resolved, each after what it requires."""
    problems: list[str]


def read_root(directory: Path) -> PluginRoot:
    """One plugin directory and its manifest. ValueError says what is wrong."""
    manifest = Manifest()
    if (file := directory / MANIFEST).is_file():
        try:
            manifest = Manifest.model_validate(tomllib.loads(file.read_text(encoding="utf-8")))
        except (tomllib.TOMLDecodeError, ValidationError) as error:
            raise ValueError(f"{MANIFEST}: {error}") from error
    if manifest.name and manifest.name != directory.name:
        raise ValueError(f"{MANIFEST} names it {manifest.name}, but a plugin's name is its directory's")
    if not IMPORT_NAME.match(directory.name):
        raise ValueError("a plugin's directory name must be an import name (letters, digits, underscores)")
    return PluginRoot(directory.name, directory, manifest)


def resolve(paths: Sequence[Path]) -> Resolution:
    """The plugins on `paths`, earlier directories first. Reads manifests only: safe anywhere."""
    found: dict[str, PluginRoot] = {}
    problems: list[str] = []
    scanned: set[Path] = set()
    for directory in paths:
        if not directory.is_dir() or (key := directory.resolve()) in scanned:
            continue
        scanned.add(key)
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(("_", ".")):
                continue
            if entry.suffix == ".py" and entry.is_file():
                problems.append(f"{entry}: a plugin is a directory; move it to {entry.with_suffix('')}/{PACKAGE}")
                continue
            if not (entry.is_dir() and ((entry / PACKAGE).is_file() or (entry / MANIFEST).is_file())):
                continue
            try:
                root = read_root(entry)
            except ValueError as error:
                problems.append(f"{entry}: {error}")
                continue
            found.setdefault(root.name, root)
    return Resolution(_order(found, problems), problems)


def _order(found: dict[str, PluginRoot], problems: list[str]) -> list[PluginRoot]:
    ordered: list[PluginRoot] = []
    state: dict[str, bool | None] = {}
    """True resolved, False dropped, None being visited."""

    def visit(name: str, chain: tuple[str, ...]) -> bool:
        if name in state:
            if (done := state[name]) is None:
                problems.append(f"plugin {name}: requirement cycle {' -> '.join((*chain, name))}")
                return False
            return done
        state[name] = None
        ok = True
        for need in found[name].manifest.requires:
            if need not in found:
                problems.append(f"plugin {name}: requires {need}, which is not on the plugins path")
                ok = False
            elif not visit(need, (*chain, name)):
                problems.append(f"plugin {name}: requires {need}, which did not resolve")
                ok = False
        state[name] = ok
        if ok:
            ordered.append(found[name])
        return ok

    for name in found:
        visit(name, ())
    return ordered


def closure(roots: Sequence[PluginRoot], names: Collection[str]) -> list[PluginRoot]:
    """`names` and everything they require, in `roots`' order."""
    by_name = {root.name: root for root in roots}
    wanted: set[str] = set()
    pending = [name for name in names if name in by_name]
    while pending:
        if (name := pending.pop()) not in wanted:
            wanted.add(name)
            pending.extend(need for need in by_name[name].manifest.requires if need in by_name)
    return [root for root in roots if root.name in wanted]


def _within(path: str | None, directory: Path) -> bool:
    return path is not None and Path(path).resolve().is_relative_to(directory.resolve())


def load(root: PluginRoot) -> ModuleType:
    """Import `root`'s package under its name, or return the one already imported from the same directory."""
    if (module := sys.modules.get(root.name)) is not None:
        if _within(getattr(module, "__file__", None), root.path):
            return module
        raise ImportError(
            f"the name {root.name} is taken by {getattr(module, '__file__', None) or 'a built-in module'}"
        )
    if (
        (taken := importlib.util.find_spec(root.name)) is not None
        and taken.origin
        and not _within(taken.origin, root.path)
    ):
        raise ImportError(f"the name {root.name} is taken by {taken.origin}")
    spec = importlib.util.spec_from_file_location(
        root.name, root.path / PACKAGE, submodule_search_locations=[str(root.path)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader for {root.path / PACKAGE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[root.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        forget(root.name)
        raise
    _LOADED[root.name] = root.path
    return module


def forget(name: str) -> None:
    """Drop plugin `name` and its submodules from this process's imports, so the next `load` runs it afresh."""
    _LOADED.pop(name, None)
    for key in [key for key in sys.modules if key == name or key.startswith(f"{name}.")]:
        del sys.modules[key]


def loaded() -> dict[str, Path]:
    return dict(_LOADED)


def _first_paragraph(doc: str | None) -> str:
    return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else ""


@dataclass(frozen=True)
class Catalog:
    roots: dict[str, PluginRoot] = field(default_factory=dict[str, PluginRoot])
    """Plugins that loaded, by name, in dependency order."""
    agents: dict[str, type[AgentDefinition]] = field(default_factory=dict[str, "type[AgentDefinition]"])
    providers: dict[str, type[InstructionsProvider]] = field(default_factory=dict[str, "type[InstructionsProvider]"])
    tools: dict[str, McpTool] = field(default_factory=dict[str, McpTool])
    plugins: dict[str, type[Plugin]] = field(default_factory=dict[str, "type[Plugin]"])
    """Daemon tasks, by plugin name."""
    declined: dict[str, str] = field(default_factory=dict[str, str])
    """Tasks that gated themselves out through `Plugin.available`, with the reason: not applicable, not broken."""
    descriptions: dict[str, str] = field(default_factory=dict[str, str])
    problems: list[str] = field(default_factory=list[str])
    """Plugins that failed to resolve or import, and names offered twice. A problem hides one plugin or one name,
    never the catalog."""

    def describe(self) -> dict[str, Any]:
        def task(name: str) -> list[str] | None:
            cls = self.plugins.get(name)
            return None if cls is None else sorted(grant.value for grant in cls.grants)

        return {
            "agents": [
                {"name": name, "description": cls.description(), "module": cls.__module__}
                for name, cls in sorted(self.agents.items())
            ],
            "providers": [
                {"name": name, "description": cls.description(), "module": cls.__module__}
                for name, cls in sorted(self.providers.items())
            ],
            "tools": [
                {"name": name, "description": tool.summary(), "module": tool.fn.__module__}
                for name, tool in sorted(self.tools.items())
            ],
            "plugins": [
                {
                    "name": name,
                    "version": root.manifest.version,
                    "description": self.descriptions.get(name, ""),
                    "path": str(root.path),
                    "requires": root.manifest.requires,
                    "grants": task(name),
                    "declined": self.declined.get(name),
                }
                for name, root in self.roots.items()
            ],
            "problems": self.problems,
        }


def _owned(owner: str, plugin: str) -> bool:
    return owner == plugin or owner.startswith(f"{plugin}.")


def _add_tool(catalog: Catalog, tool: McpTool) -> None:
    if (other := catalog.tools.get(tool.name)) is not None and other is not tool:
        catalog.problems.append(f"tool {tool.name}: {tool.fn.__module__} loses to {other.fn.__module__}")
        return
    catalog.tools[tool.name] = tool


def _collect(catalog: Catalog, root: PluginRoot, module: ModuleType) -> None:
    tasks: list[type[Plugin]] = []
    for obj in vars(module).values():
        if (tool := tool_of(obj)) is not None:
            if _owned(tool.fn.__module__, root.name):
                _add_tool(catalog, tool)
            continue
        if not isinstance(obj, type) or not _owned(obj.__module__, root.name) or inspect.isabstract(obj):
            continue
        if issubclass(obj, Plugin):
            tasks.append(obj)
        elif issubclass(obj, AgentDefinition):
            name = obj.agent_name()
            if (other := catalog.agents.get(name)) is not None and other is not obj:
                catalog.problems.append(f"agent {name}: {obj.__module__} loses to {other.__module__}")
            else:
                catalog.agents[name] = obj
        elif issubclass(obj, InstructionsProvider):
            name = obj.provider_name()
            if (other := catalog.providers.get(name)) is not None and other is not obj:
                catalog.problems.append(f"provider {name}: {obj.__module__} loses to {other.__module__}")
            else:
                catalog.providers[name] = obj
    if len(tasks) > 1:
        names = ", ".join(sorted(cls.__qualname__ for cls in tasks))
        catalog.problems.append(f"plugin {root.name}: defines more than one aid.Plugin ({names})")
        return
    for cls in tasks:
        try:
            reason = cls.available()
        except Exception as error:
            catalog.problems.append(f"plugin {root.name}: available() raised {type(error).__name__}: {error}")
            return
        if reason is not None:
            catalog.declined[root.name] = reason
        else:
            catalog.plugins[root.name] = cls
        if not catalog.descriptions.get(root.name):
            catalog.descriptions[root.name] = cls.description()


def discover(paths: Sequence[Path], only: Collection[str] | None = None) -> Catalog:
    """Resolve the plugins on `paths` and import their packages in dependency order: `only` limits it to those
    names and their requirements. A plugin that fails to import drops those that require it."""
    resolution = resolve(paths)
    catalog = Catalog(problems=list(resolution.problems))
    for obj in vars(builtin_tools).values():
        if (tool := tool_of(obj)) is not None:
            _add_tool(catalog, tool)
    roots = resolution.roots if only is None else closure(resolution.roots, only)
    for root in roots:
        if missing := [need for need in root.manifest.requires if need not in catalog.roots]:
            catalog.problems.append(f"plugin {root.name}: requires {', '.join(missing)}, which did not load")
            continue
        module = None
        if root.package:
            try:
                module = load(root)
            except Exception as error:
                catalog.problems.append(f"{root.path}: {type(error).__name__}: {error}")
                continue
        catalog.roots[root.name] = root
        catalog.descriptions[root.name] = root.manifest.description or _first_paragraph(
            getattr(module, "__doc__", None)
        )
        if module is not None:
            _collect(catalog, root, module)
    return catalog
