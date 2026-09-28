"""Agents offered to aid by code: subclasses of `PydanticAgent` in modules on the agents path.

    import aid
    from pydantic_ai import Agent

    class DeepSeek(aid.PydanticAgent):
        \"\"\"Concise answers from DeepSeek.\"\"\"

        def build(self) -> Agent:
            return Agent("deepseek:deepseek-chat")

The agents path is AID_AGENTS_PATH, directories joined with `:`, and defaults to `$XDG_CONFIG_HOME/aid/agents`.
Every top-level module and package in those directories is imported; each concrete subclass defined there is
an agent, named after its class in snake case unless it sets `name`.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic_ai.agent import AbstractAgent

ENV_AGENTS_PATH: Final = "AID_AGENTS_PATH"


class PydanticAgent(ABC):
    """The interface: subclass it, give it a docstring, and return a pydantic-ai agent from `build`."""

    name: ClassVar[str] = ""
    """The name sessions use. Empty means the class name in snake case."""

    @classmethod
    def agent_name(cls) -> str:
        return cls.name or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def description(cls) -> str:
        # `cls.__doc__`, not inspect.getdoc: that falls back to this base class's docstring.
        doc = cls.__doc__ or ""
        return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else ""

    @abstractmethod
    def build(self) -> AbstractAgent[Any, Any]:
        """Make the agent. Runs once per session worker, in that worker."""


@dataclass(frozen=True)
class Catalog:
    agents: dict[str, type[PydanticAgent]] = field(default_factory=dict[str, "type[PydanticAgent]"])
    problems: list[str] = field(default_factory=list[str])
    """Modules that failed to import and names defined twice. A problem hides one agent, not the catalog."""

    def describe(self) -> dict[str, Any]:
        return {
            "agents": [
                {"name": name, "description": cls.description(), "module": cls.__module__}
                for name, cls in sorted(self.agents.items())
            ],
            "problems": self.problems,
        }


def agents_path(env: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    if value := env.get(ENV_AGENTS_PATH):
        return [Path(part).expanduser() for part in value.split(":") if part]
    config = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return [config / "aid" / "agents"]


def _module_names(directory: Path) -> list[str]:
    names: list[str] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith(("_", ".")):
            continue
        if entry.suffix == ".py" and entry.is_file():
            names.append(entry.stem)
        elif (entry / "__init__.py").is_file():
            names.append(entry.name)
    return names


def discover(paths: Sequence[Path]) -> Catalog:
    """Import every module on `paths` and collect its agents. Imports run the modules' code: call it in a
    process that may run user code, a worker or `python -m aid.catalog`, never the daemon."""
    catalog = Catalog()
    for directory in paths:
        if not directory.is_dir():
            continue
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        for module_name in _module_names(directory):
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                catalog.problems.append(f"{directory / module_name}: {type(error).__name__}: {error}")
                continue
            for obj in vars(module).values():
                if not (isinstance(obj, type) and issubclass(obj, PydanticAgent)):
                    continue
                if obj.__module__ != module.__name__ or inspect.isabstract(obj):
                    continue
                name = obj.agent_name()
                if (other := catalog.agents.get(name)) is not None and other is not obj:
                    catalog.problems.append(f"{name}: defined by {other.__module__} and {obj.__module__}")
                    continue
                catalog.agents[name] = obj
    return catalog
