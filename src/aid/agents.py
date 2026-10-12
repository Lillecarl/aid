"""Agents and instruction providers offered to aid by code: subclasses of `AgentDefinition`
and `InstructionsProvider` in modules on the agents path.

    import aid
    from aid.agent import Agent

    class DeepSeek(aid.AgentDefinition):
        \"\"\"Concise answers from DeepSeek.\"\"\"

        def build(self) -> Agent:
            return Agent("deepseek/deepseek-chat")

The agents path is AID_AGENTS_PATH, directories joined with `:`, and defaults to `$XDG_CONFIG_HOME/aid/agents`.
Every top-level module and package in those directories is imported; each concrete subclass defined there is
an agent (or provider), named after its class in snake case unless it sets `name`.
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

from pydantic import BaseModel, ConfigDict

from aid import builtin_tools
from aid.tools import McpTool, tool_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aid.agent import Agent

ENV_AGENTS_PATH: Final = "AID_AGENTS_PATH"


class AgentDefinition(ABC):
    """The interface: subclass it, give it a docstring, and return a LiteLM agent from `build`."""

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
    def build(self) -> Agent:
        """Make the agent. Runs once per session worker, in that worker."""


class Instruction(BaseModel):
    """One named piece of session instructions: `compose` prints each under its name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    text: str


class InstructionsContext(BaseModel):
    """What a provider loads for: the session's directory, and the session's name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: Path
    session_name: str


class InstructionsProvider(ABC):
    """The interface: subclass it, and return the session's instructions from `load`."""

    name: ClassVar[str] = ""
    """The name specs select. Empty means the class name in snake case."""

    @classmethod
    def provider_name(cls) -> str:
        return cls.name or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def description(cls) -> str:
        # `cls.__doc__`, not inspect.getdoc: that falls back to this base class's docstring.
        doc = cls.__doc__ or ""
        return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else ""

    @abstractmethod
    def load(self, context: InstructionsContext) -> list[Instruction]:
        """Load the instructions for `context`."""


@dataclass(frozen=True)
class Catalog:
    agents: dict[str, type[AgentDefinition]] = field(default_factory=dict[str, "type[AgentDefinition]"])
    providers: dict[str, type[InstructionsProvider]] = field(default_factory=dict[str, "type[InstructionsProvider]"])
    tools: dict[str, McpTool] = field(default_factory=dict[str, McpTool])
    problems: list[str] = field(default_factory=list[str])
    """Modules that failed to import and names defined twice. A problem hides one agent or provider, not the catalog."""

    def describe(self) -> dict[str, Any]:
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


def _add_tool(catalog: Catalog, module: str, tool: McpTool) -> None:
    # A tool imported from another module on the path is that module's to offer.
    if tool.fn.__module__ != module:
        return
    if (other := catalog.tools.get(tool.name)) is not None and other is not tool:
        catalog.problems.append(f"tool {tool.name}: defined by {other.fn.__module__} and {module}")
        return
    catalog.tools[tool.name] = tool


def discover(paths: Sequence[Path]) -> Catalog:
    """Import every module on `paths` and collect its agents. Imports run the modules' code: call it in a
    process that may run user code, a worker or `python -m aid.catalog`, never the daemon."""
    catalog = Catalog()
    for obj in vars(builtin_tools).values():
        if (tool := tool_of(obj)) is not None:
            _add_tool(catalog, builtin_tools.__name__, tool)
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
                if (tool := tool_of(obj)) is not None:
                    _add_tool(catalog, module.__name__, tool)
                    continue
                if isinstance(obj, type) and issubclass(obj, InstructionsProvider):
                    if obj.__module__ != module.__name__ or inspect.isabstract(obj):
                        continue
                    name = obj.provider_name()
                    if (other := catalog.providers.get(name)) is not None and other is not obj:
                        catalog.problems.append(f"{name}: defined by {other.__module__} and {obj.__module__}")
                        continue
                    catalog.providers[name] = obj
                    continue
                if not (isinstance(obj, type) and issubclass(obj, AgentDefinition)):
                    continue
                if obj.__module__ != module.__name__ or inspect.isabstract(obj):
                    continue
                name = obj.agent_name()
                if (other := catalog.agents.get(name)) is not None and other is not obj:
                    catalog.problems.append(f"{name}: defined by {other.__module__} and {obj.__module__}")
                    continue
                catalog.agents[name] = obj
    return catalog
