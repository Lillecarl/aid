"""Agents, instruction providers and model providers offered to aid by code: subclasses of `AgentDefinition`,
`InstructionsProvider` and `ModelProvider` in plugin packages (`aid.discovery`).

    import aid
    from aid.agent import Agent

    class DeepSeek(aid.AgentDefinition):
        \"\"\"Concise answers from DeepSeek.\"\"\"

        def build(self) -> Agent:
            return Agent("deepseek/deepseek-flash")

Each concrete subclass a plugin defines is an agent (or provider), named after its class in snake case unless it
sets `name`.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from aid.agent import Agent
    from aid.model import Transport


class AgentDefinition(ABC):
    """The interface: subclass it, give it a docstring, and return a native agent from `build`."""

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


class ModelProvider(ABC):
    """The interface: subclass it, give it a docstring, and return a transport from `transport`. Resolution and
    lifecycle: `aid.providers`."""

    name: ClassVar[str] = ""
    """The prefix it serves (`deepseek` for `deepseek/deepseek-flash`). Empty means the class name in snake case."""

    @classmethod
    def provider_name(cls) -> str:
        return cls.name or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def description(cls) -> str:
        # `cls.__doc__`, not inspect.getdoc: that falls back to this base class's docstring.
        doc = cls.__doc__ or ""
        return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else ""

    @abstractmethod
    def transport(self, model: str) -> Transport:
        """The transport for `model`, the name after the prefix. Its `model_name` is the whole
        `provider/model`, which pricing and the history record."""
