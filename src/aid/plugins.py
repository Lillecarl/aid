"""Plugins: processes the daemon knows by a CURVE public key, each with grants that name what it may ask.

A plugin connects to `Paths.plugins` (requests: a DEALER) and `Paths.plugin_events` (published changes: a SUB), both
CURVE — or, for a task the daemon spawned, to the inproc pair below, on the daemon's context, with the task's
name as its identity. The ZAP handler names an outside plugin as its connection's User-Id; the daemon checks each
request against the plugin's grants as it is now, so a removed plugin is refused at once. Only the control
socket changes the registry.

What grants bind: what arrives on the plugin sockets. A plugin on this host, running as this user, can open the
control socket, which checks nothing: there its grants are a contract, not a wall, until it runs confined away from
the runtime directory (Lillecarl/aid#4). A plugin elsewhere, which reaches only the plugin sockets, is held to them.
The events socket cannot drop one subscriber: a plugin that loses `read` still hears what is published until it
reconnects.

A plugin's code lives outside core, in a module on the plugins path that registers itself on import by
subclassing `Plugin`:

    import aid

    class Telegram(aid.Plugin):
        \"\"\"Report settled turn ends to Telegram, and take basic commands back.\"\"\"

        grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

        async def run(self, client: aid.Client) -> None:
            ...

The plugins path is AID_PLUGINS_PATH, directories joined with `:`, then `$XDG_CONFIG_HOME/aid/plugins`,
then the installed set's directory from AID_PLUGINS_DIST (which the installed `aid` wrapper sets, and which is
empty elsewhere): explicit choice first, the user's own directory next, shipped stock last.
Discovery imports every top-level module there, so it runs the modules' code: call it in a process that may run
user code (`aid plugin run`, `python -m aid.plugin_catalog`), never the daemon. A plugin plugs into aid, agent
sessions, or both: `run` drives aid as a client over the plugin sockets, while `@aid.mcptool` functions in its
module are offered to agent sessions as tools, beside the agents path's own.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from aid.tools import McpTool, tool_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aid.client import Client

PLUGIN_FILE: Final = "plugin.json"
SECRET_FILE: Final = "secret"
"""A plugin's CURVE secret key (Z85), beside its plugin.json, for `aid.connect(plugin=...)` on this host."""

INPROC_PLUGINS: Final = "inproc://aid-plugins"
INPROC_PLUGIN_EVENTS: Final = "inproc://aid-plugin-events"
"""The plugin sockets inside the daemon: tasks it spawned speak the same protocol over these, on its context,
with the task's name as its identity. Process-private by construction, so no CURVE: no other process can
connect, and the daemon names every task it spawns."""


class Grant(StrEnum):
    READ = "read"
    """List sessions and agents, read status, history and summaries, hear what the daemon publishes."""
    PROMPT = "prompt"
    """Prompt a session and cancel its turn."""
    MESSAGE = "message"
    """Leave a message for a session; it arrives from `plugin:<name>`."""
    PERMISSIONS = "permissions"
    """Answer a session's permission requests; the history names the plugin."""
    MANAGE = "manage"
    """Create, start, stop and delete sessions."""


NEEDS: Final[dict[str, Grant]] = {
    "list": Grant.READ,
    "agents": Grant.READ,
    "status": Grant.READ,
    "history": Grant.READ,
    "summary": Grant.READ,
    "prompt": Grant.PROMPT,
    "cancel": Grant.PROMPT,
    "send": Grant.MESSAGE,
    "answer_permission": Grant.PERMISSIONS,
    "create": Grant.MANAGE,
    "start": Grant.MANAGE,
    "stop": Grant.MANAGE,
    "delete": Grant.MANAGE,
}
"""The grant each request needs, by op. No grant covers the rest: a worker's (`hook`), a channel's (`receive`), the
pane's (`screen`, `pane`) and the registry's own."""


class PluginSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    public_key: Annotated[str, Field(min_length=40, max_length=40)]
    """Z85, as `zmq.curve_keypair` makes it."""
    grants: frozenset[Grant]

    def allows(self, op: str) -> bool:
        need = NEEDS.get(op)
        return need is not None and need in self.grants


ENV_PLUGINS_PATH: Final = "AID_PLUGINS_PATH"
ENV_PLUGINS_DIST: Final = "AID_PLUGINS_DIST"
"""The installed set's plugins directory, for the path's last tier. Set by the installed `aid` wrapper."""


def sender(name: str) -> str:
    """What a message from plugin `name` names as its sender."""
    return f"plugin:{name}"


class Plugin(ABC):
    """The interface: subclass it in a module on the plugins path, declare its grants, and drive aid from `run`.

    The base gains only defaulted members: a subclass written today runs against tomorrow's base."""

    name: ClassVar[str] = ""
    """The name it registers and runs as. Empty means the class name in snake case."""
    version: ClassVar[str] = ""
    """The plugin's own version, free-form. Core compatibility is declared socially, never enforced."""
    grants: ClassVar[frozenset[Grant]] = frozenset()
    """What it asks of the daemon: `aid plugin add` grants exactly these."""

    @classmethod
    def plugin_name(cls) -> str:
        return cls.name or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def description(cls) -> str:
        # `cls.__doc__`, not inspect.getdoc: that falls back to this base class's docstring.
        doc = cls.__doc__ or ""
        return doc.split("\n\n")[0].replace("\n", " ") if doc else ""

    @classmethod
    def available(cls) -> str | None:
        """Why this plugin declines to load, or None when it loads. Static gates live here — platform,
        dependencies, build flags — not runtime secrets, which fail in `run` where the operator sees them.
        Discovery records the reason and moves on."""
        return None

    @abstractmethod
    async def run(self, client: Client) -> None:
        """Drive aid as this plugin, against the grants it registered. Runs in the daemon as a task (over the
        inproc sockets) or in its own process (`aid plugin run`); the client speaks the same protocol either
        way, and is shared across whatever loops it starts."""


@dataclass(frozen=True)
class PluginCatalog:
    plugins: dict[str, type[Plugin]] = field(default_factory=dict[str, "type[Plugin]"])
    tools: dict[str, McpTool] = field(default_factory=dict[str, McpTool])
    declined: dict[str, str] = field(default_factory=dict[str, str])
    """Plugins that gated themselves out through `available`, by name with their reason: not applicable, not broken."""
    problems: list[str] = field(default_factory=list[str])
    """Modules that failed to import and names defined twice. A problem hides one plugin, not the catalog."""

    def describe(self) -> dict[str, object]:
        return {
            "plugins": [
                {
                    "name": name,
                    "description": cls.description(),
                    "version": cls.version,
                    "grants": sorted(grant.value for grant in cls.grants),
                    "module": cls.__module__,
                }
                for name, cls in sorted(self.plugins.items())
            ],
            "tools": [
                {"name": name, "description": tool.summary(), "module": tool.fn.__module__}
                for name, tool in sorted(self.tools.items())
            ],
            "declined": {name: self.declined[name] for name in sorted(self.declined)},
            "problems": self.problems,
        }


def plugins_path(env: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    paths = [Path(part).expanduser() for part in env.get(ENV_PLUGINS_PATH, "").split(":") if part]
    config = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    paths.append(config / "aid" / "plugins")
    if dist := env.get(ENV_PLUGINS_DIST):
        paths.append(Path(dist))
    return paths


def discover(paths: Sequence[Path]) -> PluginCatalog:
    """Import every module on `paths` and collect the plugins and tools each registers. Imports run the modules'
    code: call it in a process that may run user code (`aid plugin run`, `python -m aid.plugin_catalog`), or in
    the daemon at startup, where each module loads guarded and a failure becomes a problem, never a refusal."""
    catalog = PluginCatalog()
    seen: dict[str, str] = {}
    seen_tools: dict[str, str] = {}
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
                    _add_tool(catalog, seen_tools, module.__name__, f"{module.__name__} in {directory}", tool)
                    continue
                if not (isinstance(obj, type) and issubclass(obj, Plugin)):
                    continue
                if obj.__module__ != module.__name__ or inspect.isabstract(obj):
                    continue
                name = obj.plugin_name()
                origin = f"{module.__name__} in {directory}"
                if (other := catalog.plugins.get(name)) is not None and other is not obj:
                    catalog.problems.append(f"plugin {name}: {origin} loses to {seen[name]}")
                    continue
                try:
                    reason = obj.available()
                except Exception as error:
                    catalog.problems.append(f"plugin {name}: available() raised {type(error).__name__}: {error}")
                    continue
                seen[name] = origin
                if reason is not None:
                    catalog.declined[name] = reason
                    continue
                catalog.plugins[name] = obj
    return catalog


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


def _add_tool(catalog: PluginCatalog, seen: dict[str, str], module: str, origin: str, tool: McpTool) -> None:
    # A tool imported from another module on the path is that module's to offer.
    if tool.fn.__module__ != module:
        return
    if (other := catalog.tools.get(tool.name)) is not None and other is not tool:
        catalog.problems.append(f"tool {tool.name}: {origin} loses to {seen[tool.name]}")
        return
    seen[tool.name] = origin
    catalog.tools[tool.name] = tool
