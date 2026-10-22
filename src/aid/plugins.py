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

A plugin's task lives in a plugin package on the plugins path (`aid.discovery`), as its one `Plugin` subclass:

    import aid

    class Telegram(aid.Plugin):
        \"\"\"Report settled turn ends to Telegram, and take basic commands back.\"\"\"

        grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

        async def run(self, client: aid.Client) -> None:
            ...

The plugins path is AID_PLUGINS_PATH, directories joined with `:`, then `$XDG_CONFIG_HOME/aid/plugins`,
then the installed set's directory from AID_PLUGINS_DIST (which the installed `aid` wrapper sets, and which is
empty elsewhere): explicit choice first, the user's own directory next, shipped stock last.
"""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    """The interface: subclass it once in a plugin package, declare its grants, and drive aid from `run`.

    The base gains only defaulted members: a subclass written today runs against tomorrow's base."""

    grants: ClassVar[frozenset[Grant]] = frozenset()
    """What it asks of the daemon: `aid plugin add` grants exactly these."""

    @classmethod
    def plugin_name(cls) -> str:
        """The name it registers and runs as: its plugin's, the top-level package that defines it."""
        return cls.__module__.split(".")[0]

    @classmethod
    def description(cls) -> str:
        # `cls.__doc__`, not inspect.getdoc: that falls back to this base class's docstring.
        doc = cls.__doc__ or ""
        return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else ""

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


def plugins_path(env: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    paths = [Path(part).expanduser() for part in env.get(ENV_PLUGINS_PATH, "").split(":") if part]
    config = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    paths.append(config / "aid" / "plugins")
    if dist := env.get(ENV_PLUGINS_DIST):
        paths.append(Path(dist))
    return paths
