"""Plugins: processes the daemon knows by a CURVE public key, each with grants that name what it may ask.

A plugin connects to `Paths.plugins` (requests: a DEALER) and `Paths.plugin_events` (published changes: a SUB), both
CURVE. The ZAP handler names the plugin as the connection's User-Id; the daemon checks each request against the
plugin's grants as it is now, so a removed plugin is refused at once. Only the control socket changes the registry.

What grants bind: what arrives on the plugin sockets. A plugin on this host, running as this user, can open the
control socket, which checks nothing: there its grants are a contract, not a wall, until it runs confined away from
the runtime directory (Lillecarl/aid#4). A plugin elsewhere, which reaches only the plugin sockets, is held to them.
The events socket cannot drop one subscriber: a plugin that loses `read` still hears what is published until it
reconnects.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

PLUGIN_FILE: Final = "plugin.json"
SECRET_FILE: Final = "secret"
"""A plugin's CURVE secret key (Z85), beside its plugin.json, for `aid.connect(plugin=...)` on this host."""


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


def sender(name: str) -> str:
    """What a message from plugin `name` names as its sender."""
    return f"plugin:{name}"
