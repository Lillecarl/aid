"""In-process plugins: registered plugins whose code is on the daemon's plugins path run as supervised tasks
in the daemon, speaking the plugin protocol over the inproc sockets.

What runs: a plugin the registry names (`aid plugin add`), whose code the path holds, and whose registered
grants cover what the code declares. Anything else stays outside: unregistered code (add it first),
grants the registry does not hold (add it again with them), and code that never made the path — external
plugins on other hosts, in other languages, or with conflicting dependencies, which connect over the CURVE
sockets as before.

A task that fails restarts after a delay; one that fails repeatedly stops until the next daemon start,
loudly. Removing the registration stops the task at once: a removed plugin is refused everywhere, in-process
no less than outside. Cancellation ends the task at once: shutdown does not wait out the backoff.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import structlog

from aid import zap
from aid.catalog import listing
from aid.client import spawn_plugin_client
from aid.discovery import Catalog, discover
from aid.plugins import Grant, PluginSpec, hooks, plugins_path, serves_calls
from aid.protocol import MessageHook

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import zmq.asyncio
    from pydantic import JsonValue

    from aid.client import Client
    from aid.plugins import Plugin
    from aid.protocol import AgentCatalog, Invocation

log = structlog.get_logger(__name__)

RESTART_DELAY: Final = 5.0
"""Seconds between a plugin task's failure and its restart."""
FAIL_WINDOW: Final = 60.0
"""Failures this far apart start the count over: a plugin that runs a while before failing is flaky, not broken."""
MAX_FAILURES: Final = 5
"""Failures in one window before the task stops until the next daemon start."""


async def hosted(dirs: Sequence[Path] | None) -> Catalog:
    """The daemon's explicit dirs, or the standard tiers, as the daemon may load them: a subprocess lists the
    plugins, and only those that define a task load here, with what they require. The rest, agents and tools among
    them, stay out of the daemon. Each loads guarded: a failure becomes a problem the daemon logs, never a refusal
    to serve. Raises what `aid.catalog.listing` raises."""
    if dirs is not None and not dirs:
        return Catalog()
    listed = await listing(dirs)
    catalog = await anyio.to_thread.run_sync(load_hosted, dirs, listed)
    catalog.problems[:] = dict.fromkeys([*listed.problems, *catalog.problems])
    return catalog


def load_hosted(dirs: Sequence[Path] | None, listed: AgentCatalog) -> Catalog:
    tasks = [plugin.name for plugin in listed.plugins if plugin.grants is not None and plugin.declined is None]
    return discover(list(dirs) if dirs is not None else plugins_path(), only=tasks)


def autostart(catalog: Catalog) -> dict[str, PluginSpec]:
    """Registrations for the task plugins whose manifest asks to run unregistered: their declared grants, and a
    fresh key nobody holds, since a hosted task needs none."""
    return {
        name: PluginSpec(name=name, public_key=zap.Keypair.new().public, grants=cls.grants)
        for name, cls in catalog.plugins.items()
        if (root := catalog.roots.get(name)) is not None and root.manifest.autostart
    }


def enabled(catalog: Catalog, registered: dict[str, PluginSpec]) -> list[tuple[str, type[Plugin]]]:
    """Plugin classes eligible to host, without construction: selection has no instance lifecycle effects.
    The host constructs only missing tasks. Anything else logs why it stays outside: unregistered at info
    (add it), under-granted at warning (add it again)."""
    ready: list[tuple[str, type[Plugin]]] = []
    for name, cls in catalog.plugins.items():
        spec = registered.get(name)
        if spec is None:
            log.info("plugin_not_registered", plugin=name, hint=f"aid plugin add {name} {_grant_flags(cls.grants)}")
            continue
        if missing := set(cls.grants) - set(spec.grants):
            log.warning(
                "plugin_under_granted",
                plugin=name,
                missing=sorted(grant.value for grant in missing),
                hint=f"aid plugin add {name} {_grant_flags(cls.grants)} again to replace it",
            )
            continue
        ready.append((name, cls))
    return ready


def _grant_flags(grants: frozenset[Grant]) -> str:
    return " ".join(f"--grant {grant.value}" for grant in sorted(grants, key=lambda grant: grant.value))


async def run(plugin: Plugin, client: Client) -> None:
    """Run `plugin` on `client`, serving its calls and hooks beside `run` when it overrides any (`Plugin.call`,
    `Plugin.on_message`)."""
    hooked = hooks(plugin)
    if not serves_calls(plugin) and not hooked:
        await plugin.run(client)
        return

    async def answer(invocation: Invocation) -> JsonValue:
        match invocation.hook:
            case "message":
                message = MessageHook.model_validate(invocation.args)
                await plugin.on_message(message.to, message.sender, message.text)
                return None
            case None:
                return await plugin.call(invocation.caller, invocation.method, invocation.args)

    async with anyio.create_task_group() as tg:
        tg.start_soon(client.serve_calls, answer, hooked)
        await plugin.run(client)
        tg.cancel_scope.cancel()


async def supervise(ctx: zmq.asyncio.Context, name: str, plugin: Plugin) -> None:
    """Run `plugin` until cancelled, restarting after `RESTART_DELAY` on failure; give up after `MAX_FAILURES`
    failures in `FAIL_WINDOW`, loudly. A `run` that returns counts as a failure: loops run forever."""
    failures = 0
    window_start = 0.0
    while True:
        try:
            async with spawn_plugin_client(ctx, name, private=Grant.PRIVATE in plugin.grants) as client:
                await run(plugin, client)
        except anyio.get_cancelled_exc_class():
            raise
        except BaseException as error:
            log.exception("plugin_failed", plugin=name, error=f"{type(error).__name__}: {error}")
        else:
            log.warning("plugin_returned", plugin=name)
        now = time.monotonic()
        failures = failures + 1 if now - window_start < FAIL_WINDOW else 1
        window_start = now
        if failures >= MAX_FAILURES:
            log.error("plugin_disabled", plugin=name, failures=failures)
            return
        await anyio.sleep(RESTART_DELAY)
