"""Forks: a LiteLM session calls `fork(instructions)` and goes on working; a copy of it does only what the
instructions ask, and its answer comes back to the session as a message, which wakes it.

The copy is a session of its own, `<parent>.fork-<hex>`, seeded with the parent's conversation as it stands inside
the `fork` call (`Coding.snapshot`). Its requests start with the parent's, so the provider's prompt cache carries
the shared prefix: a cheap model reads a side question in full and the parent keeps only the answer. The fork is
deleted once it answers, fails or is cancelled. Bundled and on by default (`autostart`).
"""

from __future__ import annotations

import json
import re
import secrets
from typing import TYPE_CHECKING, Final

import anyio
import structlog

import aid
from aid.protocol import AidError, GetSpec
from aid.spec import AgentSpecAdapter, LiteLMSpec

if TYPE_CHECKING:
    from anyio.abc import TaskGroup
    from pydantic import JsonValue

log = structlog.get_logger(__name__)

PLUGIN: Final = "forks"
FORK_NAME: Final = re.compile(r"\.fork-[0-9a-f]{6}$")
PARENT_CHARS: Final = 52
"""Characters of the parent's name a fork keeps: with `.fork-` and six hex digits it fits the 64 of a name."""
CLEANUP_TIMEOUT: Final = 10.0
PREAMBLE: Final = (
    "You are a fork of this session, started by this `fork` call. Do only what its instructions ask, then end "
    "your turn with only the answer they ask for: that text goes back to the parent session as a message. Do "
    "not continue the parent's other work, and do not call `fork`."
)


def is_fork(name: str) -> bool:
    return FORK_NAME.search(name) is not None


@aid.mcptool
async def fork(instructions: str) -> str:
    """Hand a narrow side task to a copy of yourself, and keep working. The copy sees this conversation so
    far, does only what `instructions` ask (look something up, read files, check a claim), and its answer
    arrives later as a message from it, which wakes you if you stopped. Ask for exactly what you need back,
    as short as will do: the point is to keep the detail out of your own context."""
    from aid.coding import CODING  # noqa: PLC0415 -- the daemon hosts this plugin and need not load the coding tools

    try:
        coding = CODING.get()
    except LookupError:
        raise aid.ModelRetry("fork works only in an aid LiteLM session") from None
    if coding.snapshot is None:
        raise aid.ModelRetry("fork works only in an aid LiteLM session")
    snapshot = await coding.snapshot(PREAMBLE)
    async with aid.connect() as client:
        name = await client.call_plugin(PLUGIN, "fork", snapshot=str(snapshot), instructions=instructions)
    return f"Fork {name} started. Its answer arrives as a message from it; carry on meanwhile."


class Forks(aid.Plugin):
    """Short-lived forks of LiteLM sessions, from the `fork` tool."""

    grants = frozenset({aid.Grant.READ, aid.Grant.PROMPT, aid.Grant.MESSAGE, aid.Grant.MANAGE, aid.Grant.SUPERVISE})

    def __init__(self) -> None:
        self._client: aid.Client | None = None
        self._tg: TaskGroup | None = None

    async def run(self, client: aid.Client) -> None:
        self._client = client
        for info in await client.sessions():
            if info.parent is not None and is_fork(info.name):
                log.info("fork_leftover_deleted", fork=info.name)
                await client.session(info.name).delete()
        async with anyio.create_task_group() as tg:
            self._tg = tg
            await anyio.sleep_forever()

    async def call(self, caller: str | None, method: str, args: dict[str, JsonValue]) -> JsonValue:
        if method != "fork":
            raise AidError("unknown_method", f"forks has no method {method!r}")
        if self._client is None or self._tg is None:
            raise AidError("not_running", "forks is not running")
        if caller is None:
            raise AidError("forbidden", "only a session can fork itself")
        snapshot, instructions = args.get("snapshot"), args.get("instructions")
        if not isinstance(snapshot, str) or not isinstance(instructions, str) or not instructions.strip():
            raise AidError("bad_args", "fork takes a snapshot path and instructions")
        parent = AgentSpecAdapter.validate_python(await self._client.call(GetSpec(session=caller)))
        if not isinstance(parent, LiteLMSpec):
            raise AidError("unsupported", "only LiteLM sessions fork")
        if parent.parent is not None:
            raise AidError("forbidden", "a session that works for another cannot fork")
        spec = parent.model_copy(
            update={"seed_history": snapshot, "parent": caller, "supervisor": None, "supervisor_permissions": []}
        )
        name = f"{caller[:PARENT_CHARS]}.fork-{secrets.token_hex(3)}"
        await self._client.create(name, spec)
        self._tg.start_soon(self._answer, caller, name, instructions)
        return name

    async def _answer(self, parent: str, name: str, instructions: str) -> None:
        client = self._client
        assert client is not None
        text = f"Fork {name} ended without an answer."
        try:
            result = await client.session(name).run(instructions)
            output = result.output
            text = output if isinstance(output, str) else json.dumps(output)
            if result.stop_reason != "end_turn":
                text = f"Fork {name} stopped ({result.stop_reason}) before it finished. Its last words:\n{text}"
        except AidError as error:
            text = f"Fork {name} failed: {error.code}: {error.message}"
        finally:
            with anyio.move_on_after(CLEANUP_TIMEOUT, shield=True):
                try:
                    await client.send_message(parent, text or "(the fork answered with no text)", sender=name)
                except AidError as error:
                    log.warning("fork_answer_undelivered", fork=name, parent=parent, code=error.code)
                try:
                    await client.session(name).delete()
                except AidError as error:
                    log.warning("fork_delete_failed", fork=name, code=error.code)
