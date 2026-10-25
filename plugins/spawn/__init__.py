"""Spawned sessions: a native session calls `spawn` and gets a session of its own that works for it, on a ceiling cut
from its own tools (Lillecarl/aid#50, phase 3).

The child is a persistent session made from instructions, `parent` set (`aid.spawn.child_spec` decides its spec).
The parent talks to it with `send_message`; every turn the child ends reaches the parent as a message from the
child, with its usage. The daemon deletes children with their parent; `dismiss` deletes one sooner. Which children
this plugin made, and the last entry of each it relayed, live in its plugin dir, so a restart relays what ended
meanwhile. Bundled and on by default (`autostart`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import anyio
import structlog
from pydantic import BaseModel, TypeAdapter, ValidationError

import aid
from aid.paths import default_paths
from aid.protocol import AidError, GetSpec, Output, TurnError, Usage
from aid.spawn import SpawnRequest, child_name, child_spec
from aid.spec import AgentSpecAdapter

if TYPE_CHECKING:
    from anyio.abc import TaskGroup
    from pydantic import JsonValue

log = structlog.get_logger(__name__)

PLUGIN: Final = "spawn"
CHILDREN_FILE: Final = "children.json"
PRESENCE_CHECK: Final = 30.0
"""Seconds of quiet after which a relay checks that its child still exists: a deleted session records nothing."""


class Child(BaseModel):
    parent: str
    relayed: int = -1
    """The child's last history entry the parent has heard about."""


_CHILDREN: Final = TypeAdapter(dict[str, Child])


def relayed_text(name: str, item: Output | TurnError) -> str:
    if isinstance(item, TurnError):
        return f"{name} failed: {item.code}: {item.message}"
    text = item.output if isinstance(item.output, str) else json.dumps(item.output)
    if item.stop_reason != "end_turn":
        return f"{name} stopped ({item.stop_reason}). Its last words:\n{text}"
    return text or f"({name} answered with no text)"


def parents_usage(name: str, usage: Usage) -> Usage:
    """The child's turn as its parent records it. Its context gauge measures the child, not the parent."""
    return usage.model_copy(update={"agent": f"spawned {name}", "context_used": None, "context_size": None})


@aid.mcptool
async def spawn(
    label: str, instructions: str, tools: list[str], model: str | None = None, task: str | None = None
) -> str:
    """Start a session that works for you, and keep working. It runs `instructions` as its own instructions, on your
    model unless `model` names another you may use, with only the tools of yours that `tools` (regexes, whole
    names) match; `[]` gives it none. `task`, if given, is its first message from you. Each turn it ends comes back
    to you as a message from it; talk to it with `send_message`. It lasts until you `dismiss` it or you are deleted.
    `label`: lowercase letters, digits and dashes, at most 16; the session is named `<you>.<label>`."""
    async with aid.connect() as client:
        result = await client.call_plugin(
            PLUGIN, "spawn", label=label, instructions=instructions, tools=list(tools), model=model, task=task
        )
    return json.dumps(result)


@aid.mcptool
async def dismiss(session: str) -> str:
    """Delete a session you spawned, and any it spawned."""
    async with aid.connect() as client:
        await client.call_plugin(PLUGIN, "dismiss", session=session)
    return f"{session} is deleted."


class Spawn(aid.Plugin):
    """Children of native sessions, from the `spawn` tool, and the relay of their turns to their parents."""

    grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE, aid.Grant.MANAGE, aid.Grant.SUPERVISE})

    def __init__(self) -> None:
        self._client: aid.Client | None = None
        self._tg: TaskGroup | None = None
        self._file = default_paths().plugin_dir(self.plugin_name()) / CHILDREN_FILE
        self._children: dict[str, Child] = {}
        self._relays: dict[str, anyio.CancelScope] = {}
        self._saving = anyio.Lock()

    @property
    def client(self) -> aid.Client:
        if self._client is None:
            raise AidError("not_running", "spawn is not running")
        return self._client

    async def run(self, client: aid.Client) -> None:
        self._client = client
        try:
            self._children = _CHILDREN.validate_json(await anyio.Path(self._file).read_bytes())
        except FileNotFoundError:
            self._children = {}
        except ValidationError:
            log.exception("spawn_children_unread", file=str(self._file))
            self._children = {}
        async with anyio.create_task_group() as tg:
            self._tg = tg
            for name in list(self._children):
                self._relay_soon(name)
            await anyio.sleep_forever()

    async def call(self, caller: str | None, method: str, args: dict[str, JsonValue]) -> JsonValue:
        if self._tg is None:
            raise AidError("not_running", "spawn is not running")
        if caller is None:
            raise AidError("forbidden", "only a session spawns, and dismisses its own children")
        match method:
            case "spawn":
                return await self._spawn(caller, args)
            case "dismiss":
                await self._dismiss(caller, args)
                return None
            case _:
                raise AidError("unknown_method", f"spawn has no method {method!r}")

    async def _spawn(self, caller: str, args: dict[str, JsonValue]) -> JsonValue:
        task = args.pop("task", None)
        if task is not None and not isinstance(task, str):
            raise AidError("bad_args", "task is text")
        try:
            request = SpawnRequest.model_validate(args)
        except ValidationError as error:
            raise AidError("bad_args", str(error)) from None
        client = self.client
        parent = AgentSpecAdapter.validate_python(await client.call(GetSpec(session=caller)))
        offered = [tool.name for tool in (await client.session(caller).status()).tools]
        children = sum(child.parent == caller for child in self._children.values())
        spec = child_spec(caller, parent, offered, request, depth=await self._depth(parent.parent), children=children)
        name = child_name(caller, request.label)
        if name in self._children:
            raise AidError("exists", f"{name} exists; dismiss it or pick another label")
        self._children[name] = Child(parent=caller)
        await self._save()
        try:
            session = await client.create(name, spec)
        except BaseException:
            self._children.pop(name, None)
            with anyio.CancelScope(shield=True):
                await self._save()
            raise
        self._relay_soon(name)
        if task:
            await client.send_message(name, task, sender=caller)
        tools: list[JsonValue] = [*sorted({tool.name for tool in (await session.status()).tools})]
        return {"session": name, "tools": tools}

    async def _depth(self, parent: str | None) -> int:
        """Ancestors of a session whose parent is `parent`."""
        depth = 0
        while parent is not None:
            depth += 1
            parent = AgentSpecAdapter.validate_python(await self.client.call(GetSpec(session=parent))).parent
        return depth

    async def _dismiss(self, caller: str, args: dict[str, JsonValue]) -> None:
        name = args.get("session")
        if not isinstance(name, str) or (child := self._children.get(name)) is None or child.parent != caller:
            raise AidError("forbidden", f"{name!r} is no session you spawned")
        await self.client.session(name).delete()
        await self._forget(name)

    def _relay_soon(self, name: str) -> None:
        assert self._tg is not None
        self._relays[name] = scope = anyio.CancelScope()
        self._tg.start_soon(self._relay, name, scope)

    async def _relay(self, name: str, scope: anyio.CancelScope) -> None:
        """Tell the parent about each turn the child ends, from the last one it heard about."""
        with scope:
            session = self.client.session(name)
            usage: tuple[str, Usage] | None = None
            try:
                async for entry in session.follow_history(self._children[name].relayed, idle=PRESENCE_CHECK):
                    if entry is None:
                        await session.status()
                        continue
                    match entry.item:
                        case Usage(agent=None) as turn:
                            usage = (entry.turn, parents_usage(name, turn))
                        case Output() | TurnError() as ended:
                            turn_usage = usage[1] if usage is not None and usage[0] == entry.turn else None
                            await self._tell(name, relayed_text(name, ended), turn_usage)
                            self._children[name] = self._children[name].model_copy(update={"relayed": entry.seq})
                            await self._save()
                        case _:
                            pass
            except AidError as error:
                if error.code != "not_found":
                    raise
                log.info("spawned_session_gone", session=name)
                await self._forget(name)

    async def _tell(self, name: str, text: str, usage: Usage | None) -> None:
        parent = self._children[name].parent
        try:
            await self.client.send_message(parent, text, sender=name, usage=usage)
        except AidError as error:
            log.warning("spawned_turn_undelivered", session=name, parent=parent, code=error.code)

    async def _forget(self, name: str) -> None:
        if self._children.pop(name, None) is not None:
            await self._save()
        if (scope := self._relays.pop(name, None)) is not None:
            scope.cancel()

    async def _save(self) -> None:
        async with self._saving:
            path = anyio.Path(self._file)
            await path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            staged = path.with_suffix(".tmp")
            await staged.write_bytes(_CHILDREN.dump_json(self._children))
            await staged.replace(path)
