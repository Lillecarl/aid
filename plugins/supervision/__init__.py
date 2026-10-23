"""Supervision: a session that names a `supervisor` in its spec is watched over by that session, through this plugin.

It follows each supervised session's history and tells the supervisor, by message from the supervised session:
how each turn ended (its tail, open goals and loop score), a live tool loop it suspects, and a permission request
the person left waiting on an allow-listed tool. It keeps the stuck episode and its nudge budget, refusing a
supervisor's nudge past the budget or one that repeats the last, and asks for a person once the budget is spent.
The supervisor acts through `aid.supervisor_tools`, which call this plugin as the supervisor's own session.

Bundled and on by default (`autostart`): the daemon keeps only the links and their validation.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
import structlog
from pydantic import TypeAdapter, ValidationError

import aid
from aid.paths import default_paths
from aid.protocol import (
    AidError,
    AnswerPermission,
    Cancel,
    GetSpec,
    GoalsAdapter,
    GoalState,
    MessageEntry,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    PublishSupervision,
    SearchResults,
    SetAttention,
    SupervisionEvent,
    SupervisionFiring,
    SupervisionGoals,
    SupervisionNudge,
    ToolCall,
    TurnError,
    clip,
)
from aid.spec import AcpSpec, AgentSpecAdapter, NativeSpec

from .loops import ToolLoopDetector
from .policy import (
    SEARCH_PAGE,
    SUPERVISION_ITEMS,
    SUPERVISION_SCORE_OUTPUTS,
    add_goal,
    loop_score,
    matches,
    nudge_repeats,
    permission_backstop_text,
    set_goal,
    supervision_stuck,
    supervision_text,
)

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.abc import TaskGroup
    from pydantic import JsonValue

    from aid.protocol import Goal, HistoryEntry
    from aid.spec import AgentSpec

    from .loops import LoopEvidence

log = structlog.get_logger(__name__)

GOALS_DIR: Final = "goals"
LEGACY_GOALS_FILE: Final = "goals.json"
"""Goals the daemon kept beside a session's history, read until this plugin first writes that session's goals."""

TURN_ITEMS: Final = (PromptEntry, ToolCall, PermissionRequest, PermissionDecision, Output, TurnError)
"""Items that belong to a turn: a message's entry carries the message's id, and a start its own."""

_ARGS: Final = TypeAdapter(dict[str, str | int | None])


@dataclass
class Watched:
    """One supervised session, as this plugin follows it."""

    name: str
    spec: AgentSpec
    turn: str | None = None
    """The turn the latest turn entry belongs to."""
    fired: str | None = None
    """The last turn whose end the supervisor heard about: an end fires once."""
    stuck_turns: int = 0
    """Consecutive turn ends scoring stuck: the stuck episode's length."""
    nudges: int = 0
    """Supervisor nudges accepted this stuck episode."""
    last_nudge: str | None = None
    """The last accepted nudge's text: a nudge too like it is refused."""
    outputs: deque[Output] | None = None
    """Loaded lazily from history, then kept from entries; independent of the clipped review tail."""
    loops: ToolLoopDetector = field(default_factory=ToolLoopDetector)
    loop_suspected: bool = False
    interrupted: bool = False
    backstops: dict[str, anyio.CancelScope] = field(default_factory=dict[str, anyio.CancelScope])
    """Permission backstops still waiting, by request id."""


class Supervision(aid.Plugin):
    """Watch over sessions that name a supervisor: turn ends, live tool loops, permission backstops, goals, nudges."""

    grants = frozenset(
        {aid.Grant.READ, aid.Grant.MESSAGE, aid.Grant.PROMPT, aid.Grant.PERMISSIONS, aid.Grant.SUPERVISE}
    )

    def __init__(self) -> None:
        self._watched: dict[str, Watched] = {}
        self._client: aid.Client | None = None
        self._started = 0.0
        self._tg: TaskGroup | None = None
        self._goals_dir = default_paths().plugin_dir(self.plugin_name()) / GOALS_DIR

    async def run(self, client: aid.Client) -> None:
        self._client = client
        self._started = time.time()
        followers: dict[str, anyio.CancelScope] = {}
        async with anyio.create_task_group() as tg:
            self._tg = tg
            async for infos in client.follow_sessions():
                names = {info.name for info in infos}
                for gone in followers.keys() - names:
                    followers.pop(gone).cancel()
                    self._forget(gone)
                    (self._goals_dir / f"{gone}.json").unlink(missing_ok=True)
                for name in names - followers.keys():
                    followers[name] = scope = anyio.CancelScope()
                    tg.start_soon(self._follow, name, scope)

    def _forget(self, name: str) -> None:
        if (watched := self._watched.pop(name, None)) is not None:
            for scope in watched.backstops.values():
                scope.cancel()

    @property
    def client(self) -> aid.Client:
        if self._client is None:
            raise RuntimeError("supervision is not running")
        return self._client

    async def _watch(self, name: str) -> Watched:
        if (watched := self._watched.get(name)) is None:
            spec = AgentSpecAdapter.validate_python(await self.client.call(GetSpec(session=name)))
            watched = self._watched.setdefault(name, Watched(name, spec))
        return watched

    async def _follow(self, name: str, scope: anyio.CancelScope) -> None:
        with scope:
            try:
                watched = await self._watch(name)
                if watched.spec.supervisor is None:
                    return
                session = self.client.session(name)
                async for entry in session.follow_history(await self._told(name)):
                    await self._observe(watched, entry)
            except AidError as error:
                # The session went between its listing and here: the next list drops it.
                log.info("supervision_follow_ended", session=name, error=error.message)

    async def _told(self, name: str) -> int:
        """The last entry recorded before this plugin started: whoever ran then told the supervisor about it."""
        session = self.client.session(name)
        before: int | None = None
        while True:
            page = await session.history(before=before, limit=SEARCH_PAGE)
            for entry in reversed(page.entries):
                if entry.at < self._started:
                    return entry.seq
            if not page.has_older or not page.entries:
                return -1
            before = page.entries[0].seq

    async def _observe(self, watched: Watched, entry: HistoryEntry) -> None:
        item = entry.item
        if isinstance(item, TURN_ITEMS) and entry.turn != watched.turn:
            watched.turn = entry.turn
            watched.loops = ToolLoopDetector()
            watched.loop_suspected = watched.interrupted = False
        match item:
            case PromptEntry():
                # A person's prompt is a new instruction, not another automatic recovery attempt.
                watched.stuck_turns = watched.nudges = 0
                watched.last_nudge = None
            case ToolCall():
                if (evidence := watched.loops.observe(item)) is not None:
                    watched.loop_suspected = True
                    await self._send_loop_review(watched, entry.turn, evidence)
            case PermissionRequest():
                await self._arm_backstop(watched, item)
            case PermissionDecision():
                if (scope := watched.backstops.pop(item.request_id, None)) is not None:
                    scope.cancel()
            case Output():
                if watched.outputs is not None:
                    watched.outputs.append(item)
                await self._turn_ended(watched, entry, "done")
            case TurnError():
                await self._turn_ended(watched, entry, item.code)
            case _:
                pass

    async def _tell(self, watched: Watched, text: str) -> bool:
        """Message the supervisor from the supervised session, as a firing always was. False when it is gone."""
        supervisor = watched.spec.supervisor
        if supervisor is None:
            return False
        try:
            await self.client.send_message(supervisor, text, sender=watched.name)
        except AidError as error:
            log.info("supervisor_gone", session=watched.name, supervisor=supervisor, error=error.message)
            return False
        return True

    async def _publish(self, watched: Watched, **change: object) -> None:
        if watched.spec.supervisor is not None:
            event = SupervisionEvent.model_validate(
                {"session": watched.name, "supervisor": watched.spec.supervisor, **change}
            )
            await self.client.call(PublishSupervision(event=event))

    async def _turn_ended(self, watched: Watched, entry: HistoryEntry, reason: str) -> None:
        """Score the turn's tail for stuckness, then hand it to the supervisor."""
        for scope in watched.backstops.values():
            scope.cancel()
        watched.backstops.clear()
        if watched.fired == entry.turn or (supervisor := watched.spec.supervisor) is None:
            return
        watched.fired = entry.turn
        session = self.client.session(watched.name)
        page = await session.history(before=entry.seq + 1, limit=SUPERVISION_ITEMS)
        outputs = await self._recent_outputs(watched, entry.seq)
        interrupted_loop = (
            watched.loop_suspected and watched.interrupted and bool(outputs) and outputs[-1].stop_reason == "cancelled"
        )
        score = 1.0 if interrupted_loop else loop_score(outputs)
        stuck, nudges, attention = supervision_stuck(
            watched.name,
            supervisor,
            watched.stuck_turns,
            watched.nudges,
            watched.spec.supervisor_nudge_budget,
            score,
        )
        watched.stuck_turns, watched.nudges = stuck, nudges
        if stuck == 0 and nudges == 0:
            watched.last_nudge = None
        await self.client.call(SetAttention(session=watched.name, text=attention))
        goals = [goal for goal in await self._read_goals(watched.name) if goal.state is GoalState.OPEN]
        text = supervision_text(watched.name, reason, page, goals, outputs=outputs)
        if interrupted_loop:
            text += (
                "\nThe supervisor interrupted a suspected live tool loop. This remains the same stuck episode; "
                "interruption alone is not evidence of progress and does not renew the nudge budget."
            )
        if await self._tell(watched, text):
            firing = SupervisionFiring(turn=entry.turn, reason=reason, score=score, stuck_turns=stuck)
            await self._publish(watched, firing=firing)

    async def _recent_outputs(self, watched: Watched, seq: int) -> list[Output]:
        """Restore the small scoring window once, up to `seq`. Tool-heavy turns may push the previous answer
        thousands of entries behind the review tail; later entries keep it current."""
        if watched.outputs is None:
            session = self.client.session(watched.name)
            newest: list[Output] = []
            before = seq + 1
            while len(newest) < SUPERVISION_SCORE_OUTPUTS:
                page = await session.history(before=before, limit=SEARCH_PAGE)
                newest.extend(entry.item for entry in reversed(page.entries) if isinstance(entry.item, Output))
                if not page.has_older or not page.entries:
                    break
                before = page.entries[0].seq
            watched.outputs = deque(reversed(newest[:SUPERVISION_SCORE_OUTPUTS]), maxlen=SUPERVISION_SCORE_OUTPUTS)
        return list(watched.outputs)

    async def _send_loop_review(self, watched: Watched, turn: str, evidence: LoopEvidence) -> None:
        """Detection never interrupts the agent; only the supervisor can decide whether this repetition is useful."""
        text = (
            f"Suspected tool loop in supervised session {watched.name!r}, live turn {turn!r}.\n"
            f"{evidence.describe()}\n"
            "Read supervised_history/supervised_search to judge whether work is progressing. "
            "If it is useful, leave it running. If it is stuck, interrupt(session, turn) stops only this turn; "
            "send_message can then give corrective instructions. A message alone waits until the turn ends."
        )
        if await self._tell(watched, text):
            firing = SupervisionFiring(
                turn=turn, reason="suspected_tool_loop", score=None, stuck_turns=watched.stuck_turns
            )
            await self._publish(watched, firing=firing)

    async def _person_recent(self, name: str, window: float) -> bool:
        """A person-authored entry in the last `window` seconds: their own message (sender None), or their
        answer to a permission request. Agent and daemon senders name themselves, so None is a person."""
        cutoff = time.time() - window
        session = self.client.session(name)
        before: int | None = None
        while True:
            page = await session.history(before=before, limit=SEARCH_PAGE)
            for entry in reversed(page.entries):
                if entry.at < cutoff:
                    return False
                match entry.item:
                    case MessageEntry(sender=None):
                        return True
                    case PermissionDecision(by=PermissionDecider.PERSON):
                        return True
                    case _:
                        pass
            if not page.has_older or not page.entries:
                return False
            before = page.entries[0].seq

    @staticmethod
    def _backstop_bound(spec: AgentSpec, request: PermissionRequest) -> float | None:
        """Seconds the worker waits on this request itself: the backstop arms only before it. An `ask_user`
        question with a recommended option ends at the autoselect instead. None when the worker's bound is
        aid-unknown (interactive Claude)."""
        match spec:
            case AcpSpec(permission_timeout=timeout):
                return timeout
            case NativeSpec(permission_timeout=timeout, ask_autoselect_after=autoselect):
                if request.tool_name == "ask_user" and any(o.recommended for o in request.options):
                    return min(timeout, autoselect)
                return timeout
            case _:
                return None

    async def _arm_backstop(self, watched: Watched, request: PermissionRequest) -> None:
        """Wake the supervisor over an unanswered permission request, once the person had their window: the
        full timeout with a person around, the shortened one without. Skipped when the tool is not
        allow-listed, and when the worker's own timeout answers first."""
        spec = watched.spec
        if self._tg is None or spec.supervisor is None or request.tool_name not in spec.supervisor_permissions:
            return
        if (old := watched.backstops.pop(request.request_id, None)) is not None:
            old.cancel()
        wait = spec.supervisor_permission_timeout
        # Presence doubles as the window: a person-authored entry this recent keeps the full wait.
        if not await self._person_recent(watched.name, wait):
            wait = spec.supervisor_permission_absent_timeout
        if (bound := self._backstop_bound(spec, request)) is not None and wait >= bound:
            log.info("backstop_moot", session=watched.name, request=request.request_id, wait=wait, bound=bound)
            return
        scope = watched.backstops[request.request_id] = anyio.CancelScope()
        self._tg.start_soon(self._await_backstop, watched, request.request_id, wait, scope)

    async def _await_backstop(self, watched: Watched, request_id: str, wait: float, scope: anyio.CancelScope) -> None:
        """Outwait the person, then hand the still-waiting request to the supervisor. A decision first cancels
        the scope, so the message only ever carries a live request."""
        with scope:
            await anyio.sleep(wait)
            if watched.backstops.get(request_id) is not scope:
                return
            del watched.backstops[request_id]
            status = await self.client.session(watched.name).status()
            pending = next((p for p in status.permissions if p.request_id == request_id), None)
            if pending is not None:
                await self._tell(watched, permission_backstop_text(watched.name, pending))

    async def on_message(self, to: str, sender: str | None, text: str) -> None:
        """A supervisor's nudge goes out within budget and unlike the last one; the episode counts it."""
        if sender is None:
            return
        watched = await self._watch(to)
        if watched.spec.supervisor != sender:
            return
        budget = watched.spec.supervisor_nudge_budget
        if watched.nudges >= budget:
            raise AidError(
                "nudge_budget", f"{to!r} had its {budget} nudges this episode; further stuck, it needs a person"
            )
        if nudge_repeats(watched.last_nudge, text):
            raise AidError(
                "repeated_nudge", "that nudge says what the last one said; say it differently or stay silent"
            )
        watched.nudges += 1
        watched.last_nudge = text
        await self._publish(watched, nudge=SupervisionNudge(text=clip(text), nudges=watched.nudges))

    async def call(self, caller: str | None, method: str, args: dict[str, JsonValue]) -> JsonValue:
        """The supervisor's tools (`aid.supervisor_tools`): each names a session the caller supervises."""
        try:
            given = _ARGS.validate_python(args)
        except ValidationError as error:
            raise AidError("invalid_args", str(error)) from None
        name = str(given.get("session") or "")
        watched = await self._watch(name)
        if caller is None or watched.spec.supervisor != caller:
            raise AidError("not_supervised", f"{caller!r} supervises no session {name!r}")
        session = self.client.session(name)
        match method:
            case "history":
                page = await session.history(limit=int(given.get("limit") or 50))
                return page.model_dump(mode="json")
            case "search":
                return (
                    await self._search(name, str(given.get("query") or ""), int(given.get("limit") or 20))
                ).model_dump(mode="json")
            case "goals":
                return [goal.model_dump(mode="json") for goal in await self._read_goals(name)]
            case "goal_add":
                text = str(given.get("text") or "")
                if not text.strip():
                    raise AidError("empty_goal", "a goal needs text")
                goals = await self._read_goals(name)
                goal = add_goal(goals, text)
                await self._write_goals(name, goals)
                await self._publish(watched, goals=SupervisionGoals(action="added", goal=goal))
                return goal.model_dump(mode="json")
            case "goal_set":
                state = GoalState(str(given.get("state")))
                if state is GoalState.OPEN:
                    raise AidError("invalid_args", "a goal is set done or dropped")
                goals = await self._read_goals(name)
                goal = set_goal(goals, str(given.get("goal_id")), state)
                await self._write_goals(name, goals)
                action = "done" if goal.state is GoalState.DONE else "dropped"
                await self._publish(watched, goals=SupervisionGoals(action=action, goal=goal))
                return goal.model_dump(mode="json")
            case "interrupt":
                turn = str(given.get("turn"))
                if not (await session.status()).busy:
                    raise AidError("stale_turn", "the reviewed turn is no longer running")
                # Before the cancel: the turn's cancelled Output can reach this plugin before the cancel's reply.
                interrupted = watched.interrupted
                watched.interrupted = True
                try:
                    await self.client.call(Cancel(session=name, turn=turn))
                except AidError:
                    watched.interrupted = interrupted
                    raise
                return None
            case "answer_permission":
                request_id = str(given.get("request_id"))
                status = await session.status()
                pending = next((p for p in status.permissions if p.request_id == request_id), None)
                if pending is None:
                    raise AidError("no_pending", f"{name!r} waits on no permission request {request_id}")
                if pending.tool_name not in watched.spec.supervisor_permissions:
                    raise AidError("unlisted_tool", f"{caller!r} may not answer {pending.tool_name!r} for {name!r}")
                option = given.get("option_id")
                text = given.get("text")
                await self.client.call(
                    AnswerPermission(
                        session=name,
                        request_id=request_id,
                        option_id=None if option is None else str(option),
                        text=None if text is None else str(text),
                        supervisor=caller,
                    )
                )
                return None
            case _:
                raise AidError("unknown_method", f"supervision has no method {method!r}")

    async def _search(self, name: str, query: str, limit: int) -> SearchResults:
        """Entries whose rendered line holds `query`, newest first, over the whole history."""
        if not query.strip():
            raise AidError("empty_query", "a supervision search needs a query")
        session = self.client.session(name)
        found: list[HistoryEntry] = []
        before: int | None = None
        while True:
            page = await session.history(before=before, limit=SEARCH_PAGE)
            found.extend(reversed(matches(page.entries, query)))
            if not page.has_older or not page.entries:
                break
            before = page.entries[0].seq
        return SearchResults(entries=found[:limit], total=len(found))

    def _goals_file(self, name: str) -> Path:
        return self._goals_dir / f"{name}.json"

    async def _read_goals(self, name: str) -> list[Goal]:
        """The session's goals, or none: a missing file means none yet; an unreadable one is logged and
        skipped, so a corrupt file never breaks supervision."""
        for path in (self._goals_file(name), default_paths().session_dir(name) / LEGACY_GOALS_FILE):
            try:
                data = await anyio.Path(path).read_bytes()
            except FileNotFoundError:
                continue
            except OSError:
                log.exception("goals_unreadable", session=name)
                return []
            try:
                return GoalsAdapter.validate_json(data)
            except ValidationError:
                log.exception("goals_unreadable", session=name)
                return []
        return []

    async def _write_goals(self, name: str, goals: list[Goal]) -> None:
        path = anyio.Path(self._goals_file(name))
        await path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        await tmp.write_bytes(GoalsAdapter.dump_json(goals))
        await tmp.rename(path)
