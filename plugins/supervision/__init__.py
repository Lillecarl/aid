"""Supervision: a session that names a `supervisor` in its spec is watched over by that session, through this plugin.

It follows each supervised session's history and tells the supervisor, by message from the supervised session:
how each turn ended (its tail, open goals and loop score), a live tool loop it suspects, and a permission request
whose tool's mode involves the supervisor (`supervisor`: it answers first; `ask_supervisor`: after the person).
Either holder may hand the request on once (`ROUTES`). It keeps the stuck episode and its nudge budget, refusing a
supervisor's nudge past the budget or one that repeats the last, and asks for a person once the budget is spent.
The supervisor acts through `aid.supervisor_tools`, which call this plugin as the supervisor's own session.

Bundled and on by default (`autostart`): the daemon keeps only the links and their validation.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import anyio
import apsw
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
    QueryResult,
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
from aid.readonly import reads
from aid.removals import PLANS_FILE
from aid.spec import AcpSpec, AgentSpecAdapter, ClaudeSdkSpec, NativeSpec, PermissionMode
from aid.sql import ask

from .agent import Supervisor as Supervisor
from .classifier import Classifier as Classifier
from .classifier import ClassifierVerdict, classifier_prompt
from .loops import ToolLoopDetector
from .policy import (
    SEARCH_PAGE,
    SUPERVISION_ITEMS,
    SUPERVISION_SCORE_OUTPUTS,
    add_goal,
    loop_score,
    matches,
    nudge_repeats,
    permission_attention,
    permission_review_text,
    set_goal,
    supervision_stuck,
    supervision_text,
)
from .rulings import RULINGS_FILE, RulingStore, StepOutcome

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
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

SUPERVISOR: Final = "supervisor"
PERSON: Final = "person"
ROUTES: Final[dict[PermissionMode, tuple[str, ...]]] = {
    PermissionMode.SUPERVISOR: (SUPERVISOR, PERSON, SUPERVISOR),
    PermissionMode.ASK_SUPERVISOR: (PERSON, SUPERVISOR, PERSON),
}
"""Who holds a request at each step. Each holder but the last hands it on when it forwards or stays silent; the
last decides, and nothing moves the request further."""
RULES: Final = "rules"
"""The holder of a request `aid.readonly` calls a plain read: answered at once, no model asked."""
ALLOW_ONCE: Final = "allow_once"
REJECT_ONCE: Final = "reject_once"
CLASSIFIER: Final = "classifier"
"""The holder before the route's first: a fresh model call that allows, denies, or escalates (`classifier.py`)."""
TASK_JUDGED: Final = frozenset({"remove"})
"""Tools the classifier never judges: whether their effect is sound depends on the task, which it does not see.
Measured: it allowed a remove plan its instructions told it to escalate."""
CLASSIFIER_SUFFIX: Final = ".classifier"
"""A supervisor's classifier session is `<supervisor[:52]>.classifier`, its `parent` the supervisor."""
CLASSIFIER_WINDOW: Final = 90.0
"""Seconds the classifier may take, waiting for its session included, before the request moves on."""
SUPERVISOR_WINDOW: Final = 120.0
"""Seconds the supervisor holds a request it may forward before a person gets it: one review turn."""


@dataclass
class Route:
    """One permission request on its way between the supervisor and a person."""

    request: PermissionRequest
    holders: tuple[str, ...]
    deadline: float | None
    """`time.monotonic()` past which the worker refuses the request itself; None where aid does not know."""
    step: int = 0
    timer: anyio.CancelScope | None = None
    attention: bool = False
    """This route raised the session's attention, and clears it when the request is decided."""

    @property
    def holder(self) -> str:
        return self.holders[self.step]

    @property
    def final(self) -> bool:
        return self.step == len(self.holders) - 1


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
    routes: dict[str, Route] = field(default_factory=dict[str, Route])
    """Permission requests routed between supervisor and person, by request id."""


class Supervision(aid.Plugin):
    """Watch over sessions that name a supervisor: turn ends, live tool loops, permission routes, goals, nudges."""

    grants = frozenset(
        {
            aid.Grant.READ,
            aid.Grant.MESSAGE,
            aid.Grant.PROMPT,
            aid.Grant.PERMISSIONS,
            aid.Grant.SUPERVISE,
            aid.Grant.MANAGE,
        }
    )

    def __init__(self) -> None:
        self._watched: dict[str, Watched] = {}
        self._client: aid.Client | None = None
        self._started = 0.0
        self._tg: TaskGroup | None = None
        self._goals_dir = default_paths().plugin_dir(self.plugin_name()) / GOALS_DIR
        self._rulings: RulingStore | None = None
        self._classifier_locks: dict[str, anyio.Lock] = {}
        self._dropping: set[str] = set()
        """Classifier sessions being deleted."""
        """One request at a time per classifier session: a second prompt would find it busy."""

    async def run(self, client: aid.Client) -> None:
        self._client = client
        self._started = time.time()
        self._rulings = await RulingStore.open(default_paths().plugin_dir(self.plugin_name()) / RULINGS_FILE)
        try:
            followers: dict[str, tuple[anyio.CancelScope, str | None]] = {}
            async with anyio.create_task_group() as tg:
                self._tg = tg
                async for infos in client.follow_sessions():
                    supervisors = {info.name: info.supervisor for info in infos}
                    for info in infos:
                        orphan = info.name.endswith(CLASSIFIER_SUFFIX) and info.parent not in supervisors
                        if orphan and info.name not in self._dropping:
                            # Lists keep arriving while the delete runs: one delete per orphan.
                            self._dropping.add(info.name)
                            tg.start_soon(self._drop_classifier, info.name)
                    for gone in followers.keys() - supervisors.keys():
                        followers.pop(gone)[0].cancel()
                        await self._forget(gone)
                        (self._goals_dir / f"{gone}.json").unlink(missing_ok=True)
                    for name, supervisor in supervisors.items():
                        since: int | None = None
                        if (held := followers.get(name)) is not None:
                            if held[1] == supervisor:
                                continue
                            # A supervisor set or changed later (`SetSupervisor`): from now, never what was told.
                            held[0].cancel()
                            await self._forget(name)
                            since = await self._latest(name)
                        followers[name] = (scope := anyio.CancelScope(), supervisor)
                        tg.start_soon(self._follow, name, scope, since)
        finally:
            await self._rulings.aclose()
            self._rulings = None

    async def _forget(self, name: str) -> None:
        if (watched := self._watched.pop(name, None)) is not None:
            await self._drop_routes(watched)

    async def _drop_routes(self, watched: Watched) -> None:
        """Every request ends with its turn: stop their timers without telling anyone."""
        routes = list(watched.routes.values())
        watched.routes.clear()
        for route in routes:
            if route.timer is not None:
                route.timer.cancel()
            await self._record(lambda r, route=route: r.ended(route.request, route.step, None))

    async def _record(self, write: Callable[[RulingStore], Awaitable[None]]) -> None:
        """Keep a ruling. A failed write is logged: supervision goes on without its record."""
        if (rulings := self._rulings) is None:
            return
        try:
            await write(rulings)
        except apsw.Error:
            log.exception("ruling_unrecorded")

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

    async def _follow(self, name: str, scope: anyio.CancelScope, since: int | None) -> None:
        """Follow a supervised session's history after `since`; None: after what whoever ran before told."""
        with scope:
            try:
                watched = await self._watch(name)
                if watched.spec.supervisor is None:
                    return
                session = self.client.session(name)
                async for entry in session.follow_history(await self._told(name) if since is None else since):
                    await self._observe(watched, entry)
            except AidError as error:
                # The session went between its listing and here: the next list drops it.
                log.info("supervision_follow_ended", session=name, error=error.message)

    async def _latest(self, name: str) -> int:
        """The session's newest entry, -1 for none or a session gone meanwhile."""
        try:
            page = await self.client.session(name).history(limit=1)
        except AidError:
            return -1
        return page.entries[-1].seq if page.entries else -1

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
                await self._route(watched, item, entry)
            case PermissionDecision():
                await self._settle(watched, item.request_id, item)
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
        await self._drop_routes(watched)
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
    def _worker_bound(spec: AgentSpec, request: PermissionRequest) -> float | None:
        """Seconds the worker waits on this request itself: a handover past it is moot. An `ask_user` question
        with a recommended option ends at the autoselect instead. None when the worker's bound is aid-unknown
        (interactive Claude)."""
        match spec:
            case AcpSpec(permission_timeout=timeout) | ClaudeSdkSpec(permission_timeout=timeout):
                return timeout
            case NativeSpec(permission_timeout=timeout, ask_autoselect_after=autoselect):
                if request.tool_name == "ask_user" and any(o.recommended for o in request.options):
                    return min(timeout, autoselect)
                return timeout
            case _:
                return None

    async def _route(self, watched: Watched, request: PermissionRequest, entry: HistoryEntry) -> None:
        """Start a request on its way when its mode involves the supervisor. claude-tty's requests name no mode:
        the spec's mode for the tool stands in."""
        spec = watched.spec
        mode = request.mode or (spec.tool_permissions.get(request.tool_name) if request.tool_name else None)
        holders = ROUTES.get(mode) if mode is not None else None
        if self._tg is None or (supervisor := spec.supervisor) is None or mode is None or holders is None:
            return
        kinds = {o.kind for o in request.options}
        if reads(request.tool_name, request.input) and ALLOW_ONCE in kinds:
            holders = (RULES,)
        elif (
            spec.supervisor_classifier is not None
            and request.tool_name not in TASK_JUDGED
            and {ALLOW_ONCE, REJECT_ONCE} <= kinds
        ):
            holders = (CLASSIFIER, *holders)
        await self._settle(watched, request.request_id)
        bound = self._worker_bound(spec, request)
        route = Route(request, holders, None if bound is None else time.monotonic() + bound)
        watched.routes[request.request_id] = route
        await self._record(
            lambda r: r.started(
                request, session=watched.name, turn=entry.turn, seq=entry.seq, supervisor=supervisor, mode=mode
            )
        )
        await self._enter(watched, route, None)

    async def _enter(self, watched: Watched, route: Route, note: str | None) -> None:
        """Give the request to its current holder, and arm the handover to the next one unless this is the last."""
        spec, request = watched.spec, route.request
        await self._record(lambda r: r.entered(request.request_id, route.step, route.holder))
        if route.holder == RULES:
            await self._allow_read(watched, request)
            return
        if route.holder == CLASSIFIER:
            if self._tg is not None:
                route.timer = anyio.CancelScope()
                self._tg.start_soon(self._classify, watched, route, route.timer)
            return
        if route.holder == SUPERVISOR:
            await self._tell(watched, permission_review_text(watched.name, request, final=route.final, note=note))
            wait = SUPERVISOR_WINDOW
        else:
            if route.step > 0:
                route.attention = True
                await self.client.call(
                    SetAttention(session=watched.name, text=permission_attention(watched.name, request, note))
                )
            wait = spec.supervisor_permission_timeout
            # Presence doubles as the window: a person-authored entry this recent keeps the full wait.
            if not await self._person_recent(watched.name, wait):
                wait = spec.supervisor_permission_absent_timeout
        if route.final or self._tg is None:
            return
        if route.deadline is not None and time.monotonic() + wait >= route.deadline:
            log.info("handover_moot", session=watched.name, request=request.request_id, wait=wait)
            return
        route.timer = anyio.CancelScope()
        self._tg.start_soon(self._hand_on_after, watched, route, wait, route.timer)

    async def _allow_read(self, watched: Watched, request: PermissionRequest) -> None:
        """Allow a plain read once, as this plugin. A request gone meanwhile was answered by someone else."""
        option = next(o.option_id for o in request.options if o.kind == ALLOW_ONCE)
        try:
            await self.client.call(
                AnswerPermission(session=watched.name, request_id=request.request_id, option_id=option)
            )
        except AidError as error:
            log.info("read_answer_failed", session=watched.name, request=request.request_id, error=error.message)

    async def _classify(self, watched: Watched, route: Route, scope: anyio.CancelScope) -> None:
        """Ask the classifier, then act on its verdict: answer, or hand the request on with its reason. A decision
        elsewhere cancels `scope`; silence or a failure past `CLASSIFIER_WINDOW` hands it on too."""
        verdict: ClassifierVerdict | None = None
        problem = "the classifier did not answer in time"
        with scope, anyio.move_on_after(CLASSIFIER_WINDOW):
            try:
                verdict = await self._ask_classifier(watched, route.request)
            except (AidError, ValidationError) as error:
                problem = f"the classifier failed: {error}"
                log.info("classifier_failed", session=watched.name, error=str(error))
        if scope.cancel_called:
            return
        route.timer = None
        request = route.request
        try:
            match verdict:
                case ClassifierVerdict(verdict="allow"):
                    option = next(o.option_id for o in request.options if o.kind == ALLOW_ONCE)
                    await self.client.call(
                        AnswerPermission(session=watched.name, request_id=request.request_id, option_id=option)
                    )
                case ClassifierVerdict(verdict="deny", reason=reason):
                    option = next(o.option_id for o in request.options if o.kind == REJECT_ONCE)
                    await self.client.call(
                        AnswerPermission(
                            session=watched.name, request_id=request.request_id, option_id=option, text=reason
                        )
                    )
                case ClassifierVerdict(reason=reason):
                    await self._hand_on(watched, route, StepOutcome.FORWARDED, f"classifier: {reason}")
                case None:
                    await self._hand_on(watched, route, StepOutcome.SILENT, problem)
        except AidError as error:
            # Answered meanwhile, or the session went: the request ends with its turn.
            log.info("classifier_verdict_unused", session=watched.name, error=error.message)

    async def _ask_classifier(self, watched: Watched, request: PermissionRequest) -> ClassifierVerdict:
        """One fresh turn of the supervisor's classifier session, made on first use."""
        supervisor, classifier = watched.spec.supervisor, watched.spec.supervisor_classifier
        if supervisor is None or classifier is None:
            raise AidError("no_classifier", f"{watched.name!r} names no supervisor or classifier")
        name = await self._classifier_session(supervisor, classifier)
        session = self.client.session(name)
        async with self._classifier_locks.setdefault(name, anyio.Lock()):
            try:
                result = await session.run(classifier_prompt(watched.name, watched.spec.cwd, request))
            except anyio.get_cancelled_exc_class():
                with anyio.CancelScope(shield=True):
                    with contextlib.suppress(AidError):
                        await session.cancel()
                raise
        return ClassifierVerdict.model_validate(result.output)

    async def _classifier_session(self, supervisor: str, classifier: str) -> str:
        """The supervisor's classifier session, running `classifier`; made, or remade for another classifier."""
        name = f"{supervisor[:52]}{CLASSIFIER_SUFFIX}"
        agent, target = (None, classifier) if ":" in classifier else (classifier, None)
        try:
            current = AgentSpecAdapter.validate_python(await self.client.call(GetSpec(session=name)))
        except AidError:
            current = None
        if isinstance(current, NativeSpec) and (current.agent, current.target) == (agent, target):
            return name
        if current is not None:
            await self.client.session(name).delete()
        base = AgentSpecAdapter.validate_python(await self.client.call(GetSpec(session=supervisor)))
        native = base if isinstance(base, NativeSpec) else None
        spec = NativeSpec(
            cwd=base.cwd,
            agent=agent,
            target=target,
            python_path=native.python_path if native is not None else [],
            env=native.env if native is not None else {},
            parent=supervisor,
            permission=PermissionMode.DENY,
            aid_tools=False,
        )
        await self.client.create(name, spec)
        return name

    async def _drop_classifier(self, name: str) -> None:
        """A classifier session whose supervisor went."""
        try:
            await self.client.session(name).delete()
        except AidError as error:
            if error.code != "not_found":
                log.info("classifier_drop_failed", session=name, error=error.message)
        finally:
            self._dropping.discard(name)

    async def _hand_on_after(self, watched: Watched, route: Route, wait: float, timer: anyio.CancelScope) -> None:
        """The holder stayed silent: the next one gets the request. A decision or a forward cancels the timer."""
        with timer:
            await anyio.sleep(wait)
        if not timer.cancel_called:
            route.timer = None
            try:
                await self._hand_on(watched, route, StepOutcome.SILENT, None)
            except AidError as error:
                # The session or its supervisor went meanwhile; the request ends with its turn.
                log.info("handover_failed", session=watched.name, error=error.message)

    async def _hand_on(self, watched: Watched, route: Route, how: StepOutcome, note: str | None) -> None:
        """Move a request still waiting to its next holder: `how` says why the holder let it go."""
        request_id = route.request.request_id
        if watched.routes.get(request_id) is not route or route.final:
            return
        if route.timer is not None:
            route.timer.cancel()
            route.timer = None
        status = await self.client.session(watched.name).status()
        if not any(p.request_id == request_id for p in status.permissions):
            await self._settle(watched, request_id)
            return
        await self._record(lambda r: r.left(request_id, route.step, how, note))
        route.step += 1
        await self._enter(watched, route, note)

    async def _settle(self, watched: Watched, request_id: str, decision: PermissionDecision | None = None) -> None:
        """The request is decided, or stopped waiting: stop its timer, and clear the attention it raised."""
        if (route := watched.routes.pop(request_id, None)) is None:
            return
        if route.timer is not None:
            route.timer.cancel()
        await self._record(lambda r: r.ended(route.request, route.step, decision))
        if route.attention:
            await self.client.call(SetAttention(session=watched.name, text=None))

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
        if method == "query":
            session = given.get("session")
            result = await self._query(caller, None if session is None else str(session), str(given.get("sql") or ""))
            return result.model_dump(mode="json")
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
            case "entry":
                seq = int(given.get("seq") or 0)
                page = await session.history(before=seq + 1, limit=1)
                if not page.entries or page.entries[0].seq != seq:
                    raise AidError("no_entry", f"{name!r} has no history entry {seq}")
                return page.entries[0].model_dump(mode="json")
            case "forward_permission":
                request_id = str(given.get("request_id"))
                if (route := watched.routes.get(request_id)) is None:
                    raise AidError("no_pending", f"{name!r} waits on no permission request {request_id} for you")
                if route.holder != SUPERVISOR or route.final:
                    raise AidError("not_forwardable", "a person had this request already: decide it yourself")
                note = given.get("note")
                await self._hand_on(watched, route, StepOutcome.FORWARDED, None if note is None else str(note))
                return None
            case "answer_permission":
                request_id = str(given.get("request_id"))
                status = await session.status()
                pending = next((p for p in status.permissions if p.request_id == request_id), None)
                if pending is None:
                    raise AidError("no_pending", f"{name!r} waits on no permission request {request_id}")
                if request_id not in watched.routes:
                    raise AidError(
                        "person_only",
                        f"{pending.tool_name!r} asks a person in {name!r}; its mode involves no supervisor",
                    )
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

    async def _query(self, caller: str | None, session: str | None, sql: str) -> QueryResult:
        """Read-only SQL over the rulings, and with `session` over its removal plans. A supervisor names only a
        session it supervises, and must supervise one to ask at all; the operator (None) names any."""
        if not sql.strip():
            raise AidError("empty_query", "a query needs SQL")
        attach = {"supervision": default_paths().plugin_dir(self.plugin_name()) / RULINGS_FILE}
        if session is not None:
            if caller is None:
                await self.client.call(GetSpec(session=session))
            elif (await self._watch(session)).spec.supervisor != caller:
                raise AidError("not_supervised", f"{caller!r} supervises no session {session!r}")
            attach["session"] = default_paths().session_dir(session) / PLANS_FILE
        elif caller is not None and not any(w.spec.supervisor == caller for w in self._watched.values()):
            raise AidError("not_supervising", f"{caller!r} supervises no session")
        return await ask(attach, sql)

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
