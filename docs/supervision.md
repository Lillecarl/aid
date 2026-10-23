# Supervision: overseer agents over worker sessions

Tracks Lillecarl/aid#25. The Ralph shape (loop until done, never unbounded, verify against goals)
with a judging driver instead of a bash loop: a cheap supervisor session watches worker sessions,
nudges them with human-like encouragements, tracks goals, and answers allow-listed permissions and
questions when no person is around.

## Relationship

- `supervisor: str | None` on the supervised session's spec. Restart-proof via the persisted spec file.
- Validated when the supervised session starts: the supervisor exists, no self-watch, no cycles.
  Links form a DAG; a supervisor may itself be supervised, but nobody gets one by default.
- Deleting a supervised session unlinks; firing for a gone session skips and logs.

## Firing

- The bundled `supervision` plugin (`plugins/supervision/`) follows each supervised session's history and
  fires the supervisor at every turn end it records (`Output` or `TurnError`), and on live tool-loop
  suspicions. It messages the supervisor as the supervised session, so delivery is the accept→inbox→wake
  path (idle: new turn, live: next turn), and a reply is a nudge.
- Payload is small by design (cheap model): last-N history items clipped (order ~10 items / few k
  chars), open goals only, loop score as one line (difflib ratio over recent normalized Outputs,
   threshold ~0.85 — advisory, computed by the plugin at turn end). Silence ("nothing to
  do, no nudge") is a valid supervisor verdict.
- Turn-end similarity compares the latest two completed answers, separately from the clipped review tail.
  Tool-heavy turns do not hide the previous answer. Cancelled/missing answers reset the signal; an old
  repeated pair cannot conceal subsequent progress.

## Live loop review

- `supervision.loops.ToolLoopDetector` watches the tool calls the session's history records. Three
  identical completed calls, or three cycles of a short sequence (up to eight calls), raise a suspicion.
  Arguments, status and recorded output must all match; missing telemetry breaks the sequence rather than
  being interpreted as an unchanged result. The detector retains 24 completed steps and at most 128 pending
  calls. A repeated episode notifies once, including across rotations of an A/B cycle.
- The plugin immediately messages the supervisor with the live turn id, the repeated sequence and tool
  call ids for history inspection. `supervision/` publishes `reason=suspected_tool_loop` for observation;
  the reliable inbox drives review. The detector never cancels a turn or claims repetition proves a loop.
- The supervisor may leave useful repetition running, use `interrupt(session, turn)` when it judges the
  agent stuck, and use `send_message` for corrective instructions. A message alone waits until turn end.
  The plugin checks the supervision link and that the session is busy; the worker checks the turn again
  (`Cancel(turn=…)`). A delayed review cannot cancel newer work.
- Semantic/embedding signals are not implemented yet. A bounded per-session vector window can complement
  exact fingerprints without requiring a vector database. This first detector needs tool telemetry; it does
  not detect silent model stalls.
- No model-request budget is imposed by default. An explicit `max_requests` is a resource budget, not a
  loop detector. The native backend preserves available history and usage on failure/cancellation,
  closing unanswered tool calls as interrupted without replaying their potentially applied effects.

## Supervisor session

- Always native, pinned cheap model, short verdicts, dedicated `supervisor_tools`:
  `supervised_history(session, limit)`, `supervised_search(session, query)`, `goals(session)`,
  `goal_add` / `goal_done` / `goal_drop`, `answer_permission(request_id, option, text)`, plus the
  turn-scoped `interrupt(session, turn)` and builtin `send_message` for nudges. Each firing names its
  subject, so one supervisor watches N. The tools call the plugin (`CallPlugin`); the daemon names the
  caller from the session's connection, and the plugin enforces the supervision link.
- Nudges ride `send_message`: attributed, replyable, zero new transport.

## Goals

- `goals/<session>.json` in the plugin's directory. The plugin reads open items into every firing payload;
  supervisor tools mutate. Goals are the source of done-ness; a DONE-style token in
  supervised output is a hint the supervisor verifies, never an event.

## Permission backstop

- `permit` and `ask_user` share `PermissionWaits`, so one answer tool covers allow/deny and questions.
- Person first, supervisor after a configurable timeout, shortened when no person-authored entry
  (message, permission answer) is recent. First resolution wins; ask_user's recommended-auto keeps
  working unchanged beside it.
- Allow-listed tool names only in the supervised spec; unlisted tools never reach the supervisor.
  The daemon stamps `by=supervisor` itself (new `PermissionDecider`, worker-never-claimed, same
  rule as `by=plugin`).

## Guards

- Nudge budget per supervised episode; never send a nudge similar to the last (the same difflib
  check turned around); K stuck turns after K nudges stops nudging and raises `attention` to the
  person. Every firing and nudge is metered; nothing loops without a cap.
- Interrupting a suspected live loop retains the stuck episode. Missing/cancelled answers do not renew
  the recovery budget; evidence of progress or an explicit new prompt does. Otherwise a supervisor could
  interrupt, nudge and retry forever while resetting its own budget on every cancellation.

## ZeroMQ

- Two planes, both ZeroMQ, never mixed. Control (fire, wake, answer) stays on addressed reliable
  sockets + inbox + request/reply — PUB drops slow subscribers by design, so a wake or answer must
  never ride it. Observation rides topics any plugin subscribes to.
- Done: `turn/` (started once the worker holds the prompt, ended on every close), `tool/`
  (ToolCall lifecycle by tool_call_id), `permission/` (requests and decisions, observability only).
- With this feature: `supervision/` (fired with reason and score, nudge sent, goals changed).
- The daemon is the single publisher per topic (Majordomo-broker discipline); workers report via
  existing replies. Drops repair through history paging, status republishes, and the goals tool;
  firing payloads carry what the supervisor must act on inline, so a dropped event changes nothing.

## Out of scope (v1)

- `SessionInfo`/schema/web changes: supervision stays invisible in Chat until someone wants it rendered.
- Supervised-side goal reads (nudges carry the slice); supervisor-side config beyond model+toolset.

## Build splits, in order

1. Spec field + validation + `_end_turn` firing with payload (e2e: firing arrives, silence case).
2. `supervisor_tools` + `goals.json` + `SearchSessionHistory`.
3. Permission backstop: timeout, presence detection, allow-list, `by=supervisor` (e2e: timeout answer,
   person-beats-supervisor, unlisted tool stays human).
4. Guards (budget, no-repeat, escalation) + `supervision/` topic + full e2e.
