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

- The daemon fires the supervisor at every supervised turn end, from `_end_turn` — the only new
  decision point. No new wire hooks; delivery reuses the accept→inbox→wake path (idle: new turn,
  live: next turn).
- Payload is small by design (cheap model): last-N history items clipped (order ~10 items / few k
  chars), open goals only, loop score as one line (difflib ratio over recent normalized Outputs,
  threshold ~0.85, counter ≥3 — advisory, computed daemon-side at turn end). Silence ("nothing to
  do, no nudge") is a valid supervisor verdict.
- Mid-turn loop detection is deferred: it needs worker-side streaming observation, and turn-end
  catches cross-turn loops.

## Supervisor session

- Always pydantic-ai, pinned cheap model, short verdicts, dedicated `supervisor_tools`:
  `supervised_history(session, limit)`, `supervised_search(session, query)` (new worker→daemon
  `SearchSessionHistory` request; daemon enforces the supervision link), `goals(session)`,
  `goal_add` / `goal_done` / `goal_drop`, `answer_permission(request_id, option, text)`, plus the
  builtin `send_message` for nudges. Each firing names its subject, so one supervisor watches N.
- Nudges ride `send_message`: attributed, replyable, zero new transport.

## Goals

- `goals.json` beside the supervised session's `history.jsonl`. Daemon reads open items into every
  firing payload; supervisor tools mutate. Goals are the source of done-ness; a DONE-style token in
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
