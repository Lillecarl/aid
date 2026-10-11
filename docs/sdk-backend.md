# Claude SDK backend: measured protocol facts

Driving `claude` directly over NDJSON instead of through claude-agent-acp, so
sessions get Remote Control. Measured 2026-10-10 against CLI 2.1.296 (checkout
`~/Code/claude-code` read alongside; versions drift, re-measure on surprises).

## Invocation

`claude -p --verbose --output-format stream-json --input-format stream-json
--model <id> [--permission-mode <mode>] [--resume <session-id>]`. Without
`--verbose`, stream-json refuses to start. stdin and stdout are NDJSON, one
message per line.

## Inbound (stdin)

- `{"type":"user","message":{"role":"user","content":"text"}}` starts a turn.
- `{"type":"control_request","request":{"subtype":"remote_control","enabled":true},"request_id":"..."}`:
  enables the bridge mid-session. The answer arrives on stdout as
  `control_response` with `session_url`, `connect_url`, `environment_id`,
  framed by `system bridge_state` `ready` then `connected`. Cost of one
  enable: nothing beyond the turn.
- `control_response` also answers `can_use_tool` permission prompts (shape
  below); denial behavior measured at implementation time.

## Outbound (stdout)

- `system init` first: session id, tools, models, skills. `system
  bridge_state` tracks the bridge. `assistant` carries model content blocks
  (tool_use included). `result` ends the turn (`subtype: success`, spend
  attached); aid closes stdin and the process exits — one process per turn.
- `rate_limit_event` may appear mid-stream; worth surfacing, not fatal.
- Measured 2026-10-11: with stdin held open the process stays alive after
  `result` (multi-turn server posture), so aid stops reading at `result`,
  closes stdin, and reaps the exit. Permission answers go over stdin while
  the turn runs, which is why stdin stays open until `result`.

## Permissions

`--permission-mode`: acceptEdits, bypassPermissions, default, dontAsk, plan
(+auto behind a flag). Working hypothesis, verify at implementation:
aid ALLOW runs bypassPermissions, ASK runs default (prompts surface as
`control_request` and aid answers), DENY runs dontAsk. Default mode runs
`echo` with no prompt; the exact `can_use_tool` bytes
(`tool_name`, `action_description`, `tool_use_id`, `request_id`, `input`)
were read in structuredIO.ts, not yet captured live.

## Resume and abort

- `--resume <session-id>` continues the on-disk transcript
  (`~/.claude/projects/<slug>/`); the id is stable across processes and is
  what aid persists per session. Verify with one live run.
- Abort is process kill (measured 2026-10-11: exit -9, the turn ends
  `sdk_failed`); what lands on stdout then is unmeasured. Cancel
  must also answer pending `can_use_tool` requests, as the ACP backend does.

## Spend log

- `say hi` on Opus 5.5: $0.18. `run echo hi` + RC enable on Haiku: $0.006.
  Spike discipline: Haiku, microscopic prompts, batch questions per run.
