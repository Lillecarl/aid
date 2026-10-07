# Channels end as user messages

Claude Code's channel machinery controls when text arrives and how it is framed. It adds no new
message kind. Everything the model sees is a normal user message with an XML tag around it.

Source: the Claude Code CLI source of 2026-03-31 (`~/Code/claude-code`, `tanbiralam/claude-code`).
Re-measure before changing anything built on it; channels are a research preview.

## The two paths

- A `notifications/claude/channel` event is wrapped in `<channel source="…">…</channel>`
  (`services/mcp/channelNotification.ts: `wrapChannelMessage``) and enqueued with `mode: 'prompt'`
  (`cli/print.ts:4751`). The queue drains between turns; each entry becomes its own user message
  with its own UUID, batched with other prompts into one follow-up turn (`utils/queueProcessor.ts:40`).
- A background-agent completion arrives as `mode: 'task-notification'` (`<task-id>`, `<status>` tags)
  and falls through to `ask()`, so the model sees the result (`cli/print.ts:2015`).

## The gate chain

`gateChannelServer` registers the notification handler only after all of these pass: the server
declares `experimental['claude/channel']`; the `tengu_harbor` runtime gate is on (default off);
auth is OAuth (API key users are blocked); a team/enterprise org opts in; the server is on the
session `--channels` list; and a plugin server is on the GrowthBook allowlist. A bare `server:`
entry always fails that last check — only `--dangerously-load-development-channels` bypasses it.

## What this means for aid

- aid's tty channel is the same mechanism: `aid.mcp_server` declares `claude/channel` and pushes
  one event per mailbox message; Claude starts with `server:aid` on the dev-channels flag. Measured
  on 2.1.284: the model sees `<channel source="aid" from="…">` and treats it as untrusted.
- The same server attached to `claude-agent-acp` 0.75.1 delivers nothing. Its inner Claude 2.1.266
  runs without the flag, and the model reports no channel event while wake-disabled (measured
  2026-10-07, scratch daemon, `uses_channel` flipped for isolation, reverted after). Turns XOR
  channel stays: ACP keeps wake turns.
- Framing is available everywhere with no protocol: `wake_prompt` can tag firings the way Claude
  tags channel and system-reminder blocks. Mid-turn timing over ACP needs a blocking wait tool
  (their SleepTool pattern); nothing else in this source offers it.
