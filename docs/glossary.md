# Glossary

Words this project uses with one meaning. When conversation drifts, this file wins. "Wake" once meant
two things here; it no longer does.

## Process and session

- **Daemon** — the long-lived aid process. Owns session state; routes clients, workers and plugins.
- **Worker** — one process per session, forked per launch. Runs the agent backend (ACP, pydantic-ai,
  claude-tty) and dies with the session.
- **Session** — a named, persistent agent conversation: history, status, working directory, store.
- **Spawn** — creating (or restarting) the worker subprocess. Process mechanics only; says nothing about
  what the agent hears.

## Turns and the loop

- **Turn** — one agent run over the session: a prompt in, model steps and tool calls, then usage and output.
- **Agentic loop** — the model↔tools iteration inside a turn. Either running or not; never "woken".
- **Idle loop** — no turn running. What conversation called "dead" or "paused".
- **Live loop** — a turn is running right now.
- **End of turn** — the loop finishes (the model returns its final output) and no new turn starts. Loose
  shorthand, not a protocol op.

## Waking, in two senses that must not mix

- **Wake turn** — a daemon-started turn with no client (ACP and pydantic-ai get one after the running
  turn; interactive Claude gets a channel event instead). Turn mechanics.
- **Notify** — a background task or monitor firing: a worker-side event carrying facts ("bg3 finished,
  exit 0, tail …"). Creates no turn by itself. This is what "a background task wakes" means.
- **Notification** — the event plus its payload, worker→daemon. No worker-initiated path exists yet;
  see open questions.
- **Resume message** — what a notification becomes inside the session: a developer/system message
  (pydantic-ai: a system prompt part) appended so the loop sees the result.
  - Loop idle → the message starts a new turn: a **resume**.
  - Loop live → the message waits for the next iteration: an **injection**. Mechanism open.

## Background work

- **Background task** — a worker-side process started by `background`. Outlives its turn, dies with the
  worker. Reported through `task_output` / `tasks`, stopped with `task_stop`.
- **Monitor** — a condition watched on the task registry (output pattern, exit, timeout). Firing produces
  a notification. Planned, not built (aid#8).

## Open questions

1. Injection into a live loop: can the pydantic-ai backend queue a message for the next iteration
   mid-run, or must every notification wait for end of turn?
2. The worker→daemon notification channel: new protocol message plus daemon handling, including what
   happens when the session ended or the worker died first.
3. Where resume messages live in history: appended as system parts, or a history item of their own?
