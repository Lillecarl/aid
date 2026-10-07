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
- **Notify** — the worker→daemon message (`Notify`, wire `notify`): a fired monitor's task id plus the
  report text for the agent. This is what "a background task wakes" means.
- **Notification** — the event plus its payload, carried by `Notify`. Creates no turn by itself.
- **Resume message** — what a notification becomes inside the session: a `MessageEntry` whose sender names
  the background task (`background task bg1`). `wake_prompt` renders it as written — a report, not a
  quote — and skips the send_message trailer for it.
  - Loop idle → the message starts a new turn: a **resume**.
  - Loop live → the message waits in the inbox for end of turn: an **injection**. No mid-run injection
    exists (a run's input is fixed at start), so the turn boundary is the earliest possible point, and
    the existing end-of-turn delivery is what delivers it.

## Background work

- **Background task** — a worker-side process started by `background`. Outlives its turn, dies with the
  worker. Reported through `task_output` / `tasks`, stopped with `task_stop`.
- **Monitor** — a watch on one background task, armed by `monitor` with a condition. Terminal
  conditions (`ended`, `failed`) fire once at the task's end; live ones fire while it runs (`pattern` on
  every matching line, `running_after` once past its age). Watches live with their task and die with it;
  the agent stops one early with `monitor_stop`, lists them with `monitors`. Watches die with the worker
  (aid#8).

## Answered questions

1. Injection into a live loop: impossible mid-run, and unneeded — the inbox holds the notification and
   end-of-turn delivery hands it to the next turn. Answered by the notify design.
2. The worker→daemon notification channel: the `Notify` message, handled like a message from no person
   or session. A worker whose session is gone notifies nothing.
3. Where resume messages live in history: `MessageEntry` items with a `background task <id>` sender,
   recorded like any message.
