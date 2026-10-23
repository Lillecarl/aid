# aid

AI server for persistent, cooperating agent sessions, with separate execution engines and extension hosts.
Runs external ACP agents (`claude-agent-acp`, `opencode acp`, …), `litelm` agents, and Claude Code through
terminal and SDK backends, behind an async Python API, CLI and web UI.

## Architecture stance

Fixed: a plugin architecture, async (anyio). Everything else is a current shape, not a decision to defend.

- aid is agent glue. The daemon does as little as possible: sessions, routing, delivery, authorization. Behaviour
  that can live in an agent, a backend or a plugin goes there. Before adding to the daemon, ask what could own it.
- What can reasonably be a plugin is one, bundled and on by default where it matters (skills, chat bridges).
- Essential conventions (Agent Skills, AGENTS.md, MCP) follow the common baseline other harnesses share; go beyond
  it, never around it.
- Re-evaluate the architecture as work lands; each layer built on a weak shape makes it costlier to change. A better
  way found is a way taken: change the code, then this file.
- Architectural suggestions go to User before building them: what, why, cost, recommendation.

## Working method — read first

The goal is reliable agent work, not merely code that passes a happy-path test. Use this sequence:

1. **Define the observable outcome.** State what should happen and what must remain unchanged. A request to
   stop loops does not authorize an arbitrary turn cap; a request for plugins does not imply one universal ABC.
2. **Trace the actual path before editing.** Follow caller → protocol → daemon → worker → backend → stored
   state → response. Read the relevant tests too. A helper's docstring is not proof that callers use it correctly.
3. **Identify the owner and invariant.** Who owns the state, who may mutate it, how long does it live, and what
   happens on cancellation/restart? Prefer one owner and explicit transitions over two copies kept in sync.
4. **Reproduce with the smallest real path.** Use deterministic local models/fake agents, not paid API calls.
   Check the installed dependency version and runtime behaviour when a default or API is involved. Separate
   confirmed failures, hypotheses and recommendations in updates; do not present guesses as findings.
5. **Fix the cause at its owning layer.** Keep decisions pure where practical; keep IO and lifecycle effects
   explicit. Make small coherent changes. Do not rearrange directories, add abstractions or tune thresholds
   merely to make the code look organized.
6. **Prove the lifecycle, not just the return value.** Add a regression that fails before the fix. Check success,
   failure, cancellation, resume/restart and stale/duplicate input where relevant. Verify persisted state, usage,
   attribution and actual effects. A compaction test outside a turn does not prove the compact tool works inside one.
7. **Validate and close the change.** Run targeted tests, formatting/lint/types, then the wider affected suite.
   Regenerate schema/types when models change. Report what changed, the checks run and remaining limitations.
   Never claim a suite passed if only a subset ran, and never weaken a test just to obtain green output.

If an experiment fails, preserve the evidence and investigate the boundary it exposed. Do not patch blindly
around a failed check. Change a test's expectation only when the intended contract changed, and explain why.

## Decision guide

| Tempting shortcut | Better decision |
| --- | --- |
| Increase a request limit to stop errors | First check whether the user requested a limit, and whether `None` inherits an engine default. |
| Cancel whenever messages look similar | Raise a suspicion with evidence; let the supervisor judge progress. Polling and repetitive work can be legitimate. |
| Send a nudge to an indefinitely running turn | Check delivery timing. A queued message cannot redirect that turn; interruption needs an explicit, turn-scoped operation. |
| Retry a failed/interrupted tool automatically | Its effects may already have happened. Preserve the result/state and inspect before replaying a side effect. |
| Compare only the newest history entries | Define the signal's unit. Ten tool events are not ten answers; load/cache the relevant completed units. |
| Catch every error and continue | Distinguish normal outcomes from faults. Preserve work and usage, propagate cancellation, and log faults with their original cause. |
| Reconstruct every plugin during reconciliation | Select eligible classes first; the lifecycle host constructs only instances it needs to start. |
| Put critical control on PUB | Use addressed request/reply or the reliable inbox. PUB is observation and may drop events. |

## Ownership and extension boundaries

- **Daemon:** durable session state, routing, supervision links and authorization of daemon operations.
- **Worker:** one session's execution lifecycle and backend capabilities. Recheck turn identity here before an
  interruption: a daemon check can become stale while a request travels.
- **Backend:** engine adaptation, engine conversation history, cancellation and usage. Do not spread engine
  assumptions across the daemon, CLI and UI. Different engines expose different capabilities; say so explicitly.
- **Launcher:** process/environment setup and placement, not the agent's conversation or supervision policy.
- **Extensions:** use the appropriate ABC/contract for their host (`Plugin`, `AgentDefinition`,
  `InstructionsProvider`, backend capabilities), not a catch-all object. Specify construction, context,
  shutdown, restart and failure ownership before adding a new extension type. Discovery is not instantiation.

The daemon transcript and the engine's resumable conversation are different stores; preserving one does not
prove the other survived. A status flag is not enforcement, and a grant checked on one socket does not secure
another socket. Missing telemetry is unknown, not proof that nothing changed.

The layout below describes the current implementation, not a mandate to preserve every historical design.
When prose and code disagree, reproduce the behaviour, fix the contract and update both. Keep this file focused
on reusable decisions and current boundaries; put long investigations and dated measurements in `docs/`.

## Layout

- `src/aid/spec.py` — `AcpSpec` / `LiteLMSpec` / `ClaudeTtySpec` / `ClaudeSdkSpec`: what a session runs.
- `src/aid/protocol.py` — pydantic messages on the zmq sockets; `AidError`.
- `src/aid/daemon.py` — ROUTER for clients (`control.sock`), ROUTER for workers (`workers.sock`), session state.
- `src/aid/launcher.py` — `Launcher` protocol; `ForkserverLauncher`.
- `src/aid/worker.py` — worker entry `main(endpoint, server_key, public_key, secret_key, spec_json, state_dir)`;
  DEALER to the daemon.
- `src/aid/zap.py` — every worker connection is CURVE. The daemon issues a keypair per launch; its ZAP handler
  names the session as the connection's User-Id, and the daemon routes by that, never by a worker's routing id.
  A `wss://` worker sets WSS_HOSTNAME: without it libzmq accepts a trusted certificate for any name. Domains:
  workers, plugins, plugin events (wants `read`).
- `src/aid/plugins.py` — plugins: a registered CURVE public key + grants (`read`, `prompt`, `message`,
  `permissions`, `manage`, `supervise`; `NEEDS` maps op → grant, unlisted ops no plugin may send). Sockets `plugins.sock`
  (ROUTER) and `plugin-events.sock` (PUB), both CURVE; `_plugin_loop` checks each request against the grants as they
  are now → `forbidden`. The daemon stamps what a plugin cannot claim: a message's sender is `plugin:<name>`,
  `AnswerPermission.plugin` is the connection's (None on control) → `PermissionDecision.by=plugin` + name.
  Calls: a plugin overriding `Plugin.call(caller, method, args)` serves them (`plugin_host.run` → `serve_calls`,
  `ServeCalls` renewed every `SERVE_RENEW`, no grant needed); `CallPlugin` from a session tool or control gets
  `caller` stamped (agent connection → its session; control → as sent, None = operator), goes out as an
  `Invocation` under a daemon id, answered by `CallResult` from that plugin only, bounded by `PLUGIN_CALL_TIMEOUT`.
  Plugins themselves may not send `CallPlugin`. The plugin authorizes by `caller`. `supervise` (trusted: watching
  over sessions) adds `spec`, `attention`, `publish_supervision`, the `message` hook (`Plugin.on_message`, asked
  before each `SendMessage` is accepted; refusal = its code, silence/absence = pass), a kept `SendMessage.sender`
  and `AnswerPermission.supervisor`. plugin.toml `autostart = true`: hosted with declared grants, registered in
  memory each start; removal keeps it off for the daemon's lifetime.
  Registry ops (`add_plugin`, `plugins`, `remove_plugin`) only on control.sock, never in `PageRequest`. State:
  `state_dir/plugins/<name>/plugin.json` + `secret` (0600, `aid plugin add` makes it); the daemon writes its public
  key to `runtime_dir/server.key` at start (new each start). Client: `aid.connect(plugin=NAME)`,
  `register_plugin`. Grants bind only what arrives on the plugin sockets: a same-user local plugin can open
  control.sock, which checks nothing, until confined (Lillecarl/aid#4). The events PUB cannot drop a subscriber: a
  plugin losing `read` hears events until it reconnects.
- `src/aid/discovery.py` — the plugins path (`AID_PLUGINS_PATH`, `~/.config/aid/plugins`, `AID_PLUGINS_DIST`):
  a plugin is a directory named as an import name, with `__init__.py` and/or optional `plugin.toml` (`name` = dir,
  `version`, `description`, `requires`). `resolve` reads manifests only (first tier wins a name; missing requirement
  or cycle drops dependents); `discover` imports packages in dependency order, under their own names, never via
  `sys.path`, and sorts by interface: `AgentDefinition`, `InstructionsProvider`, `@aid.mcptool`, `Plugin` (≤1 per
  plugin; `plugin_name()` = top-level package). Data-only plugins (`skills/`) need no `__init__.py`. Workers and
  `aid.mcp_server` import everything; the daemon never imports a plugin without a task (below).
- `src/aid/catalog.py` — `python -m aid.catalog [--path DIR]`, the catalog as JSON (`AgentCatalog`, plugins
  included); `listing()` runs it as a subprocess for the daemon. `session_catalog()` is what a session gets.
- `src/aid/plugin_host.py` — in-process plugins: `hosted()` lists the daemon's `plugin_dirs` (None means the standard
  tiers) in a subprocess and imports only plugins with a task, plus what they require. Registered ones run as
  supervised tasks speaking the same protocol over
  `inproc://aid-plugins` / `inproc://aid-plugin-events`, through the same loop (identity is the DEALER name the
  daemon set). `_reconcile_plugins` runs at startup and after every registry change: unregistered, under-granted
  and declined code stays outside with a log; removal stops the task at once. Failures restart after
  `RESTART_DELAY`, give up after `MAX_FAILURES` in `FAIL_WINDOW`. Tests pass `plugin_dirs=[]`; the daemon's own
  environment is what hosted plugins read.
- `src/aid/guard.py` — `aid guard`: plugin (read, permissions) that allows a request whose command is one
  program from `READERS` with no shell syntax, and leaves the rest for a person. Seed for aid#3.
- `plugins/discord/` — the Discord plugin (read, message): `gateway.py` (the gateway over anyio streams, no
  Discord library — wsproto frames, anyio owns the sockets; resume policy follows the close-code table),
  `rest.py` (channels, messages, interaction answers; 429s honored and retried, the rest raises at once),
  `channels.py` (one channel per session under one category, id-mapped on disk, reconciled at startup and
  on list changes), `serve.py` (bare text in session channels as messages, `!`-commands everywhere,
  taps as new messages or in-place rewrites, settled notices to session channels). Custom ids hold
  `action:session` directly (100 bytes); Discord messages clip at 1900/2000.
- `plugins/skills/` — Agent Skills, bundled: an `InstructionsProvider` lists each skill's name and description, the
  `skill` tool loads a body or file. Spec-checked frontmatter (YAML); project `.aid|.agents|.claude/skills` up to the
  git root, then `AID_SKILLS_PATH`, then the global dirs, then each resolved plugin's `skills/`.
- `plugins/forks/` — short-lived forks, bundled, `autostart` (read, prompt, message, manage, supervise). Tool
  `fork(instructions)`: `Coding.snapshot` writes the conversation inside the call (history.json lacks the running
  turn) to `state_dir/forks/`; the plugin creates `<parent[:52]>.fork-<6hex>` from the parent's `LiteLMSpec` with
  `seed_history` (a seeded session skips instruction composing: same cache prefix) and `parent`, prompts it, sends
  the answer to the parent as the fork with its turn `Usage` (`agent="fork <name>"`; `SendMessage.usage`, kept
  only from control and `supervise` plugins, recorded after the message), deletes it. A session with a `parent` can't fork; leftovers
  deleted at plugin start. Spec `parent` (also on `SessionInfo`/`SessionStatus`): chat plugins skip such sessions
  (no Discord channel, no settled-turn notice, not in session lists).
- `src/aid/backends/` — `acp.py`, `litelm.py`, `claude_tty.py`, `claude_sdk.py`; each implements `base.Backend`.
- `src/aid/model.py` — the `litelm` backend's transports (no LiteLM library; the name stays): `model_for(name)`,
  `anthropic/` → `AnthropicModel` (Messages API, native blocks, `extra_headers` carry betas, automatic caching
  on), else `OpenAIChatModel` (Chat Completions; `OPENAI_COMPATIBLE` maps prefix → base URL + key variable).
  `model_settings`: `api_key`/`api_base`/`max_retries` build the client, the rest goes into each request.
- `src/aid/mcp.py` — a spec's `mcp_servers` as ACP `session/new` params and as Claude Code `--mcp-config` JSON,
  plus the built-in `aid` server (`session_servers`). claude-agent-acp restarts its query when `session/load` gets
  servers other than `session/new` did.
- `src/aid/tools.py` — `@aid.mcptool`, a marker attribute; `discovery.discover` collects the marked functions.
  Keep it free of `mcp` imports: every worker and the daemon import it.
- `src/aid/mcp_server.py` — `python -m aid.mcp_server`, the tools as a stdio MCP server. The agent starts it; it
  imports user code, so never the daemon. LiteLM sessions get the same functions in-process as a toolset.
  For interactive Claude it is also the session's channel (research preview): it long-polls the daemon for the
  session's messages and pushes `notifications/claude/channel`.
- `src/aid/builtin_tools.py` — aid's own tools, in every session: `send_message`, `list_sessions`.
- Request ids route replies across all clients: the daemon refuses an id in flight (`duplicate_id`); relays and
  plugins never pass a caller's id through unchecked.
- Messages: `SendMessage` → recipient's history + inbox. ACP and LiteLM get a daemon-started wake turn after
  the running turn; interactive Claude gets a channel event. A wake turn has no client (`_Route.client` None).
- `src/aid/transcript.py` — Claude Code transcript entries → aid events.
- Recording: each turn ends with `Usage` (tokens, models, context, the agent's session cost) before `Output`; each
  worker start records `Started` (agent session id, agent, model) under a turn id of its own. What a turn's tokens
  cover differs per agent: `Usage`'s docstring says how, measured. The web UI must render every `HistoryItem` type:
  an unknown one breaks Chat.
- claude-tty hooks: `aid/hook.py` runs as a file (no aid import: 0.04 s against 0.2 s per event), sends a `hook`
  request, prints the answer, and fails open (exit 0, no output). The daemon routes it to the worker
  (`HookBackend.hook`); worker state goes back as `Activity` (working, attention). Events: `HOOK_EVENTS`.
- claude-tty follows its transcript for the worker's life (`FollowingBackend.follow`): a turn aid's `prompt()` waits
  for is aid's; any other (typed into the pane, woken by a channel event) goes to the daemon as `Observed` entries
  under a turn id of its own.
- `src/aid/agents.py` — `aid.AgentDefinition` and `aid.InstructionsProvider`, interfaces plugins implement. Never
  import agent plugins in the daemon itself: it asks `aid.catalog` in a subprocess.
- `src/aid/env.py` — agent environments, minus the markers of the Claude session that started us.
- `src/aid/client.py` — public API: `aid.connect()`, `Client`, `Session.run/stream`.
- `src/aid/client.py:daemon_call` — what hosted loops await daemon calls through: silence past
  `DAEMON_CALL_TIMEOUT` exits loudly, so the supervisor restarts into a fresh baseline.
- `src/aid/settle.py` — settled turn ends, platform-agnostic: `Tracker` fed by turn and supervision events,
  `settled_turns` calling back per settled episode. Both chat plugins share it.
- `plugins/supervision/` — supervision, bundled, `autostart` (read, message, prompt, permissions, supervise). Follows each
  supervised session's history (gap-filled), from the last entry before it started: turn ends (`Output`,
  `TurnError`, once per turn) message the supervisor as the supervised session with tail/goals/loop score;
  `loops.py` exact tool-call/cycle suspicions (never cancels); permission backstops; stuck episode → attention;
  nudge budget via the `message` hook. Goals in its plugin dir (`goals/<session>.json`, legacy session-dir
  `goals.json` read). `aid.supervisor_tools` (core) call it via `CallPlugin`; it authorizes by `caller`.
  `interrupt` = `Cancel(turn=…)`, the worker rechecks the turn. Daemon keeps only `spec.supervisor` links and
  their validation (`supervision_cycle`). State is in memory: a restart starts a fresh episode.
- `src/aid/mutes.py` — runtime mutes per chat plugin, on disk beside the registration.
- `src/aid/events.py` — the daemon's PUB socket (`Paths.events`): `[topic, JSON]` for the session list, each
  status, each history entry. The daemon compares state with what it last published after every request,
  worker reply and worker exit (`_publish_changes`); history publishes from `HistoryLog.on_append`. Readers use
  `Client.follow_sessions` / `Session.follow_status` / `Session.follow_history`: subscribe, then fetch the
  baseline, then follow; a jump in seq is fetched. `idle=` yields None on quiet spells (never cancel the
  generator's receive). One SUB socket per follower, closed with it.
- `src/aid/web/` — `aid web`: Starlette on hypercorn, OIDC login (`auth.py`), HTTP for what aid web answers itself
  (`app.py`: files, speech, pane).
- `src/aid/web/zws.py` — the daemon for the page, as ZWS 2.0 (flags byte + body per WebSocket message; bare
  `ZWS2.0`, no mechanism): `/api/zws/control` a DEALER per WebSocket, `/api/zws/events` a SUB (`\x01`/`\x00` + topic).
  Login + Origin gate both. Control admits only `PageRequest` (the allowlist; never `Hook`, `SendMessage`,
  `ReceiveMessages`, pane requests) and swaps in its own request ids: the daemon routes and names turns by id across
  clients. Page side: `web/src/lib/zws.ts` (socket, PING heartbeat, backoff) and `api.ts` (`call`, `follow*`:
  subscribe, then fetch the baseline, changes that beat it applied after; a dropped request fails, never resent).
- `src/aid/schema.py` — `python -m aid.schema`: JSON Schema of what the page reads (serialization mode: defaults
  required) and sends (validation mode). Committed as `web/src/lib/protocol.schema.json`; `npm run types` in web/
  generates `protocol.ts` from it, which `api.ts` re-exports. Never hand-edit either: `tests/test_schema.py` and the
  web build's `types:check` fail on drift. After changing a model: regenerate both.
- `web/` — the Svelte 5 UI (runes, TypeScript, Vite). `web/default.nix` builds it; `aid web` serves the result
  from `AID_WEB_ASSETS`, which the installed `aid` wrapper sets.
- `src/aid/coding.py` — `aid.coding_tools`, a toolset LiteLM agents opt into: list and read, pyedit edits through its
  library (`EditSession`, staged until `apply_edits`; no agent code runs in the worker), and `python` (pyrun
  `run_script` in a child, each command asked of `LiteLMSpec.permission`). Results over `SPILL_LIMIT`
  spill to content-addressed files under the session, sent as a stable head plus the path. `compact`
  summarizes the history into a digest through a direct model call. One `Coding` per session, set per
  turn in the `CODING` contextvar; tool errors go back to the model as `ModelRetry`. pyedit is built into the
  set from Lillecarl/pyedit (`nix/pyedit.nix`); its grammars come from `tree-sitter-grammars` (`grammarsByName`).
- `pyrun/` — async process library, a project of its own (own pyproject, `pyrun/default.nix`, tests) to be
  extractable; contract in `pyrun/README.md`. Kills by the PYRUN_SCOPE/PYRUN_ID env marks found in /proc (no
  subreaper, no process-wide state). Check: `cd pyrun && ruff check src tests && pyright && python -m pytest`;
  sandbox: `nix build --file . pyrun-tests`.
- `default.nix` — a pyproject.nix set from pyterm's builders (`mkPythonSet`, `mkProject`, its `overlay`);
  `aid/default.nix` — the aid project in it; `nix/` — NixOS and home-manager modules.

## Commands

```sh
nix develop --file . shell --command sh -c 'ruff format src tests plugins && ruff check src tests plugins && pyright'
nix develop --file . shell --command python -m pytest -p no:cacheprovider -n auto  # ~2 min; serial ~10
nix build --file . tests      # the same suite in the sandbox, against a real pymux
nix run --file . pyrefly -- check   # project mode: files and import roots come from pyrefly.toml
```

The dev shell puts `src/` ahead of the aid its venv carries. Dependencies go in `pyproject.toml`; the set
reads them. Test-only ones go in the `test` extra.

For a focused change, run its test module first, not the entire suite on every edit. Before finishing, expand
coverage to the affected daemon/client/backends. A generated schema change also needs `tests/test_schema.py`
and `npm run types:check` in `web/`. Run code generators in an initialized environment: outer-shell `>` can
truncate an input file before `nix develop` evaluates the repository. Stage generated output outside the repo
or perform generation inside the already initialized shell, then regenerate its dependent types.

UI: `cd web && npm run check && npm run dev` (proxies the API to `aid web` on `AID_WEB_BACKEND`, default
127.0.0.1:8080). After changing `package-lock.json`, update `npmDepsHash` in `web/default.nix`.

## Rules

- Architecture beats velocity. When the quick fix and the right fix diverge, build the right one: a shortcut
  taken once becomes permanent tax at the scale of every session on the daemon, and scale is the whole point.
  Precedents, all slower than the alternative: per-session worker commands ship behind `--allow-worker-command`
  because the command receives worker credentials; history filtering is store-full plus a deterministic send
  policy (monotonic sends keep the prefix cache) instead of mutating stored history or sliding windows; user
  compaction is one daemon op behind CLI, page and tool instead of a web-only hack; schema and web types
  regenerate through the project's own commands, never by hand.
- anyio only. ruff bans `asyncio` and `subprocess` (TID251). The asyncio seam lives inside the `acp` library.
- Worker isolation is a process per session, forked from a forkserver that preloads only the transport
  (`launcher.PRELOAD`, default `zmq.asyncio`). Everything else — aid itself, the model SDKs, `acp` — imports fresh
  in each worker, so restarting a worker picks up new code. `LiteLMSpec.python_path` is prepended per worker,
  but a preloaded module and its dependencies win over it. `aid daemon --preload-module` shares more pages
  copy-on-write, `--no-preload` starts bare.
- Keep workers subinterpreter-ready, for when pydantic-core and pyzmq load in more than one interpreter
  (measured 2026-09 on 3.14.7: both refuse isolated subinterpreters; PyO3 allows one interpreter per process):
  - `worker.main` takes only `str` arguments and mutates no process-wide state (cwd, `os.environ`, signals).
  - Process-wide setup (chdir, environ) lives in `launcher.process_main`.
  - The daemon depends on `Launcher`/`WorkerHandle`, never on multiprocessing.
- ACP `session/update` must be read through the connection observer (`AcpBackend.observe`). The library runs
  `Client.session_update` as separate tasks, which can land after `prompt()` returns.
- LiteLM: an unset `max_requests` explicitly disables the engine's default request cap. Turn cleanup
  persists available new messages and usage on success, failure and cancellation. Compaction owns the old
  history prefix; finalization appends only the run's new messages, never overwrites that prefix. Unanswered
  tool calls are closed as interrupted, not replayed: their effects may already have happened.
- multiprocessing re-imports the parent's `__main__` in each worker: every entry script needs a
  `if __name__ == "__main__":` guard.
- Daemon shutdown: `_stop_all` runs inside the task group; worker watchers are shielded and must see every exit.
- ACP permission requests: `AcpSpec.permission` allow/deny answers at once; `ask` waits for `AnswerPermission` (web
  card, `aid answer`), refused after `permission_timeout`. The worker's `_Client` emits `PermissionRequest` and
  `PermissionDecision` into the running prompt's events, whatever the mode; the daemon keeps the pending ones in
  `SessionStatus.permissions`. Turn cancel/end answers pending ones `cancelled` (ACP requires it). Requests only
  come mid-prompt; outside one, `ask` falls back to deny. claude-tty asks in its own pane, not here.
- claude-tty depends on Claude Code's screen and transcript, neither an API. `claude_tty.py` and
  `transcript.py` say what was measured, on which version; re-measure before changing them.
- libpymux and the `pymux` binary come from one pyterm pin (`default.nix`): the wire protocol still moves.
- Web live views stream only while their tab is mounted and the browser tab is visible: status, list and history by
  `api.follow*` (the events WebSocket closes on a hidden tab), Terminal by `<pymux-pane>` (pyterm's `pymux-element`, linked in as `node_modules/pymux-pane`)
  over `/api/sessions/{name}/pane`, which relays `libpymux.PaneStream` without reading it. The element styles
  through the CSSOM, so the CSP stays `default-src 'self'`: do not add inline allowances.
- Web: a session runs commands on the host, so login needs a verified email on the allowlist; every WebSocket checks
  the Origin (it carries no CSRF header), and a mutating HTTP request (logout) needs the CSRF header. The CSP forbids inline script; agent output is text, never HTML.
  Markdown goes through marked's lexer into Svelte elements (`Markdown.svelte`), never `{@html}`.
- CodeMirror (`CodeView.svelte`) must live in a shadow root: on a document style-mod adds a style element,
  which the CSP refuses; in a shadow root it adopts a constructed stylesheet.
- Colours: `aid web --theme pygments:<name>|base16:<name>`, pymux's spelling, served as `/theme.css`
  (`web/theme.py`): page variables from pymux's roles, code as Pygments short token classes under `.hl`.
  Components use the variables (`--bad`, `--muted`, …), never literal colours.
- Highlighting: `/file` carries `highlights`, tree-sitter spans (`web/highlight.py`) as UTF-16 offsets and Pygments
  classes. Grammars: `nix/tree-sitter-grammars.nix` → `AID_TREE_SITTER_GRAMMARS` (wrapper, shell, tests). Same node:
  the later pattern wins, as tree-sitter-highlight does. Unknown capture names fall back by prefix (`CLASSES`).
  Measured: 1 MiB of Python, 1.0 s and 105k spans.
- `/api/sessions/{name}/files` and `/file` (`web/files.py`) read the session cwd in aid web itself; every path
  must resolve inside it.
