# aid

AI daemon. Hosts persistent agent sessions — external ACP agents (`claude-agent-acp`, `opencode acp`, …),
`pydantic_ai` agents imported by `module:attr`, and interactive Claude Code in pymux — behind an async Python
API and a CLI.

## Layout

- `src/aid/spec.py` — `AcpSpec` / `PydanticAISpec` / `ClaudeTtySpec`: what a session runs.
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
  `permissions`, `manage`; `NEEDS` maps op → grant, unlisted ops no plugin may send). Sockets `plugins.sock`
  (ROUTER) and `plugin-events.sock` (PUB), both CURVE; `_plugin_loop` checks each request against the grants as they
  are now → `forbidden`. The daemon stamps what a plugin cannot claim: a message's sender is `plugin:<name>`,
  `AnswerPermission.plugin` is the connection's (None on control) → `PermissionDecision.by=plugin` + name.
  Registry ops (`add_plugin`, `plugins`, `remove_plugin`) only on control.sock, never in `PageRequest`. State:
  `state_dir/plugins/<name>/plugin.json` + `secret` (0600, `aid plugin add` makes it); the daemon writes its public
  key to `runtime_dir/server.key` at start (new each start). Client: `aid.connect(plugin=NAME)`,
  `register_plugin`. Grants bind only what arrives on the plugin sockets: a same-user local plugin can open
  control.sock, which checks nothing, until confined (Lillecarl/aid#4). The events PUB cannot drop a subscriber: a
  plugin losing `read` hears events until it reconnects.
- `src/aid/guard.py` — `aid guard`: plugin (read, permissions) that allows a request whose command is one
  program from `READERS` with no shell syntax, and leaves the rest for a person. Seed for aid#3.
- `src/aid/backends/` — `acp.py`, `pydantic_ai.py`, `claude_tty.py`; each implements `base.Backend`.
- `src/aid/mcp.py` — a spec's `mcp_servers` as ACP `session/new` params and as Claude Code `--mcp-config` JSON,
  plus the built-in `aid` server (`session_servers`). claude-agent-acp restarts its query when `session/load` gets
  servers other than `session/new` did.
- `src/aid/tools.py` — `@aid.mcptool`, a marker attribute; `agents.discover` collects the marked functions.
  Keep it free of `mcp` imports: every worker and the daemon import it.
- `src/aid/mcp_server.py` — `python -m aid.mcp_server`, the tools as a stdio MCP server. The agent starts it; it
  imports user code, so never the daemon. pydantic-ai sessions get the same functions in-process as a toolset.
  For interactive Claude it is also the session's channel (research preview): it long-polls the daemon for the
  session's messages and pushes `notifications/claude/channel`.
- `src/aid/builtin_tools.py` — aid's own tools, in every session: `send_message`, `list_sessions`.
- Request ids route replies across all clients: the daemon refuses an id in flight (`duplicate_id`); relays and
  plugins never pass a caller's id through unchecked.
- Messages: `SendMessage` → recipient's history + inbox. ACP and pydantic-ai get a daemon-started wake turn after
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
- `src/aid/agents.py` — `aid.PydanticAgent`, the interface agent modules implement, and discovery on
  AID_AGENTS_PATH. `catalog.py` runs discovery in a subprocess for the daemon; never import agent modules
  in the daemon itself.
- `src/aid/env.py` — agent environments, minus the markers of the Claude session that started us.
- `src/aid/client.py` — public API: `aid.connect()`, `Client`, `Session.run/stream`.
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
- `src/aid/coding.py` — `aid.coding_tools`, a toolset pydantic-ai agents opt into: list and read, pyedit edits through its
  library (`EditSession`, staged until `apply_edits`; no agent code runs in the worker), and `python` (pyrun
  `run_script` in a child, each command asked of `PydanticAISpec.permission`). Results over `SPILL_LIMIT`
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
nix develop --file . shell --command sh -c 'ruff format src tests && ruff check src tests && pyright'
nix develop --file . shell --command python -m pytest -p no:cacheprovider
nix build --file . tests      # the same suite in the sandbox, against a real pymux
```

The dev shell puts `src/` ahead of the aid its venv carries. Dependencies go in `pyproject.toml`; the set
reads them. Test-only ones go in the `test` extra.

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
  (`launcher.PRELOAD`, default `zmq.asyncio`). Everything else — aid itself, `pydantic_ai`, `acp` — imports fresh
  in each worker, so restarting a worker picks up new code. `PydanticAISpec.python_path` is prepended per worker,
  but a preloaded module and its dependencies win over it. `aid daemon --preload-module` shares more pages
  copy-on-write, `--no-preload` starts bare.
- Keep workers subinterpreter-ready, for when pydantic-core and pyzmq load in more than one interpreter
  (measured 2026-09 on 3.14.7: both refuse isolated subinterpreters; PyO3 allows one interpreter per process):
  - `worker.main` takes only `str` arguments and mutates no process-wide state (cwd, `os.environ`, signals).
  - Process-wide setup (chdir, environ) lives in `launcher.process_main`.
  - The daemon depends on `Launcher`/`WorkerHandle`, never on multiprocessing.
- ACP `session/update` must be read through the connection observer (`AcpBackend.observe`). The library runs
  `Client.session_update` as separate tasks, which can land after `prompt()` returns.
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
