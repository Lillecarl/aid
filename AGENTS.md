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
  A `wss://` worker sets WSS_HOSTNAME: without it libzmq accepts a trusted certificate for any name.
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
- `src/aid/web/` — `aid web`: Starlette on hypercorn, OIDC login (`auth.py`), JSON API + SSE (`app.py`).
- `web/` — the Svelte 5 UI (runes, TypeScript, Vite). `web/default.nix` builds it; `aid web` serves the result
  from `AID_WEB_ASSETS`, which the installed `aid` wrapper sets.
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

- anyio only. ruff bans `asyncio` and `subprocess` (TID251). The asyncio seam lives inside the `acp` library.
- Worker isolation is a process per session, forked from a forkserver that preloads the heavy stack.
  `PydanticAISpec.python_path` is prepended per worker, but modules in `launcher.PRELOAD` and their
  dependencies are already imported and win: an agent cannot bring its own version of them. Shrinking
  `PRELOAD` trades shared memory for isolation.
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
- Web live views stream only while their tab is mounted and the browser tab is visible: Status by SSE
  (`api.watch`), Terminal by `<pymux-pane>` (pyterm's `pymux-element`, linked in as `node_modules/pymux-pane`)
  over `/api/sessions/{name}/pane`, which relays `libpymux.PaneStream` without reading it. The element styles
  through the CSSOM, so the CSP stays `default-src 'self'`: do not add inline allowances.
- Web: a session runs commands on the host, so login needs a verified email on the allowlist, and every
  mutating request needs the CSRF header. The CSP forbids inline script; agent output is text, never HTML.
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
