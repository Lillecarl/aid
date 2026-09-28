# aid

AI daemon. Hosts persistent agent sessions — external ACP agents (`claude-agent-acp`, `opencode acp`, …),
`pydantic_ai` agents imported by `module:attr`, and interactive Claude Code in pymux — behind an async Python
API and a CLI.

## Layout

- `src/aid/spec.py` — `AcpSpec` / `PydanticAISpec` / `ClaudeTtySpec`: what a session runs.
- `src/aid/protocol.py` — pydantic messages on the zmq sockets; `AidError`.
- `src/aid/daemon.py` — ROUTER for clients (`control.sock`), ROUTER for workers (`workers.sock`), session state.
- `src/aid/launcher.py` — `Launcher` protocol; `ForkserverLauncher`.
- `src/aid/worker.py` — worker entry `main(endpoint, name, spec_json, state_dir)`; DEALER to the daemon.
- `src/aid/backends/` — `acp.py`, `pydantic_ai.py`, `claude_tty.py`; each implements `base.Backend`.
- `src/aid/transcript.py` — Claude Code transcript entries → aid events.
- `src/aid/env.py` — agent environments, minus the markers of the Claude session that started us.
- `src/aid/client.py` — public API: `aid.connect()`, `Client`, `Session.run/stream`.
- `src/aid/web/` — `aid web`: Starlette on hypercorn, OIDC login (`auth.py`), JSON API + SSE (`app.py`).
- `web/` — the Svelte 5 UI (runes, TypeScript, Vite). `web/default.nix` builds it; `aid web` serves the result
  from `AID_WEB_ASSETS`, which the installed `aid` wrapper sets.
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
  - Process-wide setup (chdir, environ) lives in `launcher._process_main`.
  - The daemon depends on `Launcher`/`WorkerHandle`, never on multiprocessing.
- ACP `session/update` must be read through the connection observer (`AcpBackend.observe`). The library runs
  `Client.session_update` as separate tasks, which can land after `prompt()` returns.
- multiprocessing re-imports the parent's `__main__` in each worker: every entry script needs a
  `if __name__ == "__main__":` guard.
- Daemon shutdown: `_stop_all` runs inside the task group; worker watchers are shielded and must see every exit.
- ACP permission requests are answered by `AcpSpec.permission`; nobody is attached to ask.
- claude-tty depends on Claude Code's screen and transcript, neither an API. `claude_tty.py` and
  `transcript.py` say what was measured, on which version; re-measure before changing them.
- libpymux and the `pymux` binary come from one pyterm pin (`default.nix`): the wire protocol still moves.
- Web: a session runs commands on the host, so login needs a verified email on the allowlist, and every
  mutating request needs the CSRF header. The CSP forbids inline script; agent output is text, never HTML.
