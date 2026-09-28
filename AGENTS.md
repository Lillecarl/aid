# aid

AI daemon. Hosts persistent agent sessions — external ACP agents (`claude-agent-acp`, `opencode acp`, …) and
`pydantic_ai` agents imported by `module:attr` — behind an async Python API and a CLI.

## Layout

- `src/aid/spec.py` — `AcpSpec` / `PydanticAISpec`: what a session runs.
- `src/aid/protocol.py` — pydantic messages on the zmq sockets; `AidError`.
- `src/aid/daemon.py` — ROUTER for clients (`control.sock`), ROUTER for workers (`workers.sock`), session state.
- `src/aid/launcher.py` — `Launcher` protocol; `ForkserverLauncher`.
- `src/aid/worker.py` — worker entry `main(endpoint, name, spec_json, state_dir)`; DEALER to the daemon.
- `src/aid/backends/` — `acp.py`, `pydantic_ai.py`; both implement `base.Backend`.
- `src/aid/client.py` — public API: `aid.connect()`, `Client`, `Session.run/stream`.
- `aid/default.nix` — package; `nix/` — NixOS and home-manager modules.

## Commands

```sh
nix develop --file . shell --command sh -c 'ruff format src tests && ruff check src tests && pyright'
nix develop --file . shell --command python -m pytest -p no:cacheprovider
nix build --file . aid        # runs the same tests in the sandbox
```

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
