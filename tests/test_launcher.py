from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from aid.control import ENV_AGENT_CONNECTION, AgentConnection
from aid.launcher import PRELOAD, ForkserverLauncher, WorkerArgs, process_main
from aid.paths import hook_socket
from aid.spec import NativeSpec

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_preload_keeps_worker_code_fresh() -> None:
    """Only the transport may be preloaded: aid and the agent frameworks import fresh in each worker, so a
    worker restart picks up new code and `python_path` can override them."""
    assert PRELOAD == ("zmq.asyncio",)
    assert not any(name == "aid" or name.startswith("aid.") for name in PRELOAD)
    assert not {"openai", "anthropic", "acp"} & set(PRELOAD)


def test_worker_endpoint_needs_command(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="need worker_command"):
        NativeSpec(cwd=str(tmp_path), target="agents:echo", worker_endpoint="ipc://x")
    with pytest.raises(ValidationError, match="need worker_command"):
        NativeSpec(cwd=str(tmp_path), target="agents:echo", worker_ca="/ca.pem")
    with pytest.raises(ValidationError, match="names no program"):
        NativeSpec(cwd=str(tmp_path), target="agents:echo", worker_command=[""])
    spec = NativeSpec(
        cwd=str(tmp_path),
        target="agents:echo",
        worker_command=["aid", "worker"],
        worker_endpoint="ipc://x",
    )
    assert spec.worker_command == ["aid", "worker"]


def test_hook_socket_lives_beside_the_daemons_own() -> None:
    """ipc paths cap at ~107 chars, and a session dir can run deeper: hooks bind by the daemon's sockets."""
    assert hook_socket("/run/aid", "s1") == "ipc:///run/aid/hook-s1.sock"


def test_process_main_hands_the_worker_its_hook_socket(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_main(*args: str) -> None:
        calls.append(args)

    monkeypatch.setattr("aid.worker.main", fake_main)
    monkeypatch.setattr("aid.launcher.configure_logging", lambda: None)
    spec_json = NativeSpec(cwd=str(tmp_path), target="agents:echo").model_dump_json()
    process_main(
        WorkerArgs(
            endpoint="ipc://workers.sock",
            trust_pem="",
            server_key="s" * 40,
            public_key="p",
            secret_key="q",
            agent_public_key="a" * 40,
            agent_secret_key="b" * 40,
            name="s1",
            spec_json=spec_json,
            state_dir=str(tmp_path),
            daemon_runtime_dir="/run/aid",
            daemon_state_dir="/state/aid",
        )
    )
    assert calls == [
        (
            "ipc://workers.sock",
            "",
            "s" * 40,
            "p",
            "q",
            spec_json,
            str(tmp_path),
            "ipc:///run/aid/hook-s1.sock",
        )
    ]
    connection = AgentConnection.model_validate_json(os.environ[ENV_AGENT_CONNECTION])
    assert connection.endpoint == "ipc://workers.sock"
    assert connection.public_key == "a" * 40
    assert connection.secret_key == "b" * 40


class _FakeContext:
    def __init__(self) -> None:
        self.preloads: list[list[str]] = []

    def set_forkserver_preload(self, modules: list[str]) -> None:
        self.preloads.append(modules)


def test_forkserver_launcher_passes_preload(monkeypatch: MonkeyPatch) -> None:
    # The forkserver imports its preload once, when it starts: in-process only the first launcher counts,
    # so this checks the handoff, not a running forkserver.
    context = _FakeContext()

    def fake_context(name: str) -> _FakeContext:
        assert name == "forkserver"
        return context

    monkeypatch.setattr(multiprocessing, "get_context", fake_context)
    ForkserverLauncher(["a", "b"])
    ForkserverLauncher()
    assert context.preloads == [["a", "b"], list(PRELOAD)]
