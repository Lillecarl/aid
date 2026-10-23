"""`usertee`: unchanged passthrough always, a copy to the session's worker when there is one, and the daemon's
live topic and buffer behind it."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
import zmq.asyncio

import aid
from aid import events, usertee
from aid.launcher import WorkerArgs, install_usertee, process_main
from aid.paths import ENV_RUNTIME_DIR, ENV_SESSION, hook_socket
from aid.protocol import GetLive, LiveChunk
from aid.spec import NativeSpec
from aid.worker import usertee_text
from tests.conftest import py_spec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

DATA = b"line one\n\xff\xfe not utf-8\n" + "café — done\n".encode() * 3000


async def tee(data: bytes, env: dict[str, str]) -> tuple[int, bytes, float]:
    started = time.monotonic()
    clean = {k: v for k, v in os.environ.items() if k not in (ENV_SESSION, ENV_RUNTIME_DIR)}
    result = await anyio.run_process([sys.executable, usertee.__file__], input=data, env={**clean, **env}, check=False)
    return result.returncode, result.stdout, time.monotonic() - started


async def test_without_a_session_it_is_cat() -> None:
    assert await tee(DATA, {}) == (0, DATA, pytest.approx(0, abs=5))


async def test_a_dead_worker_delays_it_at_most_the_linger(tmp_path: Path) -> None:
    """A socket file nobody listens on: the copies queue, exit drops them after LINGER_MS."""
    (tmp_path / "hook-s.sock").write_text("")
    code, out, took = await tee(DATA, {ENV_SESSION: "s", ENV_RUNTIME_DIR: str(tmp_path)})
    assert (code, out) == (0, DATA)
    assert took < 5


async def test_it_copies_to_the_worker_socket(tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(hook_socket(tmp_path, "s"))
    try:
        code, out, _ = await tee(DATA, {ENV_SESSION: "s", ENV_RUNTIME_DIR: str(tmp_path)})
        texts: list[str] = []
        with anyio.fail_after(10):
            while "".join(texts) != DATA.decode(errors="replace"):
                *_, frame = await router.recv_multipart()
                assert len(json.loads(frame)["text"]) <= usertee.SEND_MAX
                text = usertee_text(frame)
                assert text is not None
                texts.append(text)
    finally:
        router.close()
        ctx.term()
    assert (code, out) == (0, DATA)


def test_the_worker_tells_its_frames_from_hook_requests() -> None:
    assert usertee_text(json.dumps({"op": "usertee", "text": "hi"}).encode()) == "hi"
    assert usertee_text(json.dumps({"op": "hook", "id": "1"}).encode()) is None
    assert usertee_text(b"[1]") is None


async def test_the_shim_runs_it(tmp_path: Path) -> None:
    directory = install_usertee(str(tmp_path))
    assert install_usertee(str(tmp_path)) == directory
    env = {**os.environ, "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}"}
    env.pop(ENV_SESSION, None)
    result = await anyio.run_process(["sh", "-c", "printf 'a\\nb\\n' | usertee | tail -n 1"], env=env)
    assert result.stdout == b"b\n"


def test_process_main_puts_usertee_first_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", {**os.environ, "PATH": "/usr/bin"})
    monkeypatch.chdir(tmp_path)

    def fake_main(*args: str) -> None:
        pass

    monkeypatch.setattr("aid.worker.main", fake_main)
    monkeypatch.setattr("aid.launcher.configure_logging", lambda: None)
    spec_json = NativeSpec(cwd=str(tmp_path), target="agents:echo").model_dump_json()
    args = WorkerArgs(
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
        daemon_runtime_dir=str(tmp_path / "run"),
        daemon_state_dir=str(tmp_path / "state"),
    )
    process_main(args)
    assert os.environ["PATH"] == f"{tmp_path / 'run' / 'bin'}{os.pathsep}/usr/bin"
    assert (tmp_path / "run" / "bin" / "usertee").stat().st_mode & 0o111


async def test_a_sessions_output_reaches_the_live_topic_and_buffer(daemon: Paths, tmp_path: Path) -> None:
    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(daemon.events)
    sub.subscribe(events.live_topic("teed"))
    try:
        async with aid.connect(daemon) as client:
            await client.create("teed", py_spec(tmp_path, "agents:echo"))
            with anyio.fail_after(20):
                while True:
                    if (daemon.runtime_dir / "hook-teed.sock").exists():
                        break
                    await anyio.sleep(0.05)
                env = {ENV_SESSION: "teed", ENV_RUNTIME_DIR: str(daemon.runtime_dir)}
                code, out, _ = await tee(b"progress 1\nprogress 2\n", env)
                topic, payload = await sub.recv_multipart()
                while True:
                    live = LiveChunk.model_validate(await client.call(GetLive(session="teed")))
                    if live.text == "progress 1\nprogress 2\n":
                        break
                    await anyio.sleep(0.05)
    finally:
        sub.close()
        ctx.term()
    assert (code, out) == (0, b"progress 1\nprogress 2\n")
    assert topic == events.live_topic("teed")
    assert LiveChunk.model_validate_json(payload).text.startswith("progress 1")
    assert live.turn is None
