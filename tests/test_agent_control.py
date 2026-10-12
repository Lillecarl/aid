from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq

import aid
from aid.control import ENV_AGENT_CONNECTION, AgentConnection, authorize
from aid.daemon import Daemon
from aid.env import agent_environment
from aid.launcher import WorkerArgs
from aid.paths import ENV_SESSION
from aid.protocol import (
    AddGoal,
    AidError,
    Cancel,
    GetGoals,
    GetHistory,
    GetStatus,
    Hello,
    ListSessions,
    MessageEntry,
    ReceiveMessages,
    RequestAdapter,
    Started,
    SupervisedHistory,
    decode_reply,
    encode,
)
from tests.conftest import py_spec
from tests.test_hook import run_hook

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path

    from aid.launcher import ForkserverLauncher, Launcher, WorkerHandle
    from aid.paths import Paths

pytestmark = pytest.mark.anyio


@dataclass
class RecordingLauncher:
    base: Launcher
    launches: list[WorkerArgs] = field(default_factory=list[WorkerArgs])

    async def launch(self, args: WorkerArgs) -> WorkerHandle:
        self.launches.append(args)
        return await self.base.launch(args)


@pytest.fixture
async def server(paths: Paths, launcher: ForkserverLauncher) -> AsyncIterator[tuple[Paths, RecordingLauncher]]:
    recorder = RecordingLauncher(launcher)
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, recorder, plugin_dirs=[]).serve)
        yield paths, recorder
        tg.cancel_scope.cancel()


@asynccontextmanager
async def agent_client(args: WorkerArgs) -> AsyncGenerator[aid.Client]:
    connection = AgentConnection(
        endpoint=args.endpoint,
        server_key=args.server_key,
        public_key=args.agent_public_key,
        secret_key=args.agent_secret_key,
        trust_pem=args.trust_pem,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(ENV_AGENT_CONNECTION, connection.model_dump_json())
        async with aid.connect() as client:
            yield client


async def test_agent_identity_rejects_supervisor_spoofing_and_administration(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, launcher = server
    with anyio.fail_after(30), pytest.MonkeyPatch.context() as patch:
        async with aid.connect(paths) as control:
            await control.create("boss", py_spec(tmp_path, "agents:echo"))
            await control.create("worker", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            intruder = await control.create("intruder", py_spec(tmp_path, "agents:control_probe"))
            args = launcher.launches[-1]
            async with agent_client(args) as client:
                patch.setenv(ENV_SESSION, "boss")
                with pytest.raises(AidError, match="forbidden"):
                    await client.call(AddGoal(supervisor="boss", session="worker", text="spoofed"))
                with pytest.raises(AidError, match="forbidden"):
                    await client.call(SupervisedHistory(supervisor="boss", session="worker"))
                with pytest.raises(AidError, match="forbidden"):
                    await client.call(Cancel(session="worker"))
                await client.call(GetHistory(session="worker"))
                with pytest.raises(AidError, match="forbidden"):
                    await client.call(ReceiveMessages(session="worker", wait=0))
                with pytest.raises(AidError, match="forbidden"):
                    await client.create("unapproved", py_spec(tmp_path, "agents:echo"))
                await client.send_message("worker", "authored by the intruder", sender="boss")
                async with client.open_socket(zmq.DEALER) as raw:
                    await raw.send(encode(Hello(started=Started(pid=1, agent="forged"))))
                    response = decode_reply(await raw.recv())
                    assert response.reply == "failure"
                async with client.open_socket(zmq.SUB):
                    pass
            patch.delenv(ENV_SESSION)
            assert await control.call(GetGoals(supervisor="boss", session="worker")) == []
            entries = (await control.session("worker").history()).entries
            messages = [entry.item for entry in entries if isinstance(entry.item, MessageEntry)]
            assert messages == [MessageEntry(sender="intruder", text="authored by the intruder")]
            assert (await intruder.status()).agent != "forged"
            result = await intruder.run(
                json.dumps(
                    [
                        [
                            "control_call",
                            {
                                "request": {
                                    "op": "goal_add",
                                    "supervisor": "boss",
                                    "session": "worker",
                                    "text": "spoofed",
                                }
                            },
                        ]
                    ]
                )
            )
            assert json.loads(str(result.output)) == {"error": "forbidden"}


async def test_supervisor_credentials_allow_only_its_supervision_links(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, launcher = server
    with anyio.fail_after(30):
        async with aid.connect(paths) as control:
            await control.create("boss", py_spec(tmp_path, "agents:echo"))
            args = launcher.launches[-1]
            await control.create("worker", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "boss"}))
            await control.create("other", py_spec(tmp_path, "agents:echo"))
            async with agent_client(args) as client:
                goal = await client.call(AddGoal(supervisor="boss", session="worker", text="authorized"))
                assert isinstance(goal, dict) and goal["text"] == "authorized"
                with pytest.raises(AidError, match="not_supervised"):
                    await client.call(AddGoal(supervisor="boss", session="other", text="unauthorized"))
                await client.call(GetStatus(session="boss"))


async def test_revocation_rejects_an_existing_connection_after_worker_restart(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, launcher = server
    with anyio.fail_after(30):
        async with aid.connect(paths) as control:
            session = await control.create("agent", py_spec(tmp_path, "agents:echo"))
            old = launcher.launches[-1]
            async with agent_client(old) as client:
                await client.call(ListSessions())
                await session.stop()
                with pytest.raises(AidError, match="unauthenticated"):
                    await client.call(ListSessions())
                await session.start()
                fresh = launcher.launches[-1]
                assert fresh.agent_public_key != old.agent_public_key
                with pytest.raises(AidError, match="unauthenticated"):
                    await client.call(ListSessions())
                async with agent_client(fresh) as current:
                    await current.call(ListSessions())


async def test_session_marker_without_credentials_cannot_fall_back_to_administration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_SESSION, "pretend")
    monkeypatch.delenv(ENV_AGENT_CONNECTION, raising=False)
    with pytest.raises(AidError, match="unauthenticated"):
        async with aid.connect():
            pytest.fail("unauthenticated agent reached control")
    monkeypatch.setenv(ENV_AGENT_CONNECTION, "invalid")
    with pytest.raises(AidError, match="unauthenticated"):
        async with aid.connect():
            pytest.fail("invalid credentials fell back to control")


async def test_a_revoked_hook_connection_fails_open_without_waiting_for_permission(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, launcher = server
    with anyio.fail_after(5):
        async with aid.connect(paths) as control:
            session = await control.create("agent", py_spec(tmp_path, "agents:echo"))
            old = launcher.launches[-1]
            await session.stop()
            async with agent_client(old):
                code, output, _ = await run_hook(paths.control, "agent", "PermissionRequest")
                assert (code, output) == (0, "")


@pytest.mark.parametrize("inherit", [True, False])
def test_external_agent_environment_preserves_issued_credentials(
    monkeypatch: pytest.MonkeyPatch, inherit: bool
) -> None:
    monkeypatch.setenv(ENV_AGENT_CONNECTION, "issued credentials")
    monkeypatch.setenv(ENV_SESSION, "actual")
    env = agent_environment({ENV_AGENT_CONNECTION: "forged", ENV_SESSION: "other"}, inherit=inherit)
    assert env[ENV_AGENT_CONNECTION] == "issued credentials"
    assert env[ENV_SESSION] == "actual"


@pytest.mark.parametrize(
    "op",
    ["prompt", "cancel", "start", "stop", "delete", "answer_permission", "add_plugin", "plugins", "remove_plugin"],
)
def test_agent_requests_cannot_use_administrator_authority(op: str) -> None:
    bodies = {
        "prompt": {"session": "actor", "text": "self-directed prompt"},
        "cancel": {"session": "actor"},
        "start": {"session": "actor"},
        "stop": {"session": "actor"},
        "delete": {"session": "actor"},
        "answer_permission": {"session": "actor", "request_id": "permission", "option_id": "allow"},
        "add_plugin": {"spec": {"name": "actor", "public_key": "a" * 40, "grants": ["manage"]}},
        "plugins": {},
        "remove_plugin": {"name": "actor"},
    }
    request = RequestAdapter.validate_python({"op": op, **bodies[op]})
    with pytest.raises(AidError, match="forbidden"):
        authorize(request, "actor")
