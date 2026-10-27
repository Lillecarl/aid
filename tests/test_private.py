from __future__ import annotations

import re
from typing import TYPE_CHECKING

import anyio
import pytest
import zmq
from pydantic import ValidationError

import aid
from aid import events
from aid.client import register_plugin
from aid.model import AnthropicModel, GoogleModel, OpenAIChatModel, check_host
from aid.plugins import Grant
from aid.protocol import AidError, CallPlugin, GetHistory, SendMessage, SessionInfosAdapter
from aid.spec import AcpSpec, NativeSpec, PermissionMode, Private
from tests.conftest import py_spec
from tests.models import HOST
from tests.test_agent_control import RecordingLauncher, agent_client, server

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

__all__ = ["server"]

TIMEOUT = 30
HOSTS = Private(hosts=[HOST])


def private_spec(tmp_path: Path, target: str = "agents:echo", **fields: object) -> NativeSpec:
    spec = py_spec(tmp_path, target, PermissionMode.DENY)
    return NativeSpec.model_validate({**spec.model_dump(), "private": HOSTS.model_dump(), **fields})


def test_private_is_native_only_and_names_bare_hosts() -> None:
    with pytest.raises(ValidationError):
        AcpSpec.model_validate({"cwd": "/", "command": ["x"], "private": HOSTS.model_dump()})
    for hosts in ([], ["https://vllm.example"], ["vllm.example:8000"], ["VLLM.example"]):
        with pytest.raises(ValidationError):
            Private(hosts=hosts)


async def test_a_transport_names_the_host_its_client_sends_to(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(variable, "fake")
    vllm: dict[str, JsonValue] = {"api_base": "https://vllm.example/v1"}
    cases: list[tuple[OpenAIChatModel | AnthropicModel | GoogleModel, dict[str, JsonValue], str]] = [
        (OpenAIChatModel("vllm/glm"), vllm, "vllm.example"),
        (OpenAIChatModel("vllm/glm"), {}, "localhost"),
        (OpenAIChatModel("openai/gpt-x"), {}, "api.openai.com"),
        (AnthropicModel("anthropic/claude-haiku-5-5"), {}, "api.anthropic.com"),
        (AnthropicModel("anthropic/claude-haiku-5-5"), vllm, "vllm.example"),
        (GoogleModel("google/gemini-3.1-flash-lite"), {}, "generativelanguage.googleapis.com"),
    ]
    try:
        assert [model.host(settings) for model, settings, _ in cases] == [host for _, _, host in cases]
        check_host(cases[0][0], vllm, ["vllm.example"])
        with pytest.raises(AidError, match="'localhost', which is not one of"):
            check_host(cases[1][0], {}, ["vllm.example"])
        with pytest.raises(AidError, match="cannot say which host"):
            check_host(object(), {}, ["vllm.example"])
    finally:
        for model, _, _ in cases:
            await model.aclose()


async def test_a_private_session_on_an_unlisted_host_never_starts(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, _ = server
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(paths) as control:
            elsewhere = private_spec(tmp_path, private={"hosts": ["vllm.example"]})
            with pytest.raises(AidError, match=re.escape(f"sends to {HOST!r}")):
                await control.create("elsewhere", elsewhere)


async def test_privacy_holds_across_parents_and_supervisors(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, _ = server
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(paths) as control:
            await control.create("open", py_spec(tmp_path, "agents:echo"))
            await control.create("secret", private_spec(tmp_path))
            for name, spec in (
                ("open.child", private_spec(tmp_path, parent="open")),
                ("secret.child", py_spec(tmp_path, "agents:echo").model_copy(update={"parent": "secret"})),
                ("watched", private_spec(tmp_path, supervisor="open")),
            ):
                with pytest.raises(AidError) as refused:
                    await control.create(name, spec)
                assert refused.value.code == "private"
            await control.create("secret.child", private_spec(tmp_path, parent="secret"))
            # A private session may watch an open one: what it learns stays private.
            await control.create(
                "overseen", py_spec(tmp_path, "agents:echo").model_copy(update={"supervisor": "secret"})
            )
            with pytest.raises(AidError, match="private supervisor"):
                await control.session("secret.child").set_supervisor("open")
            infos = {info.name: info.private for info in await control.sessions()}
    assert infos == {"open": False, "secret": True, "secret.child": True, "overseen": False}


async def test_a_plugin_without_the_grant_never_sees_a_private_session(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, _ = server
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(paths) as control:
            await control.create("open", py_spec(tmp_path, "agents:echo"))
            await control.create("secret", private_spec(tmp_path))
            await register_plugin(control, paths, "chat", frozenset({Grant.READ, Grant.MESSAGE, Grant.MANAGE}))
            await register_plugin(control, paths, "keeper", frozenset({Grant.READ, Grant.PRIVATE}))
            async with aid.connect(paths, plugin="chat") as chat, aid.connect(paths, plugin="keeper") as keeper:
                assert [info.name for info in await chat.sessions()] == ["open"]
                for call in (
                    chat.session("secret").status(),
                    chat.call(GetHistory(session="secret")),
                    chat.send_message("secret", "hello"),
                    chat.create(
                        "secret.child", py_spec(tmp_path, "agents:echo").model_copy(update={"parent": "secret"})
                    ),
                ):
                    with pytest.raises(AidError) as hidden:
                        await call
                    assert (hidden.value.code, hidden.value.message) == ("not_found", "no session named 'secret'")
                with pytest.raises(AidError, match="forbidden"):
                    await chat.create("another", private_spec(tmp_path))
                assert sorted(info.name for info in await keeper.sessions()) == ["open", "secret"]
                assert (await keeper.session("secret").status()).private


async def test_only_plugins_granted_private_hear_a_private_session(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, _ = server
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(paths) as control:
            await register_plugin(control, paths, "chat", frozenset({Grant.READ}))
            await register_plugin(control, paths, "keeper", frozenset({Grant.READ, Grant.PRIVATE}))
            async with (
                aid.connect(paths, plugin="chat") as chat,
                aid.connect(paths, plugin="keeper") as keeper,
                chat.open_socket(zmq.SUB) as chat_raw,
                keeper.open_socket(zmq.SUB) as keeper_raw,
            ):
                for sock in (chat_raw, keeper_raw):
                    sock.subscribe(b"")
                await anyio.sleep(0.3)  # PUB drops what a subscriber not yet joined would get.
                await control.create("secret", private_spec(tmp_path))
                await control.session("secret").run("hush")
                await control.create("open", py_spec(tmp_path, "agents:echo"))
                await control.session("open").run("hello")
                heard: dict[str, list[bytes]] = {"chat": [], "keeper": []}
                lists: dict[str, list[str]] = {"chat": [], "keeper": []}
                for who, sock in (("chat", chat_raw), ("keeper", keeper_raw)):
                    while not any(t.startswith(b"session/open/history/") for t in heard[who]) or not lists[who]:
                        topic, payload = await sock.recv_multipart()
                        heard[who].append(topic)
                        if topic == events.SESSIONS:
                            lists[who] = [info.name for info in SessionInfosAdapter.validate_json(payload)]
                    while "open" not in lists[who]:
                        topic, payload = await sock.recv_multipart()
                        if topic == events.SESSIONS:
                            lists[who] = [info.name for info in SessionInfosAdapter.validate_json(payload)]
    assert not [t for t in heard["chat"] if t.startswith(b"session/secret/")]
    assert lists["chat"] == ["open"]
    assert any(t.startswith(b"session/secret/history/") for t in heard["keeper"])
    assert sorted(lists["keeper"]) == ["open", "secret"]


async def test_a_private_session_talks_only_to_private_sessions_and_plugins(
    server: tuple[Paths, RecordingLauncher], tmp_path: Path
) -> None:
    paths, launcher = server
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(paths) as control:
            await control.create("open", py_spec(tmp_path, "agents:echo"))
            await control.create("peer", private_spec(tmp_path))
            await control.create("secret", private_spec(tmp_path, "agents:control_probe"))
            secret_worker = launcher.launches[-1]
            await control.create("nosy", py_spec(tmp_path, "agents:control_probe"))
            nosy_worker = launcher.launches[-1]
            await register_plugin(control, paths, "chat", frozenset({Grant.READ}))
            with pytest.raises(AidError, match="messages only private"):
                await control.call(SendMessage(to="open", text="leak", sender="secret"))
            with pytest.raises(AidError, match="calls only plugins granted private"):
                await control.call(CallPlugin(plugin="chat", method="anything", caller="secret"))
            async with agent_client(secret_worker) as secret:
                with pytest.raises(AidError, match="messages only private"):
                    await secret.send_message("open", "leak")
                await secret.send_message("peer", "fine")
                assert sorted(info.name for info in await secret.sessions()) == ["nosy", "open", "peer", "secret"]
            async with agent_client(nosy_worker) as nosy:
                assert sorted(info.name for info in await nosy.sessions()) == ["nosy", "open"]
                with pytest.raises(AidError, match="no session named 'peer'"):
                    await nosy.call(GetHistory(session="peer"))
                await nosy.send_message("open", "fine")
