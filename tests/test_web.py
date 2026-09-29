"""The web UI against a real dex: two static users, one of them on aid's allowlist."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import anyio
import httpx
import pytest
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus

from aid.web import OidcConfig, create_app, serve
from aid.web.app import ENV_ASSETS
from tests.conftest import GRAMMARS, SPEECH_MODEL, fake_spec, needs_pymux, py_spec
from tests.test_speech import SAID, speech

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from websockets.typing import Origin

    from aid.paths import Paths
    from aid.speech import Recognizer
    from aid.web.highlight import Grammars

# The built Svelte UI. The Nix test run and the dev shell set it; without it the API tests still run.
ASSETS = Path(os.environ[ENV_ASSETS]) if os.environ.get(ENV_ASSETS) else None

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(shutil.which("dex") is None, reason="dex is not on PATH")]

TIMEOUT = 30
ALLOWED = "admin@example.com"
REFUSED = "other@example.com"
# bcrypt of "password", from dex's example configuration.
PASSWORD_HASH = "$2a$10$2b2cU8CPhOTaGrs1HRQuAueS7JTT5ZHsHSzYiFPm1leZck7Mc8T4W"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass(frozen=True)
class Web:
    url: str


def dex_config(port: int, web_url: str) -> dict[str, Any]:
    users = [
        {"email": email, "hash": PASSWORD_HASH, "username": email.split("@")[0], "userID": f"user-{i}"}
        for i, email in enumerate((ALLOWED, REFUSED))
    ]
    return {
        "issuer": f"http://127.0.0.1:{port}/dex",
        "storage": {"type": "memory"},
        "web": {"http": f"127.0.0.1:{port}"},
        "oauth2": {"skipApprovalScreen": True},
        "staticClients": [
            {"id": "aid", "secret": "aid-secret", "name": "aid", "redirectURIs": [f"{web_url}/auth/callback"]}
        ],
        "enablePasswordDB": True,
        "staticPasswords": users,
    }


def open_log(path: Path) -> BinaryIO:
    return path.open("wb")


async def wait_for(url: str) -> None:
    async with httpx.AsyncClient() as client:
        while True:
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await anyio.sleep(0.1)


@pytest.fixture
async def web(
    daemon: Paths, tmp_path: Path, speech_recognizer: Recognizer | None, grammars: Grammars | None
) -> AsyncIterator[Web]:
    dex_port, web_port = free_port(), free_port()
    web_url = f"http://127.0.0.1:{web_port}"
    config = tmp_path / "dex.json"
    config.write_text(json.dumps(dex_config(dex_port, web_url)))
    issuer = f"http://127.0.0.1:{dex_port}/dex"
    log = open_log(tmp_path / "dex.log")
    oidc = OidcConfig(
        issuer=issuer,
        client_id="aid",
        client_secret="aid-secret",
        base_url=web_url,
        allowed_emails=frozenset({ALLOWED}),
    )
    shutdown = anyio.Event()
    async with await anyio.open_process(["dex", "serve", str(config)], stdout=log, stderr=log) as dex:
        try:
            with anyio.fail_after(TIMEOUT):
                await wait_for(f"{issuer}/.well-known/openid-configuration")
            async with anyio.create_task_group() as tg:
                app = create_app(
                    oidc,
                    secrets.token_hex(32),
                    daemon,
                    assets=ASSETS,
                    recognizer=speech_recognizer,
                    grammars=grammars,
                )
                tg.start_soon(lambda: serve(app, f"127.0.0.1:{web_port}", shutdown=shutdown))
                with anyio.fail_after(TIMEOUT):
                    await wait_for(f"{web_url}/healthz")
                yield Web(web_url)
                shutdown.set()
        finally:
            dex.terminate()
            log.close()


async def login(client: httpx.AsyncClient, web: Web, email: str) -> httpx.Response:
    """Walk the authorization code flow like a browser; return the callback's response."""
    response = await client.get(f"{web.url}/login")
    while response.is_redirect:
        response = await client.get(response.url.join(response.headers["location"]))
    match = re.search(r'<form[^>]*action="([^"]*)"', response.text)
    assert match, response.text[:500]
    action = response.url.join(match.group(1).replace("&amp;", "&"))
    response = await client.post(action, data={"login": email, "password": "password"})
    while response.is_redirect:
        response = await client.get(response.url.join(response.headers["location"]))
        if response.url.path == "/auth/callback":
            return response
    raise AssertionError(f"dex did not send the browser back: {response.status_code} {response.text[:300]}")


async def events(response: httpx.Response) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            found.append(json.loads(line.removeprefix("data: ")))
    return found


async def test_login_is_required(web: Web) -> None:
    async with httpx.AsyncClient() as client:
        assert (await client.get(f"{web.url}/api/sessions")).status_code == 401
        index = await client.get(f"{web.url}/")
        health = await client.get(f"{web.url}/healthz")
    assert index.status_code == 303
    assert index.headers["location"] == "/login"
    assert "default-src 'self'" in health.headers["content-security-policy"]


@pytest.mark.skipif(ASSETS is None, reason=f"{ENV_ASSETS} is not set")
async def test_ui_is_served(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            index = await client.get(f"{web.url}/")
            scripts = re.findall(r'src="(/assets/[^"]+\.js)"', index.text)
            assert scripts, index.text
            script = await client.get(f"{web.url}{scripts[0]}")
            colors = await client.get(f"{web.url}/theme.css")
    assert index.status_code == 200
    assert '<div id="app">' in index.text
    assert 'href="/theme.css"' in index.text
    assert colors.headers["content-type"].startswith("text/css")
    assert ".hl .k {" in colors.text
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]


async def test_session_round_trip(web: Web, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            callback = await login(client, web, ALLOWED)
            assert callback.status_code == 303
            me = (await client.get(f"{web.url}/api/me")).json()
            assert me["email"] == ALLOWED
            headers = {"X-CSRF-Token": me["csrf"]}
            spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")

            refused = await client.post(f"{web.url}/api/sessions", json={"name": "echo", "spec": spec})
            assert refused.status_code == 403

            created = await client.post(f"{web.url}/api/sessions", json={"name": "echo", "spec": spec}, headers=headers)
            assert created.status_code == 201, created.text
            async with client.stream(
                "POST", f"{web.url}/api/sessions/echo/prompt", json={"text": "hi"}, headers=headers
            ) as response:
                assert response.headers["content-type"].startswith("text/event-stream")
                streamed = await events(response)
            history = (await client.get(f"{web.url}/api/sessions/echo/history?limit=2")).json()
            bad_query = await client.get(f"{web.url}/api/sessions/echo/history?limit=many")
            listed = (await client.get(f"{web.url}/api/sessions")).json()
            deleted = await client.delete(f"{web.url}/api/sessions/echo", headers=headers)
            after = (await client.get(f"{web.url}/api/sessions")).json()

    assert streamed[-1] == {"type": "output", "output": "turn 1: echo hi", "stop_reason": "end_turn"}
    assert {"type": "text", "text": "echo hi"} in streamed
    assert [e["item"]["type"] for e in history["entries"]] == ["text", "output"]
    assert history["has_older"] is True
    assert bad_query.status_code == 422
    assert listed == [{"name": "echo", "kind": "pydantic-ai", "running": True}]
    assert deleted.status_code == 200
    assert after == []


async def test_session_files(web: Web, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# hi\n")
    (tmp_path / "code.py").write_text("def f(): pass\n")
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            anonymous = await client.get(f"{web.url}/api/sessions/files/files")
            await login(client, web, ALLOWED)
            headers = {"X-CSRF-Token": (await client.get(f"{web.url}/api/me")).json()["csrf"]}
            spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")
            await client.post(f"{web.url}/api/sessions", json={"name": "files", "spec": spec}, headers=headers)
            listed = (await client.get(f"{web.url}/api/sessions/files/files")).json()
            read = (await client.get(f"{web.url}/api/sessions/files/file", params={"path": "notes.md"})).json()
            code = (await client.get(f"{web.url}/api/sessions/files/file", params={"path": "code.py"})).json()
            escape = await client.get(f"{web.url}/api/sessions/files/file", params={"path": "../x"})
    assert anonymous.status_code == 401
    assert {"name": "notes.md", "dir": False, "size": 5} in listed
    assert read == {"path": "notes.md", "size": 5, "text": "# hi\n", "truncated": False, "highlights": None}
    if GRAMMARS is not None:
        assert code["highlights"][:2] == [[0, 3, "k"], [4, 5, "nf"]]
    assert escape.status_code == 403


async def test_unknown_session_is_404(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            csrf = (await client.get(f"{web.url}/api/me")).json()["csrf"]
            response = await client.post(f"{web.url}/api/sessions/nope/stop", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 404


async def test_account_off_the_allowlist_is_refused(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            callback = await login(client, web, REFUSED)
            me = await client.get(f"{web.url}/api/me")
    assert callback.status_code == 403
    assert me.status_code == 401


async def test_logout(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            csrf = (await client.get(f"{web.url}/api/me")).json()["csrf"]
            assert (await client.post(f"{web.url}/logout")).status_code == 403
            assert (await client.post(f"{web.url}/logout", headers={"X-CSRF-Token": csrf})).status_code == 200
            me = await client.get(f"{web.url}/api/me")
    assert me.status_code == 401


async def test_status_streams_changes(web: Web, tmp_path: Path) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            csrf = (await client.get(f"{web.url}/api/me")).json()["csrf"]
            spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")
            await client.post(
                f"{web.url}/api/sessions", json={"name": "echo", "spec": spec}, headers={"X-CSRF-Token": csrf}
            )
            snapshot = (await client.get(f"{web.url}/api/sessions/echo/status")).json()
            missing = await client.get(f"{web.url}/api/sessions/nope/status/events")
            seen: list[dict[str, Any]] = []
            no_csrf: httpx.Response | None = None
            async with client.stream("GET", f"{web.url}/api/sessions/echo/status/events") as response:
                lines = response.aiter_lines()
                async for line in lines:
                    if line.startswith("data: "):
                        seen.append(json.loads(line.removeprefix("data: ")))
                        if len(seen) == 1:
                            await client.post(f"{web.url}/api/sessions/echo/stop", headers={"X-CSRF-Token": csrf})
                        elif len(seen) == 2:
                            no_csrf = await client.post(f"{web.url}/api/sessions/echo/start")
                            await client.post(f"{web.url}/api/sessions/echo/start", headers={"X-CSRF-Token": csrf})
                        else:
                            break
    assert snapshot["running"] is True
    assert missing.status_code == 404
    assert no_csrf is not None
    assert no_csrf.status_code == 403
    assert [s["running"] for s in seen] == [True, False, True]


async def speak(web: Web, cookie: str, *, origin: str | None = None) -> list[dict[str, Any]]:
    """Send the speech model's sample over /api/transcribe, as a page would; return every reply."""
    rate, samples = speech()
    url = web.url.replace("http://", "ws://") + "/api/transcribe"
    headers = {"Cookie": f"aid_session={cookie}"}
    replies: list[dict[str, Any]] = []
    async with ws_connect(url, origin=cast("Origin", origin or web.url), additional_headers=headers) as ws:
        await ws.send(json.dumps({"rate": rate}))
        step = rate // 10
        for start in range(0, len(samples), step):
            await ws.send(samples[start : start + step].astype("<f4").tobytes())
        await ws.send(json.dumps({"end": True}))
        async for message in ws:
            replies.append(json.loads(message))
    return replies


@pytest.mark.skipif(SPEECH_MODEL is None, reason="AID_TEST_SPEECH_MODEL is not set")
async def test_speech_to_text(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            enabled = (await client.get(f"{web.url}/api/speech")).json()
            cookie = client.cookies["aid_session"]
        replies = await speak(web, cookie)
        with pytest.raises(InvalidStatus):
            await speak(web, cookie, origin="http://elsewhere.example")
        with pytest.raises(InvalidStatus):
            await speak(web, "not-a-session")
    assert enabled == {"enabled": True}
    assert replies[-1] == {"done": True}
    assert " ".join(r["text"] for r in replies if r.get("final")) == SAID
    assert any(r.get("final") is False for r in replies)


@needs_pymux
async def test_pane_relay(web: Web, tmp_path: Path, pymux_socket: str) -> None:
    url = web.url.replace("http://", "ws://")
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            csrf = (await client.get(f"{web.url}/api/me")).json()["csrf"]
            spec = fake_spec(tmp_path, pymux_socket).model_dump(mode="json")
            await client.post(
                f"{web.url}/api/sessions", json={"name": "tty", "spec": spec}, headers={"X-CSRF-Token": csrf}
            )
            spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")
            await client.post(
                f"{web.url}/api/sessions", json={"name": "py", "spec": spec}, headers={"X-CSRF-Token": csrf}
            )
            headers = {"Cookie": f"aid_session={client.cookies['aid_session']}"}
        origin = cast("Origin", web.url)
        async with ws_connect(f"{url}/api/sessions/tty/pane", origin=origin, additional_headers=headers) as ws:
            welcome = json.loads(await ws.recv())
            first = json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "text", "text": "typed-in-the-browser"}))
            seen = ""
            while "typed-in-the-browser" not in seen:
                frame = json.loads(await ws.recv())
                seen = json.dumps(frame.get("rows", {}))
        with pytest.raises(InvalidStatus):
            async with ws_connect(
                f"{url}/api/sessions/tty/pane",
                origin=cast("Origin", "http://elsewhere.example"),
                additional_headers=headers,
            ):
                pass
        with pytest.raises(InvalidStatus):
            async with ws_connect(f"{url}/api/sessions/py/pane", origin=origin, additional_headers=headers):
                pass
    assert welcome["type"] == "welcome"
    assert welcome["writable"] is True
    assert "background-color" in welcome["css"]
    assert first["type"] == "frame"
    assert first.get("whole") is True
