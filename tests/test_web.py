"""The web UI against a real dex: two static users, one of them on aid's allowlist."""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any, cast

import anyio
import httpx
import pytest
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

import aid
from aid.web.app import ENV_ASSETS
from tests.conftest import GRAMMARS, SPEECH_MODEL, fake_spec, needs_pymux, py_spec
from tests.test_speech import SAID, speech
from tests.web_harness import ALLOWED, ASSETS, IDLE, REFUSED, TIMEOUT, Web, login, needs_dex

if TYPE_CHECKING:
    from pathlib import Path

    from websockets.typing import Origin, Subprotocol

    from aid.paths import Paths

pytestmark = [pytest.mark.anyio, needs_dex]


async def test_login_is_required(web: Web) -> None:
    async with httpx.AsyncClient() as client:
        assert (await client.get(f"{web.url}/api/me")).status_code == 401
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


async def test_the_app_manifest_needs_no_login(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as anonymous:
            manifest = await anonymous.get(f"{web.url}/manifest.webmanifest")
            icons = [await anonymous.get(f"{web.url}{icon['src']}") for icon in manifest.json()["icons"]]
            page = await anonymous.get(f"{web.url}/")
        async with httpx.AsyncClient() as client:
            await login(client, web, ALLOWED)
            index = await client.get(f"{web.url}/")
    assert manifest.headers["content-type"] == "application/manifest+json"
    assert manifest.json()["display"] == "standalone"
    assert [(i.status_code, i.headers["content-type"].split(";")[0]) for i in icons] == [
        (200, "image/svg+xml"),
        (200, "image/png"),
        (200, "image/png"),
        (200, "image/png"),
    ]
    assert page.status_code == 303  # Everything else still needs the login.
    assert '<link rel="manifest" href="/manifest.webmanifest"' in index.text


async def test_session_files(web: Web, tmp_path: Path, daemon: Paths) -> None:
    (tmp_path / "notes.md").write_text("# hi\n")
    (tmp_path / "code.py").write_text("def f(): pass\n")
    with anyio.fail_after(TIMEOUT):
        async with httpx.AsyncClient() as client, aid.connect(daemon) as other:
            anonymous = await client.get(f"{web.url}/api/sessions/files/files")
            await login(client, web, ALLOWED)
            await other.create("files", py_spec(tmp_path, "agents:echo"))
            missing = await client.get(f"{web.url}/api/sessions/nope/files")
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
    assert missing.status_code == 404


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


async def zws_cookie(web: Web) -> dict[str, str]:
    async with httpx.AsyncClient() as client:
        await login(client, web, ALLOWED)
        return {"Cookie": f"aid_session={client.cookies['aid_session']}"}


def zws_connect(web: Web, path: str, headers: dict[str, str], *, origin: str | None = None) -> ws_connect:
    return ws_connect(
        web.url.replace("http://", "ws://") + path,
        origin=cast("Origin", origin or web.url),
        additional_headers=headers,
        subprotocols=[cast("Subprotocol", "ZWS2.0")],
    )


async def zws_recv(ws: ClientConnection) -> list[bytes]:
    """One multipart message from a ZWS relay."""
    parts: list[bytes] = []
    while True:
        frame = await ws.recv()
        assert isinstance(frame, bytes)
        assert not frame[0] & 0x02, "a data frame"
        parts.append(frame[1:])
        if not frame[0] & 0x01:
            return parts


async def zws_call(ws: ClientConnection, request: dict[str, Any]) -> list[dict[str, Any]]:
    """Send one request over the control relay; return its replies up to the last."""
    await ws.send(b"\x00" + json.dumps(request).encode())
    replies: list[dict[str, Any]] = []
    while not replies or replies[-1]["reply"] == "event":
        [reply] = await zws_recv(ws)
        replies.append(json.loads(reply))
    return replies


async def test_zws_control_relays_what_the_page_may_ask(web: Web, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")
    with anyio.fail_after(TIMEOUT):
        headers = await zws_cookie(web)
        async with zws_connect(web, "/api/zws/control", headers) as ws:
            assert ws.subprotocol == "ZWS2.0"
            created = await zws_call(ws, {"op": "create", "id": "c", "name": "echo", "spec": spec})
            prompted = await zws_call(ws, {"op": "prompt", "id": "p", "session": "echo", "text": "hi"})
            hook = await zws_call(ws, {"op": "hook", "id": "h", "session": "echo", "event": "Stop", "payload": {}})
            await ws.send(b"\x02\x04PING\x00\x0ahello")
            ponged = await ws.recv()
            listed = await zws_call(ws, {"op": "list", "id": "l"})
            [page] = await zws_call(ws, {"op": "history", "id": "hi", "session": "echo"})
            [totals] = await zws_call(ws, {"op": "summary", "id": "t", "session": "echo"})
            [unasked] = await zws_call(
                ws, {"op": "answer_permission", "id": "a", "session": "echo", "request_id": "x", "option_id": "yes"}
            )
            [missing] = await zws_call(ws, {"op": "stop", "id": "m", "session": "nope"})
            await zws_call(ws, {"op": "delete", "id": "d", "session": "echo"})
            [after] = await zws_call(ws, {"op": "list", "id": "l2"})
            await ws.send(b"\x01{}")
            await ws.send(b"\x00{}")
            with pytest.raises(ConnectionClosed) as closed:
                await ws.recv()
    assert created == [{"reply": "done", "id": "c", "data": {"name": "echo"}}]
    assert [r["reply"] for r in prompted][-1] == "done"
    assert {r["id"] for r in prompted} == {"p"}
    assert {"type": "output", "output": "turn 1: echo hi", "stop_reason": "end_turn"} in [
        r["event"] for r in prompted if r["reply"] == "event"
    ]
    # Never reaches the daemon: a page may not speak for a claude-tty worker.
    assert (hook[0]["reply"], hook[0]["id"], hook[0]["code"]) == ("failure", "h", "invalid_request")
    assert "does not match any of the expected tags" in hook[0]["message"]
    # The daemon names a turn by its request id: the relay's, never one the page chose.
    assert {e["turn"] for e in page["data"]["entries"] if e["item"]["type"] == "output"} - {"p"}
    assert ponged == b"\x02\x04PONGhello"
    assert listed[0]["data"] == [{"name": "echo", "kind": "pydantic-ai", "running": True, **IDLE}]
    assert [e["item"]["type"] for e in page["data"]["entries"]][-3:] == ["text", "usage", "output"]
    assert (totals["data"]["turns"], totals["data"]["starts"]) == (1, 1)
    assert (unasked["reply"], unasked["code"]) == ("failure", "no_pending")
    assert (missing["reply"], missing["code"]) == ("failure", "not_found")
    assert after["data"] == []
    assert closed.value.rcvd is not None
    assert closed.value.rcvd.code == 1008  # A request of two frames.


async def test_zws_events_follow_subscriptions(web: Web, tmp_path: Path) -> None:
    spec = py_spec(tmp_path, "agents:echo").model_dump(mode="json")
    with anyio.fail_after(TIMEOUT):
        headers = await zws_cookie(web)
        async with (
            zws_connect(web, "/api/zws/events", headers) as events_ws,
            zws_connect(web, "/api/zws/control", headers) as control,
        ):
            await events_ws.send(b"\x00\x01sessions/")
            await anyio.sleep(0.1)  # A subscription reaches the daemon a moment after it is sent.
            await zws_call(control, {"op": "create", "id": "c", "name": "echo", "spec": spec})
            topic, payload = await zws_recv(events_ws)
            await events_ws.send(b"\x00\x00sessions/")
            await events_ws.send(b"\x00\x01session/echo/status/")
            await anyio.sleep(0.1)
            await zws_call(control, {"op": "stop", "id": "s", "session": "echo"})
            stopped = time.monotonic()
            status_topic, status = await zws_recv(events_ws)
            arrived = time.monotonic() - stopped
    assert topic == b"sessions/"
    assert [s["name"] for s in json.loads(payload)] == ["echo"]
    assert status_topic == b"session/echo/status/"
    assert json.loads(status)["running"] is False
    assert arrived < 0.25  # Pushed as it happens.


async def test_zws_events_carry_turns_from_other_clients(web: Web, tmp_path: Path, daemon: Paths) -> None:
    with anyio.fail_after(TIMEOUT):
        headers = await zws_cookie(web)
        async with aid.connect(daemon) as other, zws_connect(web, "/api/zws/events", headers) as events_ws:
            session = await other.create("echo", py_spec(tmp_path, "agents:echo"))
            await events_ws.send(b"\x00\x01session/echo/history/")
            await anyio.sleep(0.1)  # A subscription reaches the daemon a moment after it is sent.
            await session.run("elsewhere")
            seen: list[tuple[int, str]] = []
            while not seen or seen[-1][1] != "output":
                topic, payload = await zws_recv(events_ws)
                entry = json.loads(payload)
                assert topic == b"session/echo/history/"
                seen.append((entry["seq"], entry["item"]["type"]))
    # The worker's start comes first, under a turn of its own.
    assert [kind for _seq, kind in seen] == ["started", "prompt", "text", "usage", "output"]
    assert [seq for seq, _kind in seen] == list(range(seen[0][0], seen[0][0] + 5))


async def test_zws_needs_the_login_the_origin_and_the_subprotocol(web: Web) -> None:
    with anyio.fail_after(TIMEOUT):
        headers = await zws_cookie(web)
        for path in ("/api/zws/control", "/api/zws/events"):
            with pytest.raises(InvalidStatus):
                async with zws_connect(web, path, {"Cookie": "aid_session=not-a-session"}):
                    pass
            with pytest.raises(InvalidStatus):
                async with zws_connect(web, path, headers, origin="http://elsewhere.example"):
                    pass
            with pytest.raises(InvalidStatus):
                async with ws_connect(
                    web.url.replace("http://", "ws://") + path,
                    origin=cast("Origin", web.url),
                    additional_headers=headers,
                ):
                    pass


@needs_pymux
async def test_pane_relay(web: Web, tmp_path: Path, pymux_socket: str, daemon: Paths) -> None:
    url = web.url.replace("http://", "ws://")
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as other:
            await other.create("tty", fake_spec(tmp_path, pymux_socket))
            await other.create("py", py_spec(tmp_path, "agents:echo"))
        headers = await zws_cookie(web)
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
