"""The web UI's server side for tests: real dex, real Starlette app, real daemon.

Both the HTTP tests (`test_web.py`) and the browser tests (`test_browser.py`) boot the same stack; the
browser ones then drive it through a real browser instead of httpx.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import anyio
import httpx
import pytest

from aid.web import OidcConfig, create_app, serve
from aid.web.app import ENV_ASSETS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from aid.paths import Paths
    from aid.speech import Recognizer
    from aid.web.highlight import Grammars
# The built Svelte UI. The Nix test run and the dev shell set it; without it the API tests still run.
ASSETS = Path(os.environ[ENV_ASSETS]) if os.environ.get(ENV_ASSETS) else None

needs_dex = pytest.mark.skipif(shutil.which("dex") is None, reason="dex is not on PATH")

TIMEOUT = int(os.environ.get("AID_TEST_TIMEOUT", "30"))
"""Seconds a wait may take: the default assumes host speed, and a slow runner such as a UML guest
sets AID_TEST_TIMEOUT higher."""
ALLOWED = "admin@example.com"
IDLE = {"permissions": 0, "working": False, "attention": None}
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


@dataclass(frozen=True)
class Dex:
    issuer: str


@asynccontextmanager
async def running_dex(tmp_path: Path, web_url: str) -> AsyncGenerator[Dex]:
    """Dex as the web fixture boots it, for a test that owns its own app and server."""
    port = free_port()
    config = tmp_path / "dex.json"
    config.write_text(json.dumps(dex_config(port, web_url)))
    issuer = f"http://127.0.0.1:{port}/dex"
    log = open_log(tmp_path / "dex.log")
    async with await anyio.open_process(["dex", "serve", str(config)], stdout=log, stderr=log) as dex:
        try:
            with anyio.fail_after(TIMEOUT):
                await wait_for(f"{issuer}/.well-known/openid-configuration")
            yield Dex(issuer)
        finally:
            dex.terminate()
            log.close()


@pytest.fixture
async def web(
    daemon: Paths, tmp_path: Path, speech_recognizer: Recognizer | None, grammars: Grammars | None
) -> AsyncIterator[Web]:
    web_port = free_port()
    web_url = f"http://127.0.0.1:{web_port}"
    async with running_dex(tmp_path, web_url) as dex_server:
        oidc = OidcConfig(
            issuer=dex_server.issuer,
            client_id="aid",
            client_secret="aid-secret",
            base_url=web_url,
            allowed_emails=frozenset({ALLOWED}),
        )
        shutdown = anyio.Event()
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
