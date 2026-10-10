"""The Discord REST client: exact paths and bodies, auth on every call, and the 429 policy."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from plugins.discord.rest import DiscordError, Rest, action_rows

pytestmark = pytest.mark.anyio

TOKEN = "tok"


def _transport(
    calls: list[tuple[str, str, Any, str | None]], script: dict[tuple[str, str], tuple[int, Any]]
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body, request.headers.get("authorization")))
        status, result = script.get((request.method, request.url.path), (404, {"message": "nope"}))
        return httpx.Response(status) if result is None else httpx.Response(status, json=result)

    return httpx.MockTransport(handler)


async def test_calls_post_exact_paths_bodies_and_auth() -> None:
    calls: list[tuple[str, str, Any, str | None]] = []
    script = {
        ("GET", "/api/v10/gateway/bot"): (200, {"url": "wss://g/"}),
        ("POST", "/api/v10/guilds/g1/channels"): (201, {"id": "c1"}),
        ("POST", "/api/v10/channels/c1/messages"): (200, {"id": "m1"}),
        ("PATCH", "/api/v10/channels/c1/messages/m1"): (200, {"id": "m1"}),
        ("DELETE", "/api/v10/channels/c1"): (204, None),
        ("POST", "/api/v10/interactions/i1/t1/callback"): (204, None),
        ("POST", "/api/v10/interactions/i2/t2/callback"): (204, None),
    }
    keyboard = action_rows(((("Status", "status:echo"),),))
    async with httpx.AsyncClient(transport=_transport(calls, script)) as http:
        rest = Rest(TOKEN, client=http)
        assert await rest.gateway_url() == "wss://g/"
        assert await rest.create_channel("g1", "echo", parent_id="cat", topic="idle") == "c1"
        assert await rest.send_message("c1", "hi") == "m1"
        await rest.edit_message("c1", "m1", "ho")
        await rest.delete_channel("c1")
        await rest.answer_interaction("i1", "t1", "done", update=True)
        await rest.answer_interaction("i2", "t2", "fresh", update=True, components=keyboard)
    assert calls
    assert all(auth == "Bot tok" for _, _, _, auth in calls)
    assert calls[1] == (
        "POST",
        "/api/v10/guilds/g1/channels",
        {"name": "echo", "type": 0, "parent_id": "cat", "topic": "idle"},
        "Bot tok",
    )
    assert calls[2] == ("POST", "/api/v10/channels/c1/messages", {"content": "hi"}, "Bot tok")
    assert calls[5] == (
        "POST",
        "/api/v10/interactions/i1/t1/callback",
        {"type": 7, "data": {"content": "done"}},
        "Bot tok",
    )
    assert calls[6] == (
        "POST",
        "/api/v10/interactions/i2/t2/callback",
        {"type": 7, "data": {"content": "fresh", "components": keyboard}},
        "Bot tok",
    )


async def test_throttled_calls_wait_and_retry() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"message": "slow", "retry_after": 0, "global": False})
        return httpx.Response(200, json={"url": "wss://g/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await Rest(TOKEN, client=http).gateway_url() == "wss://g/"
    assert attempts == 3


async def test_throttled_past_retrying_raises() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, json={"message": "slow", "retry_after": 0, "global": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(DiscordError, match="slow"):
            await Rest(TOKEN, client=http).gateway_url()
    assert attempts == 3


async def test_refused_calls_raise_at_once() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"message": "Unauthorized", "code": 0})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(DiscordError, match="Unauthorized"):
            await Rest(TOKEN, client=http).gateway_url()
    assert attempts == 1


def test_action_rows_build_buttons() -> None:
    assert action_rows(((("Status", "status:echo"), ("Tail", "tail:echo")),)) == [
        {
            "type": 1,
            "components": [
                {"type": 2, "style": 1, "label": "Status", "custom_id": "status:echo"},
                {"type": 2, "style": 1, "label": "Tail", "custom_id": "tail:echo"},
            ],
        }
    ]
