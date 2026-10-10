"""Discord REST over httpx: channels, messages, and interaction answers.

Writes only; receiving rides the gateway (`aid.plugins.discord.gateway`). A 429 sleeps out what Discord
asked and retries, a few times; anything else raises at once, so auth death (401/403) exits loudly instead
of retrying a token that never heals.
"""

from __future__ import annotations

from typing import Final

import anyio
import httpx
import structlog
from pydantic import BaseModel, JsonValue, TypeAdapter

log = structlog.get_logger(__name__)

API_BASE: Final = "https://discord.com/api/v10"
REQUEST_TIMEOUT: Final = 30.0
"""Seconds one REST call may take: Discord answers fast or not at all."""
_MAX_ATTEMPTS: Final = 3
"""Tries per call: the first plus two honored-429 retries."""


class DiscordError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Discord {status}: {message}")
        self.status = status
        self.message = message


class _GatewayBody(BaseModel):
    url: str


class _ChannelBody(BaseModel):
    id: str


class _MessageBody(BaseModel):
    id: str


class GuildChannel(BaseModel):
    """What a guild channel listing carries: enough to find the category and spot the living."""

    model_config = {"frozen": True}

    id: str
    name: str
    type: int
    """0 text, 4 category, and the rest the plugin never touches."""
    parent_id: str | None = None


def action_rows(rows: tuple[tuple[tuple[str, str], ...], ...]) -> JsonValue:
    """Button rows as message components: (label, custom_id) pairs. Custom ids hold 100 bytes, so unlike
    Telegram's 64, `action:session` fits directly and no token map is needed."""
    return [
        {
            "type": 1,
            "components": [{"type": 2, "style": 1, "label": label, "custom_id": custom} for label, custom in row],
        }
        for row in rows
    ]


class Rest:
    """The REST surface a session bot needs, behind a bot token."""

    def __init__(self, token: str, *, base_url: str = API_BASE, client: httpx.AsyncClient | None = None) -> None:
        self._token = token
        self._base = base_url
        self._client = client if client is not None else httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        self._owned = client is None

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()

    async def _request_once(self, method: str, path: str, payload: JsonValue | None) -> httpx.Response:
        return await self._client.request(
            method,
            f"{self._base}{path}",
            headers={"Authorization": f"Bot {self._token}"},
            json=payload,
        )

    async def _call(self, method: str, path: str, payload: JsonValue | None = None) -> httpx.Response:
        response = await self._request_once(method, path, payload)
        for attempt in range(2, _MAX_ATTEMPTS + 1):
            if response.status_code != 429:
                return response
            retry_after = _retry_after(response)
            log.warning("discord_throttled", path=path, retry_in=retry_after, attempt=attempt - 1)
            await anyio.sleep(retry_after)
            response = await self._request_once(method, path, payload)
        return response

    async def _parse[M: BaseModel](self, model: type[M], method: str, path: str, payload: JsonValue | None = None) -> M:
        response = await self._call(method, path, payload)
        if response.status_code >= 400:
            raise DiscordError(response.status_code, _error_message(response))
        try:
            return model.model_validate_json(response.content)
        except ValueError:
            raise DiscordError(response.status_code, "not Discord JSON") from None

    async def gateway_url(self) -> str:
        """The websocket URL to dial. Called at startup: it also proves the token before any identify."""
        return (await self._parse(_GatewayBody, "GET", "/gateway/bot")).url

    async def guild_channels(self, guild_id: str) -> list[GuildChannel]:
        """Every channel in the guild: the reconcile finds its category and the living by this."""
        response = await self._call("GET", f"/guilds/{guild_id}/channels")
        if response.status_code >= 400:
            raise DiscordError(response.status_code, _error_message(response))
        try:
            return TypeAdapter(list[GuildChannel]).validate_json(response.content)
        except ValueError:
            raise DiscordError(response.status_code, "not Discord JSON") from None

    async def create_channel(
        self, guild_id: str, name: str, *, channel_type: int = 0, parent_id: str | None = None, topic: str | None = None
    ) -> str:
        """A guild channel: text by default, a category with `channel_type=4`. Returns its channel id."""
        payload: dict[str, JsonValue] = {"name": name, "type": channel_type}
        if parent_id is not None:
            payload["parent_id"] = parent_id
        if topic is not None:
            payload["topic"] = topic
        return (await self._parse(_ChannelBody, "POST", f"/guilds/{guild_id}/channels", payload)).id

    async def delete_channel(self, channel_id: str) -> None:
        response = await self._call("DELETE", f"/channels/{channel_id}")
        if response.status_code not in (200, 204):
            raise DiscordError(response.status_code, _error_message(response))

    async def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        components: JsonValue | None = None,
        embeds: list[dict[str, JsonValue]] | None = None,
    ) -> str:
        """A message; returns its id. Absent components send none (Discord rejects an empty array). Embeds pass through when given."""
        payload: dict[str, JsonValue] = {"content": content}
        if components is not None:
            payload["components"] = components
        if embeds is not None:
            payload["embeds"] = embeds
        return (await self._parse(_MessageBody, "POST", f"/channels/{channel_id}/messages", payload)).id

    async def edit_message(
        self, channel_id: str, message_id: str, content: str, *, components: JsonValue | None = None
    ) -> None:
        payload: dict[str, JsonValue] = {"content": content}
        if components is not None:
            payload["components"] = components
        await self._parse(_MessageBody, "PATCH", f"/channels/{channel_id}/messages/{message_id}", payload)

    async def answer_interaction(
        self,
        interaction_id: str,
        interaction_token: str,
        content: str,
        *,
        update: bool = False,
        ephemeral: bool = False,
        components: JsonValue | None = None,
    ) -> None:
        """Answer a button tap: a new message by default, or a rewrite of the tapped one with `update`;
        `ephemeral` shows it to the tapper alone (stale buttons, errors). Type 4 answers in-channel, type 7
        rewrites; both must land within 3 seconds of the tap."""
        data: dict[str, JsonValue] = {"content": content}
        if ephemeral:
            data["flags"] = 64
        if components is not None:
            data["components"] = components
        response = await self._call(
            "POST",
            f"/interactions/{interaction_id}/{interaction_token}/callback",
            {"type": 7 if update else 4, "data": data},
        )
        if response.status_code >= 400:
            raise DiscordError(response.status_code, _error_message(response))


def _retry_after(response: httpx.Response) -> float:
    """Seconds Discord asks to wait: the body's, else the header's, else a second."""
    try:
        asked = response.json().get("retry_after")
    except ValueError:
        asked = None
    if isinstance(asked, (int, float)) and not isinstance(asked, bool) and asked > 0:
        return float(asked)
    try:
        return max(float(response.headers.get("retry-after", 1)), 0)
    except ValueError:
        return 1.0


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json().get("message")
    except ValueError:
        message = None
    return message if isinstance(message, str) else "not ok"
