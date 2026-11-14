"""Plain HTTP for aid's own model clients (#65): a streaming POST, its server-sent events, and its refusals.

No model SDK: each protocol module (`aid_models.gemini`, …) types the fields aid sends and reads, and the transports in
`aid.model` name the failures (`unavailable`, `rejected_key`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import httpx2

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

ERROR_BODY_LIMIT: Final = 64 * 1024
"""Bytes of a refusal's body kept for its message: enough for any error JSON, bounded against a runaway one."""


class StatusError(Exception):
    """The server answered outside 2xx, before any event. `body` is what it said, read whole up to the limit."""

    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"{status} from {url}: {body}")
        self.status = status
        self.body = body


def client(timeout: float) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(timeout=timeout)


async def open_stream(
    http: httpx2.AsyncClient, url: str, body: bytes, headers: Mapping[str, str], io_timeout: float
) -> httpx2.Response:
    """POST `body` and return the open response once its status is 2xx; else `StatusError`, the response closed.
    A transport error before the status line propagates as is: nothing of the request ran. `io_timeout` bounds each
    connect, read and write, as httpx does, not the whole stream."""
    request = http.build_request(
        "POST", url, content=body, headers={**headers, "content-type": "application/json"}, timeout=io_timeout
    )
    response = await http.send(request, stream=True)
    if response.is_success:
        return response
    try:
        kept = bytearray()
        async for chunk in response.aiter_bytes():
            kept += chunk
            if len(kept) >= ERROR_BODY_LIMIT:
                break
    finally:
        await response.aclose()
    raise StatusError(response.status_code, kept[:ERROR_BODY_LIMIT].decode(errors="replace"), url)


async def events(response: httpx2.Response) -> AsyncGenerator[str]:
    """Each server-sent event's data, its `data:` lines joined by newlines. Comments and other fields are skipped."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" "))
    if data:
        yield "\n".join(data)
