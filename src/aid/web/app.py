"""The web UI: a Starlette app in front of the daemon, reached through the same client as any other program.

Prompts stream back as Server-Sent Events on the POST that sends them, which a page reads with fetch. That keeps
everything on plain HTTP, where the session cookie and the CSRF header already apply.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

import anyio
from hypercorn.asyncio import (
    serve as hypercorn_serve,  # pyright: ignore[reportUnknownVariableType] -- its WSGI branch types a bare dict
)
from hypercorn.config import Config as HypercornConfig
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from aid.client import connect
from aid.protocol import AidError, CreateSession
from aid.web import auth

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from aid.client import Client
    from aid.paths import Paths

log = logging.getLogger(__name__)

ENV_ASSETS: Final = "AID_WEB_ASSETS"
SESSION_MAX_AGE: Final = 12 * 3600
SECURITY_HEADERS: Final = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
STATUS_POLL: Final = 1.0
STATUS_KEEPALIVE: Final = 15.0
_STATUS: Final = {"not_found": 404, "exists": 409, "invalid_request": 422, "busy": 409}


class PromptBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


type Endpoint = Callable[[Request], Awaitable[Response]]


def _client(request: Request) -> Client:
    return request.app.state.client


def api(*, mutating: bool = False) -> Callable[[Endpoint], Endpoint]:
    """Require a logged-in user; for requests that change something, the CSRF header too. Map AidError."""

    def wrap(endpoint: Endpoint) -> Endpoint:
        async def handler(request: Request) -> Response:
            if auth.current_user(request) is None:
                return auth.unauthorized()
            if mutating and not auth.csrf_ok(request):
                return JSONResponse({"error": "bad CSRF token"}, status_code=403)
            try:
                return await endpoint(request)
            except AidError as error:
                return JSONResponse(
                    {"error": error.message, "code": error.code}, status_code=_STATUS.get(error.code, 502)
                )
            except ValidationError as error:
                return JSONResponse({"error": error.errors(include_url=False)}, status_code=422)
            except json.JSONDecodeError:
                return JSONResponse({"error": "the body is not JSON"}, status_code=400)

        return handler

    return wrap


@api()
async def me(request: Request) -> Response:
    user = auth.current_user(request)
    return JSONResponse({"email": user.email if user else None, "csrf": request.session.get(auth.CSRF_KEY)})


@api()
async def list_sessions(request: Request) -> Response:
    return JSONResponse([info.model_dump(mode="json") for info in await _client(request).sessions()])


@api()
async def list_agents(request: Request) -> Response:
    return JSONResponse((await _client(request).agents()).model_dump(mode="json"))


@api(mutating=True)
async def create_session(request: Request) -> Response:
    create = CreateSession.model_validate(await request.json())
    await _client(request).call(create)
    return JSONResponse({"name": create.name}, status_code=201)


class HistoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: int | None = None
    after: int | None = None
    limit: int = 100


@api()
async def history(request: Request) -> Response:
    query = HistoryQuery.model_validate(dict(request.query_params))
    session = _client(request).session(request.path_params["name"])
    page = await session.history(before=query.before, after=query.after, limit=query.limit)
    return JSONResponse(page.model_dump(mode="json"))


@api(mutating=True)
async def prompt(request: Request) -> Response:
    body = PromptBody.model_validate(await request.json())
    session = _client(request).session(request.path_params["name"])

    async def events() -> AsyncIterator[str]:
        try:
            async for event in session.stream(body.text):
                yield f"data: {event.model_dump_json()}\n\n"
        except AidError as error:
            yield f"event: error\ndata: {json.dumps({'error': error.message, 'code': error.code})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@api()
async def status(request: Request) -> Response:
    found = await _client(request).session(request.path_params["name"]).status()
    return JSONResponse(found.model_dump(mode="json"))


@api()
async def status_events(request: Request) -> Response:
    """The session's status each time it changes, for as long as the page holds the stream open. A page opens it
    only while the status is on screen, so nobody polls the daemon for a status nobody sees."""
    session = _client(request).session(request.path_params["name"])
    await session.status()  # A missing session is a 404 here, not an error inside the stream.

    async def events() -> AsyncIterator[str]:
        last: str | None = None
        quiet = 0.0
        while True:
            try:
                current = (await session.status()).model_dump_json()
            except AidError as error:
                yield f"event: error\ndata: {json.dumps({'error': error.message, 'code': error.code})}\n\n"
                return
            if current != last:
                last, quiet = current, 0.0
                yield f"data: {current}\n\n"
            elif quiet >= STATUS_KEEPALIVE:
                # A comment: it tells a dead connection apart from a quiet one.
                quiet = 0.0
                yield ": still here\n\n"
            await anyio.sleep(STATUS_POLL)
            quiet += STATUS_POLL

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@api(mutating=True)
async def cancel(request: Request) -> Response:
    await _client(request).session(request.path_params["name"]).cancel()
    return JSONResponse({})


@api(mutating=True)
async def stop(request: Request) -> Response:
    await _client(request).session(request.path_params["name"]).stop()
    return JSONResponse({})


@api(mutating=True)
async def delete(request: Request) -> Response:
    await _client(request).session(request.path_params["name"]).delete()
    return JSONResponse({})


async def index(request: Request) -> Response:
    if auth.current_user(request) is None:
        return RedirectResponse("/login", status_code=303)
    assets: Path | None = request.app.state.assets
    if assets is None:
        return PlainTextResponse(f"the UI is not built; point {ENV_ASSETS} at web/dist", status_code=503)
    return FileResponse(assets / "index.html", headers={"Cache-Control": "no-store"})


async def health(_request: Request) -> Response:
    return PlainTextResponse("ok")


def security_headers(app: ASGIApp) -> ASGIApp:
    """Add SECURITY_HEADERS to every HTTP response. Pure ASGI, because BaseHTTPMiddleware breaks streaming."""
    extra = [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]

    async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await app(scope, receive, send_with_headers)

    return wrapped


def create_app(
    oidc: auth.OidcConfig, session_secret: str, paths: Paths | None = None, assets: Path | None = None
) -> Starlette:
    """The app. `assets` is the built UI (web/dist); without it the API still works and `/` says what is missing."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        async with connect(paths) as client:
            app.state.client = client
            yield

    static = [Mount("/assets", StaticFiles(directory=assets / "assets"))] if assets is not None else []
    app = Starlette(
        routes=[
            Route("/", index),
            *static,
            Route("/healthz", health),
            Route("/login", auth.login),
            Route(auth.CALLBACK_PATH, auth.callback),
            Route("/logout", auth.logout, methods=["POST"]),
            Route("/api/me", me),
            Route("/api/agents", list_agents, methods=["GET"]),
            Route("/api/sessions", list_sessions, methods=["GET"]),
            Route("/api/sessions", create_session, methods=["POST"]),
            Route("/api/sessions/{name}", delete, methods=["DELETE"]),
            Route("/api/sessions/{name}/history", history, methods=["GET"]),
            Route("/api/sessions/{name}/status", status, methods=["GET"]),
            Route("/api/sessions/{name}/status/events", status_events, methods=["GET"]),
            Route("/api/sessions/{name}/prompt", prompt, methods=["POST"]),
            Route("/api/sessions/{name}/cancel", cancel, methods=["POST"]),
            Route("/api/sessions/{name}/stop", stop, methods=["POST"]),
        ],
        middleware=[
            Middleware(security_headers),
            Middleware(
                SessionMiddleware,
                secret_key=session_secret,
                session_cookie="aid_session",
                max_age=SESSION_MAX_AGE,
                same_site="lax",
                https_only=oidc.base_url.startswith("https://"),
            ),
        ],
        lifespan=lifespan,
    )
    app.state.oidc_config = oidc
    app.state.oidc = auth.make_oauth(oidc)
    app.state.assets = assets
    return app


async def serve(app: Starlette, bind: str, *, shutdown: anyio.Event | None = None) -> None:
    config = HypercornConfig()
    config.bind = [bind]
    config.accesslog = "-"
    stop_event = shutdown or anyio.Event()
    await hypercorn_serve(app, config, shutdown_trigger=stop_event.wait)  # pyright: ignore[reportArgumentType] -- Starlette is an ASGI app; hypercorn's Framework union does not name it
