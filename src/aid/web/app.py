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
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

from aid.client import connect
from aid.protocol import AidError, CreateSession
from aid.web import auth, page

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable

    from starlette.requests import Request

    from aid.client import Client
    from aid.paths import Paths

log = logging.getLogger(__name__)

SESSION_MAX_AGE: Final = 12 * 3600
SECURITY_HEADERS: Final = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
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


@api(mutating=True)
async def create_session(request: Request) -> Response:
    create = CreateSession.model_validate(await request.json())
    await _client(request).call(create)
    return JSONResponse({"name": create.name}, status_code=201)


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
    return Response(page.HTML, media_type="text/html; charset=utf-8", headers=SECURITY_HEADERS)


async def script(_request: Request) -> Response:
    return Response(page.SCRIPT, media_type="text/javascript; charset=utf-8", headers=SECURITY_HEADERS)


async def style(_request: Request) -> Response:
    return Response(page.STYLE, media_type="text/css; charset=utf-8", headers=SECURITY_HEADERS)


async def health(_request: Request) -> Response:
    return PlainTextResponse("ok")


def create_app(oidc: auth.OidcConfig, session_secret: str, paths: Paths | None = None) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        async with connect(paths) as client:
            app.state.client = client
            yield

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/app.js", script),
            Route("/app.css", style),
            Route("/healthz", health),
            Route("/login", auth.login),
            Route(auth.CALLBACK_PATH, auth.callback),
            Route("/logout", auth.logout, methods=["POST"]),
            Route("/api/me", me),
            Route("/api/sessions", list_sessions, methods=["GET"]),
            Route("/api/sessions", create_session, methods=["POST"]),
            Route("/api/sessions/{name}", delete, methods=["DELETE"]),
            Route("/api/sessions/{name}/prompt", prompt, methods=["POST"]),
            Route("/api/sessions/{name}/cancel", cancel, methods=["POST"]),
            Route("/api/sessions/{name}/stop", stop, methods=["POST"]),
        ],
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=session_secret,
                session_cookie="aid_session",
                max_age=SESSION_MAX_AGE,
                same_site="lax",
                https_only=oidc.base_url.startswith("https://"),
            )
        ],
        lifespan=lifespan,
    )
    app.state.oidc_config = oidc
    app.state.oidc = auth.make_oauth(oidc)
    return app


async def serve(app: Starlette, bind: str, *, shutdown: anyio.Event | None = None) -> None:
    config = HypercornConfig()
    config.bind = [bind]
    config.accesslog = "-"
    stop_event = shutdown or anyio.Event()
    await hypercorn_serve(app, config, shutdown_trigger=stop_event.wait)  # pyright: ignore[reportArgumentType] -- Starlette is an ASGI app; hypercorn's Framework union does not name it
