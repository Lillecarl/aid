"""The web UI: a Starlette app in front of the daemon, reached through the same client as any other program.

Prompts stream back as Server-Sent Events on the POST that sends them, which a page reads with fetch. That keeps
everything on plain HTTP, where the session cookie and the CSRF header already apply.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Final

import anyio
import anyio.to_thread
import numpy as np
from hypercorn.asyncio import (
    serve as hypercorn_serve,  # pyright: ignore[reportUnknownVariableType] -- its WSGI branch types a bare dict
)
from hypercorn.config import Config as HypercornConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError
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
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

from aid.client import connect
from aid.protocol import AidError, CreateSession
from aid.speech import Transcription
from aid.speech import load as load_speech
from aid.web import auth

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
    from starlette.websockets import WebSocket

    from aid.client import Client
    from aid.paths import Paths
    from aid.speech import Recognizer

log = logging.getLogger(__name__)

ENV_ASSETS: Final = "AID_WEB_ASSETS"
SESSION_MAX_AGE: Final = 12 * 3600
SECURITY_HEADERS: Final = {
    # style-src-attr: pymux's HTML of a pane colours its cells with style attributes. pyte writes the declarations;
    # a program in the pane picks at most a colour.
    "Content-Security-Policy": (
        "default-src 'self'; style-src-attr 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
STATUS_POLL: Final = 1.0
# pymux has no "pane changed" wait yet (Lillecarl/pymux long poll, planned): a frame is fetched this often.
SCREEN_POLL: Final = 0.5
KEEPALIVE: Final = 15.0
WS_POLICY_VIOLATION: Final = 1008
WS_NO_SPEECH: Final = 4404
# One second of float32 at the highest rate SpeechStart takes; a page sends about 0.1 s per frame.
MAX_AUDIO_FRAME: Final = 192000 * 4
_STATUS: Final = {
    "not_found": 404,
    "exists": 409,
    "invalid_request": 422,
    "busy": 409,
    "no_screen": 404,
    "not_running": 409,
}


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


def _changes(fetch: Callable[[], Awaitable[BaseModel]], poll: float) -> StreamingResponse:
    """Server-Sent Events of `fetch()` each time its value changes, for as long as the page holds the stream open.
    A page opens one only while its tab is on screen, so nobody polls the daemon for what nobody sees."""

    async def events() -> AsyncIterator[str]:
        last: str | None = None
        quiet = 0.0
        while True:
            try:
                current = (await fetch()).model_dump_json()
            except AidError as error:
                yield f"event: error\ndata: {json.dumps({'error': error.message, 'code': error.code})}\n\n"
                return
            if current != last:
                last, quiet = current, 0.0
                yield f"data: {current}\n\n"
            elif quiet >= KEEPALIVE:
                # A comment: it tells a dead connection apart from a quiet one.
                quiet = 0.0
                yield ": still here\n\n"
            await anyio.sleep(poll)
            quiet += poll

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@api()
async def status_events(request: Request) -> Response:
    session = _client(request).session(request.path_params["name"])
    await session.status()  # A missing session is a 404 here, not an error inside the stream.
    return _changes(session.status, STATUS_POLL)


@api()
async def screen_events(request: Request) -> Response:
    """Interactive Claude's pane as HTML, each time it changes."""
    session = _client(request).session(request.path_params["name"])
    await session.screen()
    return _changes(session.screen, SCREEN_POLL)


@api()
async def screen_stylesheet(request: Request) -> Response:
    """The CSS a pane's HTML is written against, with the pane's own colours."""
    view = await _client(request).session(request.path_params["name"]).screen(stylesheet=True)
    return Response(view.stylesheet or "", media_type="text/css", headers={"Cache-Control": "no-store"})


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


@api()
async def speech(request: Request) -> Response:
    return JSONResponse({"enabled": request.app.state.recognizer is not None})


class SpeechStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate: Annotated[int, Field(ge=8000, le=192000)]


async def transcribe(websocket: WebSocket) -> None:
    """Speech to text, streamed: `{"rate": N}`, then binary frames of little-endian float32 mono samples at that
    rate, then `{"end": true}`. Each change comes back as `{"text": ..., "final": ...}`; `{"done": true}` last.

    A WebSocket carries no CSRF header, so the Origin must be aid's own, as well as the session cookie.
    """
    recognizer: Recognizer | None = websocket.app.state.recognizer
    if auth.current_user(websocket) is None or not auth.same_origin(websocket, websocket.app.state.oidc_config):
        await websocket.close(code=WS_POLICY_VIOLATION)
        return
    if recognizer is None:
        await websocket.close(code=WS_NO_SPEECH)
        return
    await websocket.accept()
    try:
        rate = SpeechStart.model_validate_json(await websocket.receive_text()).rate
    except ValidationError, KeyError:
        await websocket.close(code=WS_POLICY_VIOLATION)
        return
    transcription = Transcription(recognizer)
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if (data := message.get("bytes")) is not None:
            if len(data) > MAX_AUDIO_FRAME or len(data) % 4:
                await websocket.close(code=WS_POLICY_VIOLATION)
                return
            samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
            heard = await anyio.to_thread.run_sync(transcription.feed, rate, samples)
        else:
            heard = await anyio.to_thread.run_sync(transcription.finish, rate)
        for h in heard:
            await websocket.send_json({"text": h.text, "final": h.final})
        if data is None:
            await websocket.send_json({"done": True})
            await websocket.close()
            return


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
    oidc: auth.OidcConfig,
    session_secret: str,
    paths: Paths | None = None,
    assets: Path | None = None,
    speech_model: Path | None = None,
) -> Starlette:
    """The app. `assets` is the built UI (web/dist); without it the API still works and `/` says what is missing.
    `speech_model` is a sherpa-onnx streaming transducer (`aid.speech`); without one there is no speech to text."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        app.state.recognizer = await anyio.to_thread.run_sync(load_speech, speech_model) if speech_model else None
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
            Route("/api/sessions/{name}/screen/events", screen_events, methods=["GET"]),
            Route("/api/sessions/{name}/screen.css", screen_stylesheet, methods=["GET"]),
            Route("/api/sessions/{name}/prompt", prompt, methods=["POST"]),
            Route("/api/sessions/{name}/cancel", cancel, methods=["POST"]),
            Route("/api/sessions/{name}/stop", stop, methods=["POST"]),
            Route("/api/speech", speech, methods=["GET"]),
            WebSocketRoute("/api/transcribe", transcribe),
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
