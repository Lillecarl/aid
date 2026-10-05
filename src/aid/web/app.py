"""The web UI: a Starlette app in front of the daemon, reached through the same client as any other program.

What the daemon answers reaches the page over the ZWS relays (`zws`). What aid web answers itself (the login, the
session's files, speech to text, the pane) is plain HTTP and WebSockets here.
"""

from __future__ import annotations

import contextlib
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import anyio
import anyio.to_thread
import numpy as np
from hypercorn.asyncio import (
    serve as hypercorn_serve,  # pyright: ignore[reportUnknownVariableType] -- its WSGI branch types a bare dict
)
from hypercorn.config import Config as HypercornConfig
from libpymux.streams import PaneStream, StreamRefused
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
)
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from aid.client import connect
from aid.protocol import AidError
from aid.speech import Transcription
from aid.web import auth, files, theme, zws

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
    from starlette.websockets import WebSocket

    from aid.client import Client
    from aid.paths import Paths
    from aid.speech import Recognizer
    from aid.web.highlight import Grammars

ENV_ASSETS: Final = "AID_WEB_ASSETS"
SESSION_MAX_AGE: Final = 12 * 3600
SECURITY_HEADERS: Final = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
WS_POLICY_VIOLATION: Final = 1008
WS_NO_SPEECH: Final = 4404
# RFC 6455 allows 123 bytes of close reason.
WS_REASON_MAX: Final = 120
# One second of float32 at the highest rate SpeechStart takes; a page sends about 0.1 s per frame.
MAX_AUDIO_FRAME: Final = 192000 * 4
_STATUS: Final = {
    "not_found": 404,
    "outside": 403,
    "not_a_directory": 400,
    "not_a_file": 400,
}


type Endpoint = Callable[[Request], Awaitable[Response]]


def _client(request: Request) -> Client:
    return request.app.state.client


def api(endpoint: Endpoint) -> Endpoint:
    """Require a logged-in user; map AidError. None of these changes anything, so none needs the CSRF header."""

    async def handler(request: Request) -> Response:
        if auth.current_user(request) is None:
            return auth.unauthorized()
        try:
            return await endpoint(request)
        except AidError as error:
            return JSONResponse({"error": error.message, "code": error.code}, status_code=_STATUS.get(error.code, 502))
        except ValidationError as error:
            return JSONResponse({"error": error.errors(include_url=False)}, status_code=422)

    return handler


@api
async def me(request: Request) -> Response:
    user = auth.current_user(request)
    return JSONResponse({"email": user.email if user else None, "csrf": request.session.get(auth.CSRF_KEY)})


class PathQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = ""
    """Relative to the session's working directory."""


async def _cwd(request: Request) -> str:
    return (await _client(request).session(request.path_params["name"]).status()).cwd


@api
async def list_files(request: Request) -> Response:
    query = PathQuery.model_validate(dict(request.query_params))
    entries = await files.list_dir(await _cwd(request), query.path)
    return JSONResponse([e.model_dump(mode="json") for e in entries])


@api
async def read_file(request: Request) -> Response:
    query = PathQuery.model_validate(dict(request.query_params))
    view = await files.read_file(await _cwd(request), query.path)
    grammars: Grammars | None = request.app.state.grammars
    if grammars is not None and view.text is not None:
        view.highlights = await anyio.to_thread.run_sync(grammars.highlight, view.path, view.text)
    return JSONResponse(view.model_dump(mode="json"), headers={"Cache-Control": "no-store"})


@api
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


async def pane(websocket: WebSocket) -> None:
    """Interactive Claude's pane for `<pymux-pane>`: pymux's frames out, the viewer's keys in (Lillecarl/pymux#461).

    A relay: it reads neither side. pymux spells the keys for the program's keyboard mode, and a stream opened
    writable is the only way input reaches the pane. The boundary is here, in front of it: login and Origin, as for
    any WebSocket, because whoever reaches the pymux socket can type into every pane on it.
    """
    if auth.current_user(websocket) is None or not auth.same_origin(websocket, websocket.app.state.oidc_config):
        await websocket.close(code=WS_POLICY_VIOLATION)
        return
    client: Client = websocket.app.state.client
    try:
        address = await client.session(websocket.path_params["name"]).pane()
    except AidError as error:
        await websocket.close(code=WS_POLICY_VIOLATION, reason=error.message[:WS_REASON_MAX])
        return
    await websocket.accept()
    try:
        async with (
            PaneStream(address.socket, address.pane, writable=True) as stream,
            anyio.create_task_group() as tg,
        ):

            async def to_viewer() -> None:
                async for frame in stream:
                    await websocket.send_text(json.dumps(frame))
                # The pane is gone: its stream's end is the whole signal (no frame says so).
                tg.cancel_scope.cancel()

            tg.start_soon(to_viewer)
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (text := message.get("text")) is None:
                    continue
                said = json.loads(text)
                if not isinstance(said, dict):
                    break
                await stream.send(cast("dict[str, Any]", said))
            tg.cancel_scope.cancel()
    except StreamRefused as error:
        await websocket.close(code=WS_POLICY_VIOLATION, reason=str(error)[:WS_REASON_MAX])
        return
    except json.JSONDecodeError, WebSocketDisconnect:
        pass
    with contextlib.suppress(RuntimeError):  # Already closed by the viewer.
        await websocket.close()


async def index(request: Request) -> Response:
    if auth.current_user(request) is None:
        return RedirectResponse("/login", status_code=303)
    assets: Path | None = request.app.state.assets
    if assets is None:
        return PlainTextResponse(f"the UI is not built; point {ENV_ASSETS} at web/dist", status_code=503)
    return FileResponse(assets / "index.html", headers={"Cache-Control": "no-store"})


PUBLIC_FILES: Final = {
    "manifest.webmanifest": "application/manifest+json",
    "icon.svg": "image/svg+xml",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "icon-maskable-512.png": "image/png",
}
"""The app's manifest and icons, served without a login: browsers fetch a manifest without cookies."""


async def public_file(request: Request) -> Response:
    assets: Path | None = request.app.state.assets
    name = request.url.path.removeprefix("/")
    if assets is None or not (assets / name).is_file():
        return PlainTextResponse("not built", status_code=404)
    return FileResponse(assets / name, media_type=PUBLIC_FILES[name], headers={"Cache-Control": "no-cache"})


async def health(_request: Request) -> Response:
    return PlainTextResponse("ok")


async def theme_css(request: Request) -> Response:
    return Response(request.app.state.theme_css, media_type="text/css", headers={"Cache-Control": "no-cache"})


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
    recognizer: Recognizer | None = None,
    colors: theme.Theme | None = None,
    grammars: Grammars | None = None,
) -> Starlette:
    """The app. `assets` is the built UI (web/dist); without it the API still works and `/` says what is missing.
    `recognizer` is a loaded speech model (`aid.speech.load`); without one there is no speech to text.
    `colors` is a pymux theme for `/theme.css`; without one the page keeps the browser's light or dark.
    `grammars` highlight the file viewer's files; without them the browser highlights what it can."""

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
            Route("/theme.css", theme_css),
            *(Route(f"/{name}", public_file) for name in PUBLIC_FILES),
            Route("/login", auth.login),
            Route(auth.CALLBACK_PATH, auth.callback),
            Route("/logout", auth.logout, methods=["POST"]),
            Route("/api/me", me),
            Route("/api/sessions/{name}/files", list_files, methods=["GET"]),
            Route("/api/sessions/{name}/file", read_file, methods=["GET"]),
            Route("/api/speech", speech, methods=["GET"]),
            WebSocketRoute("/api/transcribe", transcribe),
            WebSocketRoute("/api/sessions/{name}/pane", pane),
            WebSocketRoute("/api/zws/control", zws.control),
            WebSocketRoute("/api/zws/events", zws.events),
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
    app.state.recognizer = recognizer
    app.state.theme_css = theme.css(colors)
    app.state.grammars = grammars
    return app


async def serve(app: Starlette, bind: str, *, shutdown: anyio.Event | None = None) -> None:
    config = HypercornConfig()
    config.bind = [bind]
    config.accesslog = "-"
    stop_event = shutdown or anyio.Event()
    await hypercorn_serve(app, config, shutdown_trigger=stop_event.wait)  # pyright: ignore[reportArgumentType] -- Starlette is an ASGI app; hypercorn's Framework union does not name it
