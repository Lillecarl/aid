"""The web UI: a Starlette app in front of the daemon, reached through the same client as any other program.

Prompts stream back as Server-Sent Events on the POST that sends them, which a page reads with fetch. That keeps
everything on plain HTTP, where the session cookie and the CSRF header already apply.
"""

from __future__ import annotations

import contextlib
import json
import logging
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
    StreamingResponse,
)
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from aid.client import connect
from aid.protocol import AidError, CreateSession, SessionInfosAdapter
from aid.speech import Transcription
from aid.web import auth, files, theme

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
    from starlette.websockets import WebSocket

    from aid.client import Client
    from aid.paths import Paths
    from aid.speech import Recognizer
    from aid.web.highlight import Grammars

log = logging.getLogger(__name__)

ENV_ASSETS: Final = "AID_WEB_ASSETS"
SESSION_MAX_AGE: Final = 12 * 3600
SECURITY_HEADERS: Final = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
KEEPALIVE: Final = 15.0
WS_POLICY_VIOLATION: Final = 1008
WS_NO_SPEECH: Final = 4404
# RFC 6455 allows 123 bytes of close reason.
WS_REASON_MAX: Final = 120
# One second of float32 at the highest rate SpeechStart takes; a page sends about 0.1 s per frame.
MAX_AUDIO_FRAME: Final = 192000 * 4
_STATUS: Final = {
    "not_found": 404,
    "exists": 409,
    "invalid_request": 422,
    "busy": 409,
    "no_screen": 404,
    "not_running": 409,
    "outside": 403,
    "not_a_directory": 400,
    "not_a_file": 400,
    "no_pending": 409,
    "no_option": 422,
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


class HistoryEventsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    after: int = Field(-1, ge=-1)


@api()
async def history_events(request: Request) -> Response:
    """Server-Sent Events: each history entry after `after` as the daemon records it, whoever started its turn.
    Each event's id is the entry's seq; a reconnecting EventSource sends it back as Last-Event-ID."""
    query = HistoryEventsQuery.model_validate(dict(request.query_params))
    session = _client(request).session(request.path_params["name"])
    await session.status()  # A missing session is a 404 here, not an error inside the stream.
    resumed = request.headers.get("last-event-id", "")
    after = max(query.after, int(resumed)) if resumed.isdecimal() else query.after
    entries = session.follow_history(after, idle=KEEPALIVE)
    return _follow(entries, lambda e: e.model_dump_json(), event_id=lambda e: e.seq, still_there=session.status)


@api()
async def summary(request: Request) -> Response:
    totals = await _client(request).session(request.path_params["name"]).summary()
    return JSONResponse(totals.model_dump(mode="json"))


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


class PathQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = ""
    """Relative to the session's working directory."""


async def _cwd(request: Request) -> str:
    return (await _client(request).session(request.path_params["name"]).status()).cwd


@api()
async def list_files(request: Request) -> Response:
    query = PathQuery.model_validate(dict(request.query_params))
    entries = await files.list_dir(await _cwd(request), query.path)
    return JSONResponse([e.model_dump(mode="json") for e in entries])


@api()
async def read_file(request: Request) -> Response:
    query = PathQuery.model_validate(dict(request.query_params))
    view = await files.read_file(await _cwd(request), query.path)
    grammars: Grammars | None = request.app.state.grammars
    if grammars is not None and view.text is not None:
        view.highlights = await anyio.to_thread.run_sync(grammars.highlight, view.path, view.text)
    return JSONResponse(view.model_dump(mode="json"), headers={"Cache-Control": "no-store"})


@api()
async def status(request: Request) -> Response:
    found = await _client(request).session(request.path_params["name"]).status()
    return JSONResponse(found.model_dump(mode="json"))


def _follow[T](
    values: AsyncGenerator[T | None],
    dump: Callable[[T], str],
    *,
    event_id: Callable[[T], int] | None = None,
    still_there: Callable[[], Awaitable[object]] | None = None,
) -> StreamingResponse:
    """Server-Sent Events of what the daemon publishes (a `follow_*` with `idle=KEEPALIVE`), for as long as the page
    holds the stream open. A page opens one only while its tab is on screen, so nothing streams that nobody sees.

    `still_there` runs on each quiet spell: a deleted session's topic just goes quiet, and its AidError ends the
    stream as the page expects."""

    async def events() -> AsyncIterator[str]:
        try:
            async for value in values:
                if value is None:
                    if still_there is not None:
                        await still_there()
                    # A comment: it tells a dead connection apart from a quiet one.
                    yield ": still here\n\n"
                    continue
                prefix = f"id: {event_id(value)}\n" if event_id is not None else ""
                yield f"{prefix}data: {dump(value)}\n\n"
        except AidError as error:
            yield f"event: error\ndata: {json.dumps({'error': error.message, 'code': error.code})}\n\n"
        finally:
            await values.aclose()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@api()
async def sessions_events(request: Request) -> Response:
    listed = _client(request).follow_sessions(idle=KEEPALIVE)
    return _follow(listed, lambda infos: SessionInfosAdapter.dump_json(infos).decode())


@api()
async def status_events(request: Request) -> Response:
    session = _client(request).session(request.path_params["name"])
    await session.status()  # A missing session is a 404 here, not an error inside the stream.
    return _follow(session.follow_status(idle=KEEPALIVE), lambda s: s.model_dump_json(), still_there=session.status)


@api(mutating=True)
async def cancel(request: Request) -> Response:
    await _client(request).session(request.path_params["name"]).cancel()
    return JSONResponse({})


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str | None


@api(mutating=True)
async def answer_permission(request: Request) -> Response:
    answer = _Answer.model_validate(await request.json())
    session = _client(request).session(request.path_params["name"])
    await session.answer(request.path_params["request_id"], answer.option_id)
    return JSONResponse({})


@api(mutating=True)
async def start(request: Request) -> Response:
    await _client(request).session(request.path_params["name"]).start()
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
            Route("/api/agents", list_agents, methods=["GET"]),
            Route("/api/sessions", list_sessions, methods=["GET"]),
            Route("/api/sessions", create_session, methods=["POST"]),
            Route("/api/sessions/events", sessions_events, methods=["GET"]),
            Route("/api/sessions/{name}", delete, methods=["DELETE"]),
            Route("/api/sessions/{name}/history", history, methods=["GET"]),
            Route("/api/sessions/{name}/summary", summary, methods=["GET"]),
            Route("/api/sessions/{name}/history/events", history_events, methods=["GET"]),
            Route("/api/sessions/{name}/status", status, methods=["GET"]),
            Route("/api/sessions/{name}/files", list_files, methods=["GET"]),
            Route("/api/sessions/{name}/file", read_file, methods=["GET"]),
            Route("/api/sessions/{name}/status/events", status_events, methods=["GET"]),
            Route("/api/sessions/{name}/prompt", prompt, methods=["POST"]),
            Route("/api/sessions/{name}/cancel", cancel, methods=["POST"]),
            Route("/api/sessions/{name}/permissions/{request_id}", answer_permission, methods=["POST"]),
            Route("/api/sessions/{name}/start", start, methods=["POST"]),
            Route("/api/sessions/{name}/stop", stop, methods=["POST"]),
            Route("/api/speech", speech, methods=["GET"]),
            WebSocketRoute("/api/transcribe", transcribe),
            WebSocketRoute("/api/sessions/{name}/pane", pane),
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
