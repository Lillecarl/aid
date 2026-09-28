"""OIDC login for the web UI: authorization code with PKCE, a signed session cookie, an email allowlist.

Logging in is not enough to use aid. A session runs commands on this machine, so only verified emails on the
allowlist get in; everyone else the provider knows is turned away at the callback.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

log = logging.getLogger(__name__)

USER_KEY: Final = "user"
CSRF_KEY: Final = "csrf"
CSRF_HEADER: Final = "X-CSRF-Token"
CALLBACK_PATH: Final = "/auth/callback"


@dataclass(frozen=True)
class OidcConfig:
    issuer: str
    client_id: str
    client_secret: str
    base_url: str
    """Where browsers reach aid, such as https://aid.example.com. The redirect URI is this plus /auth/callback."""
    allowed_emails: frozenset[str]

    @property
    def redirect_uri(self) -> str:
        return self.base_url.rstrip("/") + CALLBACK_PATH


@dataclass(frozen=True)
class User:
    email: str
    name: str


def make_oauth(config: OidcConfig) -> Any:
    oauth = OAuth()
    oauth.register(  # pyright: ignore[reportUnknownMemberType] -- authlib is untyped
        "oidc",
        server_metadata_url=config.issuer.rstrip("/") + "/.well-known/openid-configuration",
        client_id=config.client_id,
        client_secret=config.client_secret,
        client_kwargs={"scope": "openid email profile", "code_challenge_method": "S256"},
    )
    return cast("Any", oauth.oidc)  # pyright: ignore[reportUnknownMemberType] -- authlib is untyped


def allowed(claims: dict[str, Any], config: OidcConfig) -> str | None:
    """The email to log in as, or None. The provider must have verified it."""
    email = claims.get("email")
    if not isinstance(email, str) or claims.get("email_verified") is not True:
        return None
    return email if email.lower() in config.allowed_emails else None


def current_user(request: Request) -> User | None:
    data = request.session.get(USER_KEY)
    return User(**cast("dict[str, str]", data)) if isinstance(data, dict) else None


def csrf_ok(request: Request) -> bool:
    expected = request.session.get(CSRF_KEY)
    given = request.headers.get(CSRF_HEADER)
    return isinstance(expected, str) and given is not None and secrets.compare_digest(expected, given)


def unauthorized() -> Response:
    return JSONResponse({"error": "login required"}, status_code=401)


async def login(request: Request) -> Response:
    config: OidcConfig = request.app.state.oidc_config
    return await request.app.state.oidc.authorize_redirect(request, config.redirect_uri)


async def callback(request: Request) -> Response:
    config: OidcConfig = request.app.state.oidc_config
    try:
        token = await request.app.state.oidc.authorize_access_token(request)
    except OAuthError as error:
        log.warning("OIDC callback failed: %s", error)
        return PlainTextResponse("login failed", status_code=400)
    claims = cast("dict[str, Any]", token.get("userinfo") or {})
    email = allowed(claims, config)
    if email is None:
        log.warning("refused login for %r", claims.get("email"))
        return PlainTextResponse("this account may not use aid", status_code=403)
    request.session.clear()
    request.session[USER_KEY] = {"email": email, "name": str(claims.get("name") or email)}
    request.session[CSRF_KEY] = secrets.token_urlsafe(32)
    log.info("login: %s", email)
    return RedirectResponse("/", status_code=303)


async def logout(request: Request) -> Response:
    if not csrf_ok(request):
        return PlainTextResponse("bad CSRF token", status_code=403)
    request.session.clear()
    return JSONResponse({})
