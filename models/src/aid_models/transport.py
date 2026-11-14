"""What every model client shares: `provider/model` names, providers and their keys, and requests that sort an
outage from a refusal. Every client is aid's own (`aid_models.wire`, #65): no model SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import anyio
import httpx2
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, ValidationError

from aid_models import wire
from aid_models.errors import AidError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class StreamBroken(Exception):
    """The connection failed after the response began. No tool call of it ran, so the request may go again."""


class ModelUnavailable(StreamBroken):
    """The server refused the request or was out of reach before it answered: nothing of it ran."""


TRANSIENT_STATUS: Final = frozenset({408, 409, 429})
"""Statuses below 500 a request may go again after: the ones OpenAI's and Anthropic's SDKs retry."""


def unavailable(error: BaseException) -> bool:
    """A failure to send again later: no connection, a timeout, a 5xx, or a rate limit. Measured 2026-11: a vLLM
    gateway answered 503 "no available server" for hours, past two quick retries."""
    match error:
        case wire.StatusError():
            return error.status >= 500 or error.status in TRANSIENT_STATUS
        case _:
            return False


def rejected_key(error: BaseException) -> bool:
    """The server refused the key: a person must replace it, so sending again cannot help. Measured 2026-11-13 with a
    bogus key: OpenAI, Anthropic, DeepSeek, OpenCode and OpenRouter answer 401; an expired OpenCode key too. Gemini
    answers 400 with a reason (`gemini.refused_key`)."""
    match error:
        case wire.StatusError():
            return error.status == 401
        case _:
            return False


def refused_key(name: str, error: BaseException) -> AidError:
    return AidError("bad_credentials", f"{name}: the server refused the key: {error}")


@runtime_checkable
class Hosted(Protocol):
    """A model that says where its requests go: what a private session (`NativeSpec.private`) needs to run on it."""

    def host(self, settings: Mapping[str, JsonValue]) -> str:
        """The host name of the base URL this model's client uses with `settings`."""
        ...


def check_host(model: object, settings: Mapping[str, JsonValue], hosts: Sequence[str]) -> None:
    """Refuse a model whose requests go to a host not in `hosts`, or that cannot say where they go."""
    if not isinstance(model, Hosted):
        raise AidError("private_host", f"{model_label(model)} cannot say which host it sends to")
    if (host := model.host(settings)) not in hosts:
        raise AidError("private_host", f"{model_label(model)} sends to {host!r}, which is not one of {list(hosts)}")


def model_label(model: object) -> str:
    return str(getattr(model, "model_name", type(model).__name__))


@dataclass(frozen=True)
class Provider:
    base_url: str | None
    """None: `chat_api.DEFAULT_BASE` (api.openai.com)."""
    key_env: str
    local: bool = False
    """An unauthenticated local server: a missing key is not an error."""
    defaults: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Request settings the session's `model_settings` override."""
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    """Sent with every request of the client."""
    stream_usage: bool = True
    """The server takes `stream_options.include_usage` (Gemini's refuses it)."""


OPENAI_COMPATIBLE: Final[dict[str, Provider]] = {
    "openai": Provider(None, "OPENAI_API_KEY"),
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", stream_usage=False
    ),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": Provider("https://api.x.ai/v1", "XAI_API_KEY"),
    "together": Provider("https://api.together.xyz/v1", "TOGETHERAI_API_KEY"),
    "fireworks": Provider("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "ollama": Provider("http://localhost:11434/v1", "OLLAMA_API_KEY", local=True),
    "vllm": Provider("http://localhost:8000/v1", "VLLM_API_KEY", local=True),
    "lmstudio": Provider("http://localhost:1234/v1", "LM_STUDIO_API_KEY", local=True),
}
"""`provider/model` prefixes served over Chat Completions, for servers models.dev names no URL for and local ones.
Others resolve through `aid.providers`."""


class ClientSettings(BaseModel):
    """The `model_settings` keys that configure the client; every other key goes into each request."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    api_key: str | None = None
    base_url: str | None = Field(None, validation_alias=AliasChoices("api_base", "base_url"))
    max_retries: int = Field(0, validation_alias=AliasChoices("max_retries", "num_retries"))


def _setting_names(model: type[BaseModel]) -> frozenset[str]:
    names: set[str] = set()
    for name, info in model.model_fields.items():
        alias = info.validation_alias
        if isinstance(alias, AliasChoices):
            names.update(c for c in alias.choices if isinstance(c, str))
        else:
            names.add(name)
    return frozenset(names)


CLIENT_SETTINGS: Final = _setting_names(ClientSettings)


def split_name(name: str) -> tuple[str, str]:
    """`provider/model` → (provider, model); a bare name is OpenAI's."""
    provider, _, model = name.partition("/")
    return (provider, model) if model else ("openai", provider)


@dataclass(frozen=True)
class ClientKey:
    base_url: str | None
    api_key: str
    max_retries: int


def client_settings(name: str, settings: Mapping[str, JsonValue]) -> ClientSettings:
    try:
        return ClientSettings.model_validate(settings)
    except ValidationError as e:
        raise AidError("bad_model_settings", f"{name}: {e}") from e


def client_key(
    name: str, settings: Mapping[str, JsonValue], key_env: str | None, base_url: str | None, *, local: bool = False
) -> ClientKey:
    """The client's settings over the provider's: `key_env` and `base_url` apply where the settings name none."""
    client = client_settings(name, settings)
    api_key = client.api_key or (os.environ.get(key_env) if key_env else None)
    if not api_key:
        if not local:
            raise AidError("missing_credentials", f"{name}: set {key_env or 'api_key in model_settings'}")
        api_key = "local"
    return ClientKey(client.base_url or base_url, api_key, client.max_retries)


RETRY_BACKOFF: Final = 0.5
"""Seconds before a client retry (`max_retries`), doubling. The backend replays past them (`STREAM_RETRIES`)."""


async def model_answer(
    name: str,
    http: httpx2.AsyncClient,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    io_timeout: float,
    retries: int,
    failed: str,
    refused: Callable[[wire.StatusError], bool] = lambda error: False,
) -> bytes:
    """A whole answer that is not a stream (embeddings, reranking): an outage → `model_unavailable`, any other
    refusal → `failed`, as `open_model_stream` sorts them."""
    try:
        response = await open_model_stream(name, http, url, body, headers, io_timeout, retries, refused)
        try:
            return await response.aread()
        finally:
            await response.aclose()
    except ModelUnavailable as error:
        raise AidError("model_unavailable", f"{name}: {error}") from error
    except wire.StatusError as error:
        raise AidError(failed, f"{name}: {error}") from error


async def open_model_stream(
    name: str,
    http: httpx2.AsyncClient,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    io_timeout: float,
    retries: int,
    refused: Callable[[wire.StatusError], bool] = lambda error: False,
) -> httpx2.Response:
    """Open a model's stream: a refused key → `bad_credentials` (`refused` names a provider's own way to say it), a
    failure to send again later → retried `retries` times, then `ModelUnavailable`; anything else propagates."""
    for attempt in range(retries + 1):
        try:
            return await wire.open_stream(http, url, body, headers, io_timeout)
        except (wire.StatusError, httpx2.TransportError) as error:
            if rejected_key(error) or (isinstance(error, wire.StatusError) and refused(error)):
                raise refused_key(name, error) from error
            if not (isinstance(error, httpx2.TransportError) or unavailable(error)):
                raise
            if attempt == retries:
                raise ModelUnavailable(f"{type(error).__name__}: {error}") from error
            await anyio.sleep(RETRY_BACKOFF * 2**attempt)
    raise AssertionError("unreachable")


GOOGLE_KEY_ENV: Final = "GEMINI_API_KEY"
