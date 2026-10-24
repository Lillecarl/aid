"""models.dev's catalog (the list opencode builds its providers from): providers by id, each with `api`, `npm`,
`env` and `models`. Cached under the user's cache dir and fetched again once a day; a failed fetch keeps the stale
copy, and with no copy at all the catalog is None, never empty.

`AID_MODEL_PRICES` replaces the source: a URL, a file path, or `off`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Final

import anyio
import anyio.to_thread
import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

log = structlog.get_logger(__name__)

ENV_SOURCE: Final = "AID_MODEL_PRICES"
CATALOG_URL: Final = "https://models.dev/api.json"
MAX_AGE: Final = 24 * 3600
FETCH_TIMEOUT: Final = 10.0


class CatalogProvider(BaseModel):
    """One provider's entry, as far as aid reads it."""

    api: str | None = None
    """The base URL, where the package needs one."""
    npm: str | None = None
    """The AI SDK package that speaks to it: which protocol it serves."""
    env: list[str] = Field(default_factory=list[str])
    """Variables that hold its key, the usual one first."""


def provider(catalog: Any, name: str) -> CatalogProvider | None:
    """`name`'s entry in `catalog`, or None when it has none or the entry does not parse."""
    try:
        return CatalogProvider.model_validate(catalog[name])
    except KeyError, TypeError, ValidationError:
        return None


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "aid"


def cache_file() -> Path:
    return cache_dir() / "models.dev.json"


def stale(path: Path, max_age: float = MAX_AGE) -> bool:
    try:
        return time.time() - path.stat().st_mtime > max_age
    except FileNotFoundError:
        return True


async def write_aside(target: Path, body: bytes) -> None:
    """Replace `target` whole: workers refresh it independently, so write aside, then rename over."""
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    await anyio.Path(temporary).write_bytes(body)
    await anyio.Path(temporary).replace(target)


async def _fetch(url: str, target: Path) -> None:
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    body = response.content
    json.loads(body)  # Never replace a good copy with something that does not parse.
    await write_aside(target, body)


async def catalog() -> Any | None:
    """The catalog, or None when it is off or cannot be had."""
    source = os.environ.get(ENV_SOURCE, CATALOG_URL)
    if source == "off":
        return None
    if source.startswith(("http://", "https://")):
        path = cache_file()
        if stale(path):
            try:
                await _fetch(source, path)
            except (httpx.HTTPError, OSError, ValueError) as error:
                log.warning("models_dev_fetch_failed", url=source, error=repr(error), cached=path.exists())
    else:
        path = Path(source)
    try:
        return await anyio.to_thread.run_sync(lambda: json.loads(path.read_bytes()))
    except (OSError, ValueError) as error:
        log.warning("models_dev_unreadable", path=str(path), error=repr(error))
        return None
