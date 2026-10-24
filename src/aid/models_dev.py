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
from typing import Any, Final, cast

import anyio
import anyio.to_thread
import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

from aid.protocol import ModelCost, ModelEntry

log = structlog.get_logger(__name__)

ENV_SOURCE: Final = "AID_MODEL_PRICES"
CATALOG_URL: Final = "https://models.dev/api.json"
MAX_AGE: Final = 24 * 3600
FETCH_TIMEOUT: Final = 10.0


class _Limit(BaseModel):
    context: int | None = None
    output: int | None = None


class _Cost(BaseModel):
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


class CatalogModel(BaseModel):
    """One model's entry, as far as aid reads it."""

    id: str
    name: str | None = None
    reasoning: bool | None = None
    tool_call: bool | None = None
    limit: _Limit | None = None
    cost: _Cost | None = None
    status: str | None = None
    """`deprecated` for a model the provider is retiring."""

    def entry(self) -> ModelEntry:
        return ModelEntry(
            id=self.id,
            name=self.name,
            context=self.limit.context if self.limit else None,
            output=self.limit.output if self.limit else None,
            reasoning=self.reasoning,
            tools=self.tool_call,
            cost=ModelCost.model_validate(self.cost.model_dump()) if self.cost else None,
        )


class CatalogProvider(BaseModel):
    """One provider's entry, as far as aid reads it."""

    api: str | None = None
    """The base URL, where the package needs one."""
    npm: str | None = None
    """The AI SDK package that speaks to it: which protocol it serves."""
    env: list[str] = Field(default_factory=list[str])
    """Variables that hold its key, the usual one first."""
    models: dict[str, Any] = Field(default_factory=dict[str, Any])
    """Raw entries by id; `entries` parses those that parse."""

    def entries(self) -> dict[str, ModelEntry]:
        """Its models, deprecated ones left out."""
        found: dict[str, ModelEntry] = {}
        for raw in self.models.values():
            try:
                model = CatalogModel.model_validate(raw)
            except ValidationError:
                continue
            if model.status != "deprecated":
                found[model.id] = model.entry()
        return found


def provider_names(catalog: Any) -> list[str]:
    """The catalog's provider ids, sorted; none when there is no catalog."""
    return sorted(str(name) for name in cast(dict[object, object], catalog)) if isinstance(catalog, dict) else []


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
