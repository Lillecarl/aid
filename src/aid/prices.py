"""Model prices from models.dev's catalog, the list opencode prices from, for turns whose engine reports no cost
(native). Cached under the user's cache dir and fetched again once a day; a failed fetch keeps the stale copy, and
with no copy at all the cost is unknown, never zero.

`AID_MODEL_PRICES` replaces the source: a URL, a file path, or `off`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import anyio
import anyio.to_thread
import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

log = structlog.get_logger(__name__)

ENV_PRICES: Final = "AID_MODEL_PRICES"
CATALOG_URL: Final = "https://models.dev/api.json"
MAX_AGE: Final = 24 * 3600
FETCH_TIMEOUT: Final = 10.0
PER: Final = 1_000_000
"""models.dev prices per million tokens, in USD."""
CURRENCY: Final = "USD"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float
    tiers: tuple[tuple[int, Price], ...] = field(default=())
    """(prompt tokens, price): a request whose prompt is over the size bills at that price (Haiku 5.5: 100K)."""

    def cost(self, uncached: int, cache_read: int, cache_write: int, output: int) -> float:
        """One request's cost. Reasoning tokens are inside `output`, billed at the output price as both
        catalogued providers bill them."""
        price = self
        prompt = uncached + cache_read + cache_write
        for size, tier in sorted(self.tiers, key=lambda pair: pair[0]):
            if prompt > size:
                price = tier
        tokens = uncached * price.input + cache_read * price.cache_read + cache_write * price.cache_write
        return (tokens + output * price.output) / PER


class _Size(BaseModel):
    size: int


class _Cost(BaseModel):
    """A catalog `cost` entry, per million tokens. A missing cache price bills those tokens as plain input."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def price(self, tiers: tuple[tuple[int, Price], ...] = ()) -> Price:
        return Price(
            input=self.input,
            output=self.output,
            cache_read=self.input if self.cache_read is None else self.cache_read,
            cache_write=self.input if self.cache_write is None else self.cache_write,
            tiers=tiers,
        )


class _Tier(_Cost):
    tier: _Size


class _ModelCost(_Cost):
    tiers: list[_Tier] = Field(default_factory=list[_Tier])


class _Model(BaseModel):
    cost: _ModelCost | None = None


def lookup(catalog: Any, model: str) -> Price | None:
    """`provider/model`, as aid names native models; models.dev keys providers the same way."""
    provider, _, name = model.partition("/")
    try:
        entry = _Model.model_validate(catalog[provider]["models"][name])
    except KeyError, TypeError, ValidationError:
        return None
    if entry.cost is None:
        return None
    return entry.cost.price(tuple((tier.tier.size, tier.price()) for tier in entry.cost.tiers))


def cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "aid" / "models.dev.json"


async def _fetch(url: str, target: Path) -> None:
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    body = response.content
    json.loads(body)  # Never replace a good copy with something that does not parse.
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
    # Workers refresh it independently: write aside, then rename over.
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    await anyio.Path(temporary).write_bytes(body)
    await anyio.Path(temporary).replace(target)


async def price_for(model: str) -> Price | None:
    """The model's price, or None when the catalog lacks it or cannot be had."""
    source = os.environ.get(ENV_PRICES, CATALOG_URL)
    if source == "off":
        return None
    if source.startswith(("http://", "https://")):
        path = cache_file()
        try:
            stale = time.time() - path.stat().st_mtime > MAX_AGE
        except FileNotFoundError:
            stale = True
        if stale:
            try:
                await _fetch(source, path)
            except (httpx.HTTPError, OSError, ValueError) as error:
                log.warning("prices_fetch_failed", url=source, error=repr(error), cached=path.exists())
    else:
        path = Path(source)
    try:
        catalog: Any = await anyio.to_thread.run_sync(lambda: json.loads(path.read_bytes()))
    except (OSError, ValueError) as error:
        log.warning("prices_unreadable", path=str(path), error=repr(error))
        return None
    return lookup(catalog, model)
