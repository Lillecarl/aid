"""Model prices from models.dev's catalog (`aid.models_dev`), for turns whose engine reports no cost (native). With
no catalog at all the cost is unknown, never zero."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, Field, ValidationError

from aid import models_dev
from aid.models_dev import ENV_SOURCE as ENV_PRICES

__all__ = ["CURRENCY", "ENV_PRICES", "PER", "Price", "lookup", "price_for"]

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


def lookup(catalog: models_dev.ModelsDev | None, model: str) -> Price | None:
    """`provider/model`, as aid names native models; models.dev keys providers the same way."""
    try:
        entry = _Model.model_validate(models_dev.raw_model(catalog, model))
    except ValidationError:
        return None
    if entry.cost is None:
        return None
    return entry.cost.price(tuple((tier.tier.size, tier.price()) for tier in entry.cost.tiers))


async def price_for(model: str) -> Price | None:
    """The model's price, or None when the catalog lacks it or cannot be had."""
    catalog = await models_dev.catalog()
    return None if catalog is None else lookup(catalog, model)
