from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from aid import models_dev
from aid.backends.native import add_cost
from aid.model import AnthropicModel, OpenAIChatModel, RequestUsage
from aid.prices import ENV_PRICES, Price, lookup, price_for
from aid.protocol import Cost, HistoryEntry, Usage
from aid.summary import summarize

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

HAIKU = {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125}
CATALOG: dict[str, Any] = {
    "anthropic": {
        "models": {
            "claude-haiku-5-5": {
                "cost": {
                    **HAIKU,
                    "tiers": [
                        {
                            "input": 0.5,
                            "output": 2.5,
                            "cache_read": 0.05,
                            "cache_write": 0.625,
                            "tier": {"type": "context", "size": 100000},
                        }
                    ],
                }
            }
        }
    },
    "deepseek": {"models": {"deepseek-flash": {"cost": {"input": 0.15, "output": 0.6, "cache_read": 0.003}}}},
}


def test_lookup_reads_a_provider_model_name() -> None:
    flash = lookup(CATALOG, "deepseek/deepseek-flash")
    assert flash == Price(input=0.15, output=0.6, cache_read=0.003, cache_write=0.15)
    assert lookup(CATALOG, "deepseek/unknown") is None
    assert lookup(CATALOG, "nowhere/model") is None


def test_a_prompt_over_a_tier_bills_at_its_price() -> None:
    haiku = lookup(CATALOG, "anthropic/claude-haiku-5-5")
    assert haiku is not None
    assert haiku.cost(1000, 0, 0, 1000) == pytest.approx((1000 * 0.1 + 1000 * 0.5) / 1e6)
    assert haiku.cost(1000, 100_000, 0, 1000) == pytest.approx((1000 * 0.5 + 100_000 * 0.05 + 1000 * 2.5) / 1e6)


def test_a_prompt_of_exactly_the_tier_size_bills_at_the_base_price() -> None:
    haiku = lookup(CATALOG, "anthropic/claude-haiku-5-5")
    assert haiku is not None
    at_size = haiku.cost(100_000, 0, 0, 1000)
    over_size = haiku.cost(100_001, 0, 0, 1000)
    assert at_size == pytest.approx((100_000 * 0.1 + 1000 * 0.5) / 1e6)
    assert over_size == pytest.approx((100_001 * 0.5 + 1000 * 2.5) / 1e6)


def test_chat_completions_counts_cache_reads_inside_input() -> None:
    price = Price(input=1, output=2, cache_read=0.1, cache_write=1.25)
    usage = RequestUsage(input_tokens=1000, output_tokens=10, cache_read_tokens=800)
    openai = add_cost(None, price, usage, OpenAIChatModel("deepseek/deepseek-flash"))
    anthropic = add_cost(Cost(amount=1.0, currency="USD"), price, usage, AnthropicModel("anthropic/claude-haiku-5-5"))
    assert openai.amount == pytest.approx((200 * 1 + 800 * 0.1 + 10 * 2) / 1e6)
    assert anthropic.amount == pytest.approx(1.0 + (1000 * 1 + 800 * 0.1 + 10 * 2) / 1e6)


def test_turn_costs_add_up_beside_a_session_total() -> None:
    usages = [
        Usage(cost=Cost(amount=0.25, currency="USD")),
        Usage(session_cost=Cost(amount=2.0, currency="USD")),
        Usage(cost=Cost(amount=0.5, currency="USD")),
        Usage(session_cost=Cost(amount=3.0, currency="USD")),
    ]
    entries = [HistoryEntry(seq=seq, at=0, turn=str(seq), item=usage) for seq, usage in enumerate(usages)]
    # A fork's turn priced by aid, in a parent whose agent reports its own running total.
    assert summarize(entries).cost == {"USD": 3.75}


async def test_a_file_source_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(CATALOG))
    monkeypatch.setenv(ENV_PRICES, str(catalog))
    assert await price_for("deepseek/deepseek-flash") is not None
    monkeypatch.setenv(ENV_PRICES, "off")
    assert await price_for("deepseek/deepseek-flash") is None


async def test_a_daily_fetch_that_fails_keeps_the_stale_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PRICES, "https://catalog.invalid/api.json")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    served: list[httpx.Response] = [httpx.Response(200, json=CATALOG), httpx.Response(503)]
    real = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(lambda request: served.pop(0)), **kwargs)

    monkeypatch.setattr(models_dev.httpx, "AsyncClient", client)
    assert await price_for("deepseek/deepseek-flash") is not None
    cached = models_dev.cache_file()
    assert json.loads(cached.read_text()) == CATALOG
    # Fresh: no fetch.
    assert await price_for("anthropic/claude-haiku-5-5") is not None
    assert len(served) == 1
    old = cached.stat().st_mtime - 2 * models_dev.MAX_AGE
    os.utime(cached, (old, old))
    assert await price_for("deepseek/deepseek-flash") is not None
    assert not served
