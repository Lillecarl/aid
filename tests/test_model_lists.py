"""Model lists: which providers a listing names, and one provider's models from its own list, models.dev and the
day's cache."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

import aid
from aid import model_lists
from aid.model_lists import ENV_LISTS, collect, providers
from aid.models_dev import ENV_SOURCE
from aid.protocol import ModelCost, ModelEntry

if TYPE_CHECKING:
    from pathlib import Path

    from aid.model import Transport
    from aid.paths import Paths

pytestmark = pytest.mark.anyio

CATALOG: dict[str, Any] = {
    "zen": {
        "api": "https://zen.test/v1",
        "npm": "@ai-sdk/openai-compatible",
        "env": ["ZEN_KEY"],
        "models": {
            "big": {
                "id": "big",
                "name": "Big",
                "tool_call": True,
                "reasoning": False,
                "limit": {"context": 1000, "output": 100},
                "cost": {"input": 1, "output": 2},
            },
            "old": {"id": "old", "status": "deprecated"},
            "broken": {"name": "no id"},
        },
    },
    "bedrock": {"npm": "@ai-sdk/amazon-bedrock", "env": ["AWS_KEY"]},
}

fetched: list[list[ModelEntry] | Exception] = []


class Zen(aid.ModelProvider):
    """Zen's own list."""

    key_env = "ZEN_KEY"

    async def transport(self, model: str) -> Transport:
        raise NotImplementedError

    async def fetch_models(self) -> list[ModelEntry]:
        result = fetched.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "models.dev.json"
    source.write_text(json.dumps(CATALOG))
    monkeypatch.setenv(ENV_SOURCE, str(source))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv(ENV_LISTS, raising=False)


def test_plugins_win_then_builtin_then_catalog_providers() -> None:
    rows = providers(CATALOG, {"zen": Zen}, env={"ZEN_KEY": "k"})
    by_name = {row.name: row for row in rows}
    assert (by_name["zen"].source, by_name["zen"].description, by_name["zen"].has_key) == (
        "plugin",
        "Zen's own list.",
        True,
    )
    assert (by_name["anthropic"].source, by_name["anthropic"].key_env) == ("builtin", "ANTHROPIC_API_KEY")
    assert by_name["groq"].source == "builtin"
    assert "bedrock" not in by_name
    assert [row.name for row in rows] == sorted(by_name)
    plain = {row.name: row for row in providers(CATALOG, {}, env={})}
    assert (plain["zen"].source, plain["zen"].key_env, plain["zen"].has_key) == ("catalog", "ZEN_KEY", False)


async def test_a_catalog_provider_lists_models_dev_without_deprecated_ones(catalog: None) -> None:
    [row] = (await collect("zen", {})).providers
    assert row.models == [
        ModelEntry(
            id="big",
            name="Big",
            context=1000,
            output=100,
            reasoning=False,
            tools=True,
            cost=ModelCost(input=1, output=2),
        )
    ]
    assert (await collect("nowhere", {})).providers == []
    assert all(row.models is None for row in (await collect(None, {})).providers)


async def test_a_plugins_own_list_is_backfilled_cached_and_kept_when_a_fetch_fails(
    catalog: None, tmp_path: Path
) -> None:
    fetched[:] = [[ModelEntry(id="big", context=2000), ModelEntry(id="new")], RuntimeError("down")]
    [row] = (await collect("zen", {"zen": Zen})).providers
    assert row.problem is None
    assert row.models == [
        ModelEntry(
            id="big",
            name="Big",
            context=2000,
            output=100,
            reasoning=False,
            tools=True,
            cost=ModelCost(input=1, output=2),
        ),
        ModelEntry(id="new"),
    ]
    # Fresh: the cache answers, nothing is fetched.
    assert (await collect("zen", {"zen": Zen})).providers[0].models == row.models
    assert len(fetched) == 1
    cached = tmp_path / "cache" / "aid" / "models" / "zen.json"
    old = cached.stat().st_mtime - 2 * model_lists.models_dev.MAX_AGE
    os.utime(cached, (old, old))
    [stale] = (await collect("zen", {"zen": Zen})).providers
    assert stale.problem == "RuntimeError: down"
    assert stale.models == row.models
    assert fetched == []


async def test_the_daemon_lists_models_from_a_subprocess(daemon: Paths) -> None:
    async with aid.connect(daemon) as client:
        listing = await client.models()
        anthropic = await client.models("anthropic")
    names = {row.name for row in listing.providers}
    assert {"anthropic", "openai", "groq"} <= names
    [row] = anthropic.providers
    assert (row.name, row.source, row.models) == ("anthropic", "builtin", [])
