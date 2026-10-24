"""`python -m aid.model_lists [--provider NAME]`: model providers, and one provider's models, as JSON
(`ModelListing`).

The daemon runs this rather than importing plugins (`listing`). Providers come from three places, the first to
name a prefix wins: `aid.ModelProvider` plugins, aid's own (anthropic and `aid.model.OPENAI_COMPATIBLE`), and
models.dev's Chat Completions providers (`aid.providers.catalog_provider`).

A plugin's own list (`ModelProvider.fetch_models`) is cached a day under the cache dir, `models/<provider>.json`,
and models.dev fills what its entries leave None. A failed fetch keeps the cached copy, else models.dev's list
stands, and `problem` says why. `AID_MODEL_LISTS=off` fetches nothing: tests set it.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any, Final

import anyio
from pydantic import TypeAdapter, ValidationError

from aid import models_dev
from aid.catalog import session_catalog
from aid.model import OPENAI_COMPATIBLE
from aid.models_dev import cache_dir, stale, write_aside
from aid.protocol import AidError, ModelEntry, ModelListing, ProviderModels
from aid.providers import catalog_provider, claimed

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from aid.agents import ModelProvider

ENV_LISTS: Final = "AID_MODEL_LISTS"
TIMEOUT: Final = 60.0
"""The subprocess's whole run: each provider fetch is bounded by its own client too."""
ANTHROPIC_KEY_ENV: Final = "ANTHROPIC_API_KEY"

_entries: TypeAdapter[list[ModelEntry]] = TypeAdapter(list[ModelEntry])


def _row(
    name: str, source: str, key_env: str | None, description: str = "", *, env: Mapping[str, str]
) -> ProviderModels:
    return ProviderModels.model_validate(
        {
            "name": name,
            "source": source,
            "description": description,
            "key_env": key_env or None,
            "has_key": bool(key_env and env.get(key_env)),
        }
    )


def providers(
    catalog: Any, plugins: Mapping[str, type[ModelProvider]], env: Mapping[str, str] | None = None
) -> list[ProviderModels]:
    """Every provider a name can resolve through, by prefix, without their models."""
    env = os.environ if env is None else env
    rows = {
        name: _row(name, "plugin", cls.key_env, cls.description(), env=env) for name, cls in sorted(plugins.items())
    }
    rows.setdefault("anthropic", _row("anthropic", "builtin", ANTHROPIC_KEY_ENV, env=env))
    for name, provider in OPENAI_COMPATIBLE.items():
        rows.setdefault(name, _row(name, "builtin", provider.key_env, env=env))
    for name in models_dev.provider_names(catalog):
        if claimed(name, plugins) is not None:
            continue
        if name not in rows and (provider := catalog_provider(catalog, name)) is not None:
            rows[name] = _row(name, "catalog", provider.key_env, env=env)
    return sorted(rows.values(), key=lambda row: row.name)


def _cache_file(provider: str) -> Path:
    return cache_dir() / "models" / f"{provider}.json"


async def _read_cache(path: Path) -> list[ModelEntry] | None:
    try:
        return _entries.validate_json(await anyio.Path(path).read_bytes())
    except OSError, ValidationError:
        return None


async def own_models(cls: type[ModelProvider], name: str) -> tuple[list[ModelEntry], str | None]:
    """The plugin's own list through the day's cache, and why a fetch failed, if one did."""
    path = _cache_file(name)
    problem: str | None = None
    if os.environ.get(ENV_LISTS) != "off" and stale(path):
        try:
            fetched = await cls().fetch_models()
        except Exception as error:
            problem = f"{type(error).__name__}: {error}"
        else:
            await write_aside(path, _entries.dump_json(fetched))
    return await _read_cache(path) or [], problem


async def collect(provider: str | None, plugins: Mapping[str, type[ModelProvider]]) -> ModelListing:
    catalog = await models_dev.catalog()
    rows = providers(catalog, plugins)
    if provider is None:
        return ModelListing(providers=rows)
    row = next((row for row in rows if row.name == provider), None)
    if row is None:
        return ModelListing(providers=[])
    catalog_id = plugins[provider].catalog_name() if provider in plugins else provider
    known = entry.entries() if (entry := models_dev.provider(catalog, catalog_id)) is not None else {}
    own, problem = await own_models(plugins[provider], provider) if provider in plugins else ([], None)
    models = [model.backfill(known.get(model.id)) for model in own] if own else list(known.values())
    return ModelListing(providers=[row.model_copy(update={"models": models, "problem": problem})])


async def listing(provider: str | None = None) -> ModelListing:
    """The listing from a fresh process, for a caller that must not import plugins."""
    command = [sys.executable, "-m", "aid.model_lists", *(["--provider", provider] if provider else [])]
    with anyio.fail_after(TIMEOUT):
        result = await anyio.run_process(command, check=False)
    if result.returncode != 0:
        raise AidError("models_failed", result.stderr.decode(errors="replace").strip()[-2000:])
    return ModelListing.model_validate_json(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aid.model_lists")
    parser.add_argument("--provider", help="list this provider's models")
    args = parser.parse_args()
    plugins = session_catalog().model_providers
    print(anyio.run(collect, args.provider, plugins).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
