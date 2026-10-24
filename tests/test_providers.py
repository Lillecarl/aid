"""Model providers: which transport a `provider/model` name gets, from a plugin or the generic resolution."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

import aid
from aid.agent import Agent
from aid.backends.native import session_agent
from aid.discovery import discover
from aid.messages import ModelMessage, ModelRequest, UserPromptPart
from aid.model import AnthropicModel, ModelInfo, OpenAIChatModel, Provider
from aid.model_lists import providers
from aid.models_dev import ENV_SOURCE
from aid.protocol import AidError
from aid.providers import catalog_model, catalog_provider, claimed, resolve
from aid.spec import NativeSpec

if TYPE_CHECKING:
    from pathlib import Path

    from aid.model import Transport

pytestmark = pytest.mark.anyio

CATALOG: dict[str, Any] = {
    "zen": {"api": "https://zen.test/v1", "npm": "@ai-sdk/openai-compatible", "env": ["ZEN_KEY", "ZEN_OTHER"]},
    "bedrock": {"api": None, "npm": "@ai-sdk/amazon-bedrock", "env": ["AWS_KEY"]},
    "native": {"api": "https://native.test/v1", "npm": "@ai-sdk/anthropic", "env": ["NATIVE_KEY"]},
}

PROVIDER = '''import aid
from aid.model import OpenAIChatModel, Provider


class Zen(aid.ModelProvider):
    """Zen through its own plugin."""

    async def transport(self, model: str) -> OpenAIChatModel:
        return OpenAIChatModel(f"zen/{model}", Provider("https://plugin.test/v1", "PLUGIN_KEY"))
'''


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "models.dev.json"
    source.write_text(json.dumps(CATALOG))
    monkeypatch.setenv(ENV_SOURCE, str(source))


def test_only_openai_compatible_catalog_providers_with_a_url_serve() -> None:
    zen = catalog_provider(CATALOG, "zen")
    assert zen is not None
    assert (zen.base_url, zen.key_env) == ("https://zen.test/v1", "ZEN_KEY")
    assert catalog_provider(CATALOG, "bedrock") is None
    assert catalog_provider(CATALOG, "native") is None
    assert catalog_provider(None, "zen") is None


async def test_generic_names_resolve_by_prefix(catalog: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEN_KEY", raising=False)
    assert isinstance(await resolve("anthropic/claude-x", {}), AnthropicModel)
    table = await resolve("deepseek/deepseek-flash", {})
    zen = await resolve("zen/big-model", {})
    assert isinstance(table, OpenAIChatModel)
    assert isinstance(zen, OpenAIChatModel)
    assert zen.model_name == "zen/big-model"
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    for model, code, says in [
        (zen, "missing_credentials", "ZEN_KEY"),
        (await resolve("nowhere/x", {}), "unknown_provider", "api_base"),
    ]:
        with pytest.raises(AidError) as caught:
            await anext(model.stream(go, ModelInfo()))
        assert (caught.value.code, says in caught.value.message) == (code, True)
    monkeypatch.setenv("ZEN_KEY", "k")
    assert zen._client({}).base_url.host == "zen.test"  # pyright: ignore[reportPrivateUsage]
    await zen.aclose()


async def test_a_plugin_provider_wins_its_prefix(
    catalog: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"providers_{uuid.uuid4().hex[:8]}"
    directory = tmp_path / "plugins" / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(PROVIDER)
    found = discover([tmp_path / "plugins"])
    assert found.problems == []
    assert list(found.model_providers) == ["zen"]
    assert found.describe()["model_providers"] == [
        {"name": "zen", "description": "Zen through its own plugin.", "module": name}
    ]
    monkeypatch.setenv("PLUGIN_KEY", "k")
    model = await resolve("zen/big-model", found.model_providers)
    assert isinstance(model, OpenAIChatModel)
    assert model._client({}).base_url.host == "plugin.test"  # pyright: ignore[reportPrivateUsage]
    await model.aclose()


def test_a_sessions_model_takes_the_agents_settings_only_on_the_same_provider() -> None:
    agent = Agent("deepseek/deepseek-flash", model_settings={"thinking": {"type": "enabled"}, "max_tokens": 9})
    plain = NativeSpec(cwd="/", agent="a")
    assert session_agent(agent, plain) is agent
    same = session_agent(
        agent, NativeSpec(cwd="/", agent="a", model="deepseek/deepseek-v4-pro", model_settings={"max_tokens": 5})
    )
    assert (same.model, same.model_settings) == (
        "deepseek/deepseek-v4-pro",
        {"thinking": {"type": "enabled"}, "max_tokens": 5},
    )
    other = session_agent(agent, NativeSpec(cwd="/", agent="a", model="openrouter/vendor/model"))
    assert (other.model, other.model_settings) == ("openrouter/vendor/model", {})
    tuned = session_agent(agent, NativeSpec(cwd="/", agent="a", model_settings={"max_tokens": 1}))
    assert (tuned.model, tuned.model_settings) == (
        "deepseek/deepseek-flash",
        {"thinking": {"type": "enabled"}, "max_tokens": 1},
    )
    assert agent.model_settings["max_tokens"] == 9


class Gateway(aid.ModelProvider):
    """Zen under a name of its own."""

    name = "zen-gateway"
    catalog_id = "zen"

    async def transport(self, model: str) -> Transport:
        return OpenAIChatModel(f"{self.name}/{model}", Provider("https://gateway.test/v1", "GATEWAY_KEY"))


async def test_a_plugin_serving_a_catalog_id_under_another_name_owns_it(catalog: None) -> None:
    plugins: dict[str, type[aid.ModelProvider]] = {"zen-gateway": Gateway}
    assert claimed("zen", plugins) is Gateway
    assert claimed("zen-gateway", plugins) is None
    assert catalog_model("zen-gateway/big", plugins) == "zen/big"
    assert catalog_model("deepseek/deepseek-flash", plugins) == "deepseek/deepseek-flash"
    with pytest.raises(AidError) as caught:
        await resolve("zen/big", plugins)
    assert (caught.value.code, "zen-gateway/" in caught.value.message) == ("unknown_provider", True)
    names = [row.name for row in providers(CATALOG, plugins, env={})]
    assert "zen" not in names
    assert "zen-gateway" in names
