"""Model providers: which transport a `provider/model` name gets, from a plugin or the generic resolution."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from aid.discovery import discover
from aid.messages import ModelMessage, ModelRequest, UserPromptPart
from aid.model import AnthropicModel, ModelInfo, OpenAIChatModel
from aid.models_dev import ENV_SOURCE
from aid.protocol import AidError
from aid.providers import catalog_provider, resolve

if TYPE_CHECKING:
    from pathlib import Path

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

    def transport(self, model: str) -> OpenAIChatModel:
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
