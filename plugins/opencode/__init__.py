"""OpenCode's two gateways, each a provider of its own: Zen (`opencode-zen/`, pay per use; models.dev's `opencode`)
and Go (`opencode-go/`, the subscription), with separate URLs, model lists and keys. models.dev names each model's
protocol and URL (`provider.npm`, `provider.api`): Chat Completions for most, Anthropic's Messages API for `claude-*`, and OpenAI's
Responses API and Google's Gemini API for others, which aid has no transport for yet.

Measured 2026-10-24: both model lists are public; Zen answers over Anthropic Messages (`x-api-key`) and Chat
Completions; Zen's free models refuse every client but OpenCode; Go needs `x-opencode-session`. No Go reply yet.
"""

from __future__ import annotations

import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, ClassVar, Final

import httpx
from pydantic import BaseModel, Field, ValidationError

import aid
from aid import models_dev
from aid.model import AnthropicModel, OpenAIChatModel, Provider
from aid.protocol import ModelEntry

if TYPE_CHECKING:
    from aid.model import Transport

FETCH_TIMEOUT: Final = 10.0
ANTHROPIC_PACKAGE: Final = "@ai-sdk/anthropic"
UNSUPPORTED: Final = {
    "@ai-sdk/openai": "OpenAI's Responses API, which aid has no transport for",
    "@ai-sdk/google": "Google's Gemini API, which aid has no transport for",
}
"""models.dev's package for a model → why aid cannot run it."""


class _Route(BaseModel):
    npm: str | None = None
    api: str | None = None


class _Cost(BaseModel):
    input: float | None = None
    output: float | None = None


class _CatalogModel(BaseModel):
    provider: _Route = Field(default_factory=_Route)
    cost: _Cost | None = None

    @property
    def free(self) -> bool:
        return self.cost is not None and self.cost.input == 0 and self.cost.output == 0


class _Model(BaseModel):
    id: str


class _Models(BaseModel):
    data: list[_Model] = Field(default_factory=list[_Model])


def catalog_models(catalog: models_dev.ModelsDev | None, provider: str) -> dict[str, _CatalogModel]:
    """`provider`'s models as models.dev describes them, by id."""
    entry = models_dev.provider(catalog, provider)
    found: dict[str, _CatalogModel] = {}
    for model_id, raw in (entry.models if entry else {}).items():
        try:
            found[model_id] = _CatalogModel.model_validate(raw)
        except ValidationError:
            continue
    return found


def user_agent() -> str:
    try:
        return f"aid/{version('aid')}"
    except PackageNotFoundError:
        return "aid/dev"


def session_header(session: str) -> str:
    """A stable id per aid session, so OpenCode routes and caches each conversation as one."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aid:session:{session}"))


class _Gateway(aid.ModelProvider):
    """One OpenCode gateway: its prefix and models.dev id, its key, its Chat Completions URL for a model models.dev
    names no URL for, and its public model list (ids only).

    Requests identify as aid and carry `x-opencode-session`, as Go asks of agents other than OpenCode (its docs'
    "Where can I use it?"); without the header Go answers 400 MissingSessionID (measured 2026-10-24)."""

    chat_url: ClassVar[str]
    models_url: ClassVar[str]
    free_tier: ClassVar[str | None] = None
    """Why the gateway's free models refuse aid, if they do."""

    def unsupported(self, model: _CatalogModel) -> str | None:
        if (reason := UNSUPPORTED.get(model.provider.npm or "")) is not None:
            return reason
        return self.free_tier if model.free else None

    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": user_agent()}
        if self.session is not None:
            headers["x-opencode-session"] = session_header(self.session)
        return headers

    async def transport(self, model: str) -> Transport:
        known = catalog_models(await models_dev.catalog(), self.catalog_name()).get(model, _CatalogModel())
        full = f"{self.name}/{model}"
        key_env = self.key_env or ""
        route = known.provider
        if (reason := self.unsupported(known)) is not None:
            raise aid.AidError("unsupported_model", f"{full}: {reason}")
        if route.npm == ANTHROPIC_PACKAGE:
            # The AI SDK's URL ends in /v1; Anthropic's SDK adds /v1/messages itself.
            base = (route.api or "").removesuffix("/").removesuffix("/v1")
            return AnthropicModel(full, Provider(base or None, key_env, headers=self.headers()))
        return OpenAIChatModel(full, Provider(route.api or self.chat_url, key_env, headers=self.headers()))

    async def fetch_models(self) -> list[ModelEntry]:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers={"User-Agent": user_agent()}) as client:
            response = await client.get(self.models_url)
            response.raise_for_status()
        known = catalog_models(await models_dev.catalog(), self.catalog_name())
        return [
            ModelEntry(id=model.id, unsupported=self.unsupported(known.get(model.id, _CatalogModel())))
            for model in _Models.model_validate_json(response.content).data
        ]


class OpenCodeZen(_Gateway):
    """OpenCode Zen, pay per use, keyed by OPENCODE_API_KEY."""

    name = "opencode-zen"
    catalog_id = "opencode"
    key_env = "OPENCODE_API_KEY"
    chat_url = "https://opencode.ai/inference/openai/v1"
    models_url = "https://opencode.ai/zen/v1/models"
    """Measured 2026-10-24; `/inference/openai/v1/models` answers 404."""
    free_tier = "Zen's free tier serves only OpenCode itself (403 FreeTierError, measured 2026-10-24)"


class OpenCodeGo(_Gateway):
    """OpenCode Go, the subscription, keyed by OPENCODE_GO_API_KEY: its own key, not Zen's."""

    name = "opencode-go"
    catalog_id = "opencode-go"
    key_env = "OPENCODE_GO_API_KEY"
    chat_url = "https://opencode.ai/inference/go/openai/v1"
    models_url = "https://opencode.ai/zen/go/v1/models"
    """Measured 2026-10-24; `/inference/go/openai/v1/models` answers 404."""
