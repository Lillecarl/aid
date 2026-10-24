"""The bundled OpenRouter and OpenCode providers: what their requests carry, how they route, and their lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import httpx2
import openai
import pytest

from aid.discovery import discover
from aid.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from aid.model import AnthropicModel, ModelInfo, OpenAIChatModel
from aid.models_dev import ENV_SOURCE
from aid.protocol import AidError
from plugins import opencode, openrouter
from tests.test_model import event

pytestmark = pytest.mark.anyio

PLUGINS = Path(__file__).parent.parent / "plugins"
ZEN = "https://opencode.ai/inference"
CATALOG: dict[str, Any] = {
    "opencode": {
        "api": f"{ZEN}/openai/v1",
        "npm": "@ai-sdk/openai-compatible",
        "env": ["OPENCODE_API_KEY"],
        "models": {
            "big-pickle": {"id": "big-pickle", "provider": {"api": f"{ZEN}/openai/v1"}},
            "claude-x": {"id": "claude-x", "provider": {"npm": "@ai-sdk/anthropic", "api": f"{ZEN}/anthropic/v1"}},
            "gpt-x": {"id": "gpt-x", "provider": {"npm": "@ai-sdk/openai", "api": f"{ZEN}/openai/v1"}},
            "gemini-x": {"id": "gemini-x", "provider": {"npm": "@ai-sdk/google", "api": f"{ZEN}/google/v1beta"}},
        },
    },
    "opencode-go": {
        "api": f"{ZEN}/go/openai/v1",
        "npm": "@ai-sdk/openai-compatible",
        "env": ["OPENCODE_API_KEY"],
        "models": {
            "go-chat": {"id": "go-chat", "provider": {"api": f"{ZEN}/go/openai/v1"}},
            "claude-x": {"id": "claude-x", "provider": {"npm": "@ai-sdk/anthropic", "api": f"{ZEN}/go/anthropic/v1"}},
        },
    },
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "models.dev.json"
    source.write_text(json.dumps(CATALOG))
    monkeypatch.setenv(ENV_SOURCE, str(source))


def serve(monkeypatch: pytest.MonkeyPatch, module: Any, body: Any) -> None:
    """`module`'s httpx client answers every request with `body` as JSON."""
    real = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)), **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client)


def capture(monkeypatch: pytest.MonkeyPatch, data: bytes) -> list[httpx2.Request]:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n")

    def make(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make)
    return seen


def test_both_providers_are_discovered() -> None:
    found = discover([PLUGINS], only=["openrouter", "opencode"])
    assert found.problems == []
    assert sorted(found.model_providers) == ["opencode-go", "opencode-zen", "openrouter"]


async def test_openrouter_asks_for_usage_reads_reasoning_and_replays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture(monkeypatch, event({"reasoning": "hmm"}) + event({"content": "ok"}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    model = await openrouter.OpenRouter().transport("vendor/model")
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[ThinkingPart("before"), TextPart("said")]),
        ModelRequest(parts=[UserPromptPart("again")]),
    ]
    try:
        response = await model.request(history, ModelInfo(settings={"extra_body": {"provider": {"sort": "price"}}}))
    finally:
        await model.aclose()
    [request] = seen
    body = json.loads(request.content)
    assert body["model"] == "vendor/model"
    assert (body["usage"], body["provider"]) == ({"include": True}, {"sort": "price"})
    assert all("reasoning_content" not in message for message in body["messages"])
    assert request.headers["X-Title"] == "aid"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert response.parts == [ThinkingPart("hmm"), TextPart("ok")]


async def test_openrouter_lists_prices_per_million(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(
        monkeypatch,
        openrouter,
        {
            "data": [
                {
                    "id": "vendor/model",
                    "name": "Model",
                    "context_length": 1000,
                    "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002"},
                    "top_provider": {"max_completion_tokens": 100},
                    "supported_parameters": ["tools", "max_tokens"],
                },
                {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            ]
        },
    )
    model, router = await openrouter.OpenRouter().fetch_models()
    assert (model.name, model.context, model.output, model.tools, model.reasoning) == ("Model", 1000, 100, True, False)
    assert model.cost is not None
    assert (model.cost.input, model.cost.output, model.cost.cache_read) == pytest.approx((2.0, 10.0, 0.2))
    assert model.cost.cache_write is None
    assert (router.cost, router.tools) == (None, None)


async def test_opencode_routes_each_model_by_its_protocol(catalog: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    provider = opencode.OpenCodeZen()
    chat = await provider.transport("big-pickle")
    claude = await provider.transport("claude-x")
    unknown = await provider.transport("not-catalogued")
    assert isinstance(chat, OpenAIChatModel)
    assert isinstance(claude, AnthropicModel)
    assert isinstance(unknown, OpenAIChatModel)
    assert chat._client({}).base_url.host == "opencode.ai"  # pyright: ignore[reportPrivateUsage]
    assert str(chat._client({}).base_url) == f"{ZEN}/openai/v1/"  # pyright: ignore[reportPrivateUsage]
    # Anthropic's SDK posts to <base>/v1/messages: models.dev's URL plus /messages.
    assert str(claude._client({}).base_url).rstrip("/") == f"{ZEN}/anthropic"  # pyright: ignore[reportPrivateUsage]
    assert claude.model_name == "opencode-zen/claude-x"
    for unsupported in ("gpt-x", "gemini-x"):
        with pytest.raises(AidError) as caught:
            await provider.transport(unsupported)
        assert caught.value.code == "unsupported_model"
    for model in (chat, claude, unknown):
        await model.aclose()


async def test_opencode_go_is_its_own_gateway_with_its_own_key(catalog: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "zen")
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    go = opencode.OpenCodeGo()
    chat = await go.transport("go-chat")
    claude = await go.transport("claude-x")
    assert isinstance(chat, OpenAIChatModel)
    assert isinstance(claude, AnthropicModel)
    assert (chat.model_name, claude.model_name) == ("opencode-go/go-chat", "opencode-go/claude-x")
    with pytest.raises(AidError) as caught:
        chat._client({})  # pyright: ignore[reportPrivateUsage]
    assert (caught.value.code, "OPENCODE_GO_API_KEY" in caught.value.message) == ("missing_credentials", True)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go")
    assert str(chat._client({}).base_url) == f"{ZEN}/go/openai/v1/"  # pyright: ignore[reportPrivateUsage]
    assert str(claude._client({}).base_url).rstrip("/") == f"{ZEN}/go/anthropic"  # pyright: ignore[reportPrivateUsage]
    assert claude._client({}).api_key == "go"  # pyright: ignore[reportPrivateUsage]
    for model in (chat, claude):
        await model.aclose()


async def test_opencode_lists_its_models_and_marks_responses_ones(
    catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(
        monkeypatch, opencode, {"object": "list", "data": [{"id": "big-pickle"}, {"id": "gpt-x"}, {"id": "gemini-x"}]}
    )
    pickle, gpt, gemini = await opencode.OpenCodeZen().fetch_models()
    assert (pickle.id, pickle.unsupported) == ("big-pickle", None)
    assert (gpt.id, gpt.unsupported) == ("gpt-x", opencode.UNSUPPORTED["@ai-sdk/openai"])
    assert (gemini.id, gemini.unsupported) == ("gemini-x", opencode.UNSUPPORTED["@ai-sdk/google"])
