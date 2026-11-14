"""The bundled OpenRouter and OpenCode providers: what their requests carry, how they route, and their lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest

from aid.anthropic_api import messages_url
from aid.discovery import discover
from aid.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from aid.model import AnthropicModel, ModelInfo, OpenAIChatModel
from aid.protocol import AidError
from aid_models import chat_api, wire
from aid_models.models_dev import ENV_SOURCE
from plugins import opencode, openrouter
from tests.test_model import event

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
            "free-x": {"id": "free-x", "cost": {"input": 0, "output": 0}},
        },
    },
    "opencode-go": {
        "api": f"{ZEN}/go/openai/v1",
        "npm": "@ai-sdk/openai-compatible",
        "env": ["OPENCODE_API_KEY"],
        "models": {
            "go-chat": {"id": "go-chat", "provider": {"api": f"{ZEN}/go/openai/v1"}},
            "go-free": {"id": "go-free", "cost": {"input": 0, "output": 0}},
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

    monkeypatch.setattr(wire, "client", make)
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
    assert chat.host({}) == "opencode.ai"
    assert chat_api.url(chat._key({}).base_url, "chat/completions") == f"{ZEN}/openai/v1/chat/completions"  # pyright: ignore[reportPrivateUsage]
    # aid posts to <base>/v1/messages, as Anthropic's SDK does: models.dev's URL plus /messages.
    assert messages_url(claude._key({}).base_url) == f"{ZEN}/anthropic/v1/messages"  # pyright: ignore[reportPrivateUsage]
    assert claude.model_name == "opencode-zen/claude-x"
    for unsupported in ("gpt-x", "gemini-x", "free-x"):
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
        chat._key({})  # pyright: ignore[reportPrivateUsage]
    assert (caught.value.code, "OPENCODE_GO_API_KEY" in caught.value.message) == ("missing_credentials", True)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go")
    assert chat_api.url(chat._key({}).base_url, "chat/completions") == f"{ZEN}/go/openai/v1/chat/completions"  # pyright: ignore[reportPrivateUsage]
    key = claude._key({})  # pyright: ignore[reportPrivateUsage]
    assert (messages_url(key.base_url), key.api_key) == (f"{ZEN}/go/anthropic/v1/messages", "go")
    for model in (chat, claude):
        await model.aclose()


async def test_opencode_go_names_aid_and_the_conversation(catalog: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture(monkeypatch, event({"content": "ok"}))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go")
    model = await opencode.OpenCodeGo("tty").transport("go-free")
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    try:
        await model.request(go, ModelInfo())
        await model.request(go, ModelInfo())
    finally:
        await model.aclose()
    first, second = seen
    assert first.headers["User-Agent"].startswith("aid/")
    assert first.headers["x-opencode-session"] == second.headers["x-opencode-session"] == opencode.session_header("tty")
    assert opencode.session_header("tty") != opencode.session_header("other")


async def test_opencode_lists_its_models_and_marks_responses_ones(
    catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(
        monkeypatch,
        opencode,
        {"object": "list", "data": [{"id": "big-pickle"}, {"id": "gpt-x"}, {"id": "gemini-x"}, {"id": "free-x"}]},
    )
    pickle, gpt, gemini, free = await opencode.OpenCodeZen().fetch_models()
    assert (free.id, free.unsupported) == ("free-x", opencode.OpenCodeZen.free_tier)
    assert (pickle.id, pickle.unsupported) == ("big-pickle", None)
    assert (gpt.id, gpt.unsupported) == ("gpt-x", opencode.UNSUPPORTED["@ai-sdk/openai"])
    assert (gemini.id, gemini.unsupported) == ("gemini-x", opencode.UNSUPPORTED["@ai-sdk/google"])
