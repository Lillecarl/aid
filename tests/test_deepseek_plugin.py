"""The bundled DeepSeek provider: discovered from plugins/, its output limit and its cache counting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel

from aid.discovery import discover
from aid.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from aid.model import ModelInfo, RequestUsage
from aid.providers import resolve
from tests.test_model import event

pytestmark = pytest.mark.anyio

PLUGINS = Path(__file__).parent.parent / "plugins"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_deepseek_serves_its_prefix_with_its_limit_and_cache_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "prompt_cache_hit_tokens": 6}

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        body = event({"content": "ok"}) + event({}, usage=usage) + b"data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    catalog = discover([PLUGINS], only=["deepseek"])
    model = await resolve("deepseek/deepseek-flash", catalog.model_providers)
    # Discovery imports the plugin as `deepseek`, apart from `plugins.deepseek`.
    assert type(model).__qualname__ == "DeepSeekChat"
    assert model.model_name == "deepseek/deepseek-flash"
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    usages: list[RequestUsage] = []
    try:
        async for delta in model.stream(go, ModelInfo()):
            if delta.usage:
                usages.append(delta.usage)
        await model.request(go, ModelInfo(settings={"max_tokens": 5}))
    finally:
        await model.aclose()
    assert [(body["model"], body["max_tokens"]) for body in sent] == [("deepseek-flash", 131072), ("deepseek-flash", 5)]
    assert [(u.input_tokens, u.cache_read_tokens) for u in usages] == [(10, 6)]


class Verdict(BaseModel):
    verdict: str


async def test_deepseek_asks_structured_output_in_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek refuses `json_schema`: JSON mode, with the schema in a note after the instructions."""
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        body = event({"content": '{"verdict": "ok"}'}) + b"data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    model = await resolve("deepseek/deepseek-flash", discover([PLUGINS], only=["deepseek"]).model_providers)
    asked: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart("judge"), UserPromptPart("go")])]
    try:
        async for _ in model.stream(asked, ModelInfo(output_type=Verdict)):
            pass
    finally:
        await model.aclose()
    [body] = sent
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "system", "user"]
    assert body["messages"][0]["content"] == "judge"
    assert '"verdict"' in body["messages"][1]["content"]
