"""The Gemini transport: what goes on Google's wire, what comes back, and the tool loop through the native backend."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import google.genai  # noqa: F401  # pyright: ignore[reportUnusedImport] -- before `serve` patches httpx: the SDK subclasses httpx.AsyncClient at import
import httpx
import pytest

from aid import model as model_module
from aid.agent import Agent
from aid.backends.native import add_cost
from aid.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import GoogleModel, ModelInfo, RequestUsage, google_contents, model_for
from aid.prices import Price
from aid.protocol import AidError, SessionEvent, Usage
from aid.providers import resolve
from tests.test_native_backend import backend

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

SIGNATURE = base64.b64encode(b"signed").decode()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def chunk(parts: list[dict[str, Any]], *, usage: dict[str, Any] | None = None, finish: str | None = None) -> bytes:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}, "index": 0}
    if finish:
        candidate["finishReason"] = finish
    body: dict[str, Any] = {"candidates": [candidate], "modelVersion": "gemini-test"}
    if usage:
        body["usageMetadata"] = usage
    return f"data: {json.dumps(body)}\n\n".encode()


def serve(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[httpx.Request]:
    """Google's SDK, over aid's own httpx client, talks to `answer(request, body)` instead of Google."""
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = answer(request, json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(model_module.httpx, "AsyncClient", client)
    return seen


def test_history_goes_out_as_contents_with_signatures_on_their_parts() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart("be brief"), UserPromptPart("add")]),
        ModelResponse(
            parts=[
                ThinkingPart("unsigned, dropped"),
                ThinkingPart("signed", signature=SIGNATURE),
                ToolCallPart("add", {"a": 2}, "aid1", google_id="g1", thought_signature=SIGNATURE),
                ToolCallPart("add", {"a": 3}, "aid2"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("add", 5, "aid1"),
                RetryPromptPart("bad", tool_name="add", tool_call_id="aid2"),
                SystemPromptPart("later"),
            ]
        ),
        ModelResponse(parts=[TextPart("done", thought_signature=SIGNATURE)]),
    ]
    system, contents = google_contents(history)
    assert system == [{"text": "be brief"}]
    assert contents == [
        {"role": "user", "parts": [{"text": "add"}]},
        {
            "role": "model",
            "parts": [
                {"text": "signed", "thought": True, "thought_signature": b"signed"},
                {"function_call": {"name": "add", "args": {"a": 2}, "id": "g1"}, "thought_signature": b"signed"},
                {"function_call": {"name": "add", "args": {"a": 3}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"function_response": {"name": "add", "response": {"output": "5"}, "id": "g1"}},
                {"function_response": {"name": "add", "response": {"error": RetryPromptPart("bad").model_response()}}},
                {"text": "later"},
            ],
        },
        {"role": "model", "parts": [{"text": "done", "thought_signature": b"signed"}]},
    ]


async def test_tool_loop_replays_gemini_signatures_and_counts_its_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx.Request, body: dict[str, Any]) -> bytes:
        bodies.append(body)
        usage = {"promptTokenCount": 100, "cachedContentTokenCount": 60, "candidatesTokenCount": 5}
        if len(bodies) == 1:
            return chunk([{"text": "planning", "thought": True}]) + chunk(
                [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}, "thoughtSignature": SIGNATURE}],
                usage={**usage, "thoughtsTokenCount": 7},
                finish="STOP",
            )
        return chunk([{"text": "fi"}]) + chunk([{"text": "ve", "thoughtSignature": SIGNATURE}], usage=usage)

    seen = serve(monkeypatch, answer)

    async def add(a: int, b: int) -> int:
        return a + b

    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent("google/gemini-test", tools=[add], instructions="be brief", model_settings={"api_key": "k"})
    async with backend(tmp_path, agent) as session:
        result = await session.prompt("add", emit)
    assert result.output == "five"
    first, _ = seen
    assert first.url.path == "/v1beta/models/gemini-test:streamGenerateContent"
    assert (first.url.params["alt"], first.headers["x-goog-api-key"]) == ("sse", "k")
    assert bodies[0]["systemInstruction"]["parts"] == [{"text": "be brief"}]
    [declaration] = bodies[0]["tools"][0]["functionDeclarations"]
    assert declaration["name"] == "add"
    assert declaration["parameters_json_schema"]["required"] == ["a", "b"]
    [call, returned] = bodies[1]["contents"][1:]
    assert call["parts"] == [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}, "thoughtSignature": SIGNATURE}]
    assert returned["parts"] == [{"functionResponse": {"name": "add", "response": {"output": "5"}}}]
    [usage] = [item for item in events if isinstance(item, Usage)]
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens, usage.thought_tokens) == (
        200,
        120,
        17,
        7,
    )
    assert usage.models == ["gemini-test"]
    history = ModelMessagesTypeAdapter.validate_json((tmp_path / "history.json").read_bytes())
    final = history[-1]
    assert isinstance(final, ModelResponse)
    assert final.parts == [TextPart("five", thought_signature=SIGNATURE)]


async def test_a_truncated_response_and_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx.Request, body: dict[str, Any]) -> bytes:
        bodies.append(body)
        return chunk([{"text": '{"a":'}], finish="MAX_TOKENS")

    serve(monkeypatch, answer)
    model = GoogleModel("google/gemini-test")
    deltas = [
        delta
        async for delta in model.stream(
            [ModelRequest(parts=[UserPromptPart("go")])],
            ModelInfo(output_type=dict[str, int], settings={"api_key": "k", "max_output_tokens": 3}),
        )
    ]
    await model.aclose()
    assert any(delta.truncated for delta in deltas)
    config = bodies[0]["generationConfig"]
    assert (config["responseMimeType"], config["maxOutputTokens"]) == ("application/json", 3)
    assert config["responseJsonSchema"]["type"] == "object"


async def test_google_names_resolve_to_the_gemini_api_by_their_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = await resolve("google/gemini-test", {})
    assert isinstance(model, GoogleModel)
    assert isinstance(model_for("google/gemini-test"), GoogleModel)
    with pytest.raises(AidError) as caught:
        await anext(model.stream([ModelRequest(parts=[UserPromptPart("go")])], ModelInfo()))
    assert (caught.value.code, "GEMINI_API_KEY" in caught.value.message) == ("missing_credentials", True)


def test_cached_tokens_bill_once_inside_the_prompt() -> None:
    price = Price(input=1.0, output=10.0, cache_read=0.1, cache_write=1.0)
    usage = RequestUsage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=600_000)
    cost = add_cost(None, price, usage, GoogleModel("google/gemini-test"))
    assert cost.amount == pytest.approx(0.4 + 0.06 + 1.0)
