"""The Gemini transport: what goes on Google's wire, what comes back, and the tool loop through the native backend."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Final

import httpx2
import pytest

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
from aid.protocol import AidError, SessionEvent, Usage
from aid.providers import resolve
from aid_models import gemini, transport, wire
from aid_models.prices import Price
from aid_models.transport import ModelUnavailable
from tests.test_native_backend import backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic import JsonValue


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


def mock(monkeypatch: pytest.MonkeyPatch, handle: Callable[[httpx2.Request], httpx2.Response]) -> None:
    def client(timeout: float) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), timeout=timeout)

    monkeypatch.setattr(wire, "client", client)


def serve(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[httpx2.Request]:
    """aid's client talks to `answer(request, body)` instead of Google."""
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        content = answer(request, json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    mock(monkeypatch, handle)
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
    assert [part.model_dump(exclude_none=True) for part in system] == [{"text": "be brief"}]
    assert [content.model_dump(exclude_none=True) for content in contents] == [
        {"role": "user", "parts": [{"text": "add"}]},
        {
            "role": "model",
            "parts": [
                {"text": "signed", "thought": True, "thoughtSignature": SIGNATURE},
                {"functionCall": {"name": "add", "args": {"a": 2}, "id": "g1"}, "thoughtSignature": SIGNATURE},
                {"functionCall": {"name": "add", "args": {"a": 3}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "add", "response": {"output": "5"}, "id": "g1"}},
                {"functionResponse": {"name": "add", "response": {"error": RetryPromptPart("bad").model_response()}}},
                {"text": "later"},
            ],
        },
        {"role": "model", "parts": [{"text": "done", "thoughtSignature": SIGNATURE}]},
    ]


async def test_tool_loop_replays_gemini_signatures_and_counts_its_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx2.Request, body: dict[str, Any]) -> bytes:
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
    assert declaration["parametersJsonSchema"]["required"] == ["a", "b"]
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

    def answer(request: httpx2.Request, body: dict[str, Any]) -> bytes:
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


REFUSED_KEY: Final[JsonValue] = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
    }
}
"""How Gemini refuses a key: 400 INVALID_ARGUMENT, the reason in its details (measured 2026-11-13)."""


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, REFUSED_KEY, "bad_credentials"),
        (400, [REFUSED_KEY], "bad_credentials"),
        (401, {"error": {"code": 401, "message": "no", "status": "UNAUTHENTICATED"}}, "bad_credentials"),
        (503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}, "unavailable"),
        (429, {"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}}, "unavailable"),
        (400, {"error": {"code": 400, "message": "bad field", "status": "INVALID_ARGUMENT"}}, "raised"),
    ],
)
async def test_refusals_are_named(monkeypatch: pytest.MonkeyPatch, status: int, body: JsonValue, expected: str) -> None:
    mock(monkeypatch, lambda request: httpx2.Response(status, json=body))
    model = GoogleModel("google/gemini-test")
    try:
        with pytest.raises(Exception) as caught:
            await anext(
                model.stream(
                    [ModelRequest(parts=[UserPromptPart("go")])],
                    ModelInfo(settings={"api_key": "k", "max_retries": 0}),
                )
            )
    finally:
        await model.aclose()
    match expected:
        case "bad_credentials":
            assert isinstance(caught.value, AidError)
            assert caught.value.code == "bad_credentials"
        case "unavailable":
            assert isinstance(caught.value, ModelUnavailable)
        case _:
            assert isinstance(caught.value, wire.StatusError)
            assert caught.value.status == status


async def test_max_retries_sends_again_before_the_backend_does(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal sent
        sent += 1
        if sent == 1:
            return httpx2.Response(503, json={"error": {"code": 503}})
        return httpx2.Response(200, content=chunk([{"text": "ok"}]))

    mock(monkeypatch, handle)
    monkeypatch.setattr(transport, "RETRY_BACKOFF", 0)
    model = GoogleModel("google/gemini-test")
    try:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart("go")])], ModelInfo(settings={"api_key": "k", "max_retries": 1})
        )
    finally:
        await model.aclose()
    assert (sent, response.parts) == (2, [TextPart("ok")])


async def test_an_unknown_setting_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    mock(monkeypatch, lambda request: httpx2.Response(200, content=chunk([{"text": "ok"}])))
    model = GoogleModel("google/gemini-test")
    try:
        with pytest.raises(AidError) as caught:
            await model.request(
                [ModelRequest(parts=[UserPromptPart("go")])], ModelInfo(settings={"api_key": "k", "temprature": 1})
            )
    finally:
        await model.aclose()
    assert caught.value.code == "bad_model_settings"


def test_chunks_parse_as_google_parses_them() -> None:
    """Differential: aid's response model against google-genai's, on the same chunk JSON (#65)."""
    from google.genai.types import GenerateContentResponse  # noqa: PLC0415 -- a test dependency only

    for raw in (
        chunk([{"text": "planning", "thought": True}]),
        chunk(
            [{"functionCall": {"name": "add", "args": {"a": 2}, "id": "g1"}, "thoughtSignature": SIGNATURE}],
            usage={"promptTokenCount": 100, "cachedContentTokenCount": 60, "candidatesTokenCount": 5},
            finish="STOP",
        ),
        chunk([{"text": "ve", "thoughtSignature": SIGNATURE}], usage={"thoughtsTokenCount": 7}, finish="MAX_TOKENS"),
    ):
        data = raw.decode().removeprefix("data: ").strip()
        ours = gemini.Response.model_validate_json(data)
        theirs = GenerateContentResponse.model_validate_json(data)
        assert ours.model_version == theirs.model_version
        [candidate] = ours.candidates
        [expected] = theirs.candidates or []
        assert candidate.finish_reason == (expected.finish_reason.value if expected.finish_reason else None)
        expected_parts = (expected.content.parts if expected.content else None) or []
        assert len(candidate.content.parts if candidate.content else []) == len(expected_parts)
        for part, other in zip(candidate.content.parts if candidate.content else [], expected_parts, strict=True):
            assert (part.text, part.thought) == (other.text, other.thought)
            signature = base64.b64encode(other.thought_signature).decode() if other.thought_signature else None
            assert part.thought_signature == signature
            if other.function_call is not None:
                assert part.function_call is not None
                call = other.function_call
                assert (part.function_call.name, part.function_call.args, part.function_call.id) == (
                    call.name,
                    call.args,
                    call.id,
                )
        usage, their_usage = ours.usage_metadata, theirs.usage_metadata
        assert (usage is None) == (their_usage is None)
        if usage is not None and their_usage is not None:
            assert (usage.prompt_token_count, usage.cached_content_token_count, usage.candidates_token_count) == (
                their_usage.prompt_token_count or 0,
                their_usage.cached_content_token_count or 0,
                their_usage.candidates_token_count or 0,
            )
            assert usage.thoughts_token_count == their_usage.thoughts_token_count
