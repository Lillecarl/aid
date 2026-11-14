"""OpenAI's Responses API transport (#70): the request it builds, the stream it reads, and which names use it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from aid import responses_api
from aid.agent import Tool
from aid.messages import (
    AttachmentPart,
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
from aid.model import (
    ENCRYPTED,
    ModelInfo,
    OpenAIChatModel,
    OpenAIResponsesModel,
    RequestUsage,
    StreamDelta,
    accumulate,
    model_for,
    responses_input,
)
from aid.protocol import AidError
from aid.providers import resolve
from aid_models.transport import ModelUnavailable
from tests.test_model import serve_wire, sse

if TYPE_CHECKING:
    import httpx2
    from pydantic import JsonValue

    from aid.messages import ModelMessage

KEY = {"api_key": "k", "max_retries": 0}


def turn(*items: dict[str, Any]) -> bytes:
    """A stream: created, the items, completed with usage."""
    created = {"type": "response.created", "response": {"id": "r", "model": "gpt-test-2026", "status": "in_progress"}}
    usage = {
        "input_tokens": 50,
        "input_tokens_details": {"cached_tokens": 40},
        "output_tokens": 9,
        "output_tokens_details": {"reasoning_tokens": 6},
    }
    completed = {"type": "response.completed", "response": {"id": "r", "status": "completed", "usage": usage}}
    return sse([created, *items, completed])


THINK_AND_CALL: list[dict[str, Any]] = [
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
    {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0, "item_id": "rs_1"},
    {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "Plan."},
    {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 1, "item_id": "rs_1"},
    {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 1, "delta": "Act."},
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "sealed-1"},
    },
    {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": ""},
    },
    {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_1", "delta": '{"a": 1,'},
    {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_1", "delta": ' "b": 2}'},
    {
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": "{}"},
    },
]


def answering(body: bytes, seen: list[httpx2.Request]) -> Any:
    import httpx2  # noqa: PLC0415

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    return handle


async def add(a: int, b: int) -> int:
    return a + b


async def deltas(model: OpenAIResponsesModel, history: list[ModelMessage], info: ModelInfo) -> list[StreamDelta]:
    try:
        return [delta async for delta in model.stream(history, info)]
    finally:
        await model.aclose()


async def test_a_stream_reads_sealed_reasoning_a_call_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary parts join as one thought carrying the sealed item; arguments stream into the call; cached tokens
    count inside the prompt, as Chat Completions counts them."""
    seen: list[httpx2.Request] = []
    serve_wire(monkeypatch, answering(turn(*THINK_AND_CALL), seen))
    model = OpenAIResponsesModel("openai/gpt-test")
    got = await deltas(model, [ModelRequest(parts=[UserPromptPart("add 1 and 2")])], ModelInfo(settings=KEY))
    response = ModelResponse(parts=[])
    indexes: dict[int, int] = {}
    for delta in got:
        accumulate(response, indexes, delta)
    assert response.parts == [
        ThinkingPart("Plan.\n\nAct.", **{ENCRYPTED: "sealed-1"}),
        ToolCallPart("add", '{"a": 1, "b": 2}', "call_1"),
    ]
    assert [d.model_name for d in got if d.model_name] == ["gpt-test-2026"]
    assert [d.usage for d in got if d.usage] == [
        RequestUsage(input_tokens=50, output_tokens=9, cache_read_tokens=40, reasoning_tokens=6)
    ]
    assert str(seen[0].url) == "https://api.openai.com/v1/responses"
    assert seen[0].headers["authorization"] == "Bearer k"


async def test_the_request_is_stateless_and_tools_are_not_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured: a tool without `strict` is strict, which aid's schemas are not written for. The session's
    settings go over the defaults, and a reasoning setting keeps the summary."""
    seen: list[httpx2.Request] = []
    serve_wire(monkeypatch, answering(turn(), seen))
    settings: dict[str, JsonValue] = {**KEY, "reasoning": {"effort": "low"}, "max_output_tokens": 99}
    model = OpenAIResponsesModel("openai/gpt-test")
    await deltas(model, [ModelRequest(parts=[UserPromptPart("go")])], ModelInfo(tools=[Tool(add)], settings=settings))
    sent = json.loads(seen[0].content)
    assert (sent["model"], sent["stream"], sent["store"]) == ("gpt-test", True, False)
    assert sent["include"] == [responses_api.INCLUDE_REASONING]
    assert sent["reasoning"] == {"summary": "auto", "effort": "low"}
    assert sent["max_output_tokens"] == 99
    assert "timeout" not in sent and "api_key" not in sent
    [tool] = sent["tools"]
    assert (tool["type"], tool["name"], tool["strict"]) == ("function", "add", False)
    assert tool["parameters"]["required"] == ["a", "b"]


def test_history_replays_sealed_reasoning_and_drops_what_it_cannot() -> None:
    """Sealed reasoning goes back before what it led to, with no id and an empty summary (the same bytes every
    turn); unsealed thinking and trailing reasoning stay out; a tool's image goes inside its output."""
    retry = RetryPromptPart("bad arguments", tool_name="read", tool_call_id="call_2")
    again = RetryPromptPart("answer in JSON")
    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart("be brief"), UserPromptPart("look"), AttachmentPart("image/png", "aa")]),
        ModelResponse(
            parts=[
                ThinkingPart("other api's thought", signature="anthropic-sig"),
                ThinkingPart("Plan.", **{ENCRYPTED: "sealed-1"}),
                TextPart("Looking"),
                TextPart(" now."),
                ToolCallPart("read", {"path": "a.png"}, "call_1"),
                ToolCallPart("read", '{"path":', "call_2"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("read", "a.png: image/png", "call_1"),
                AttachmentPart("image/png", "bb", tool_call_id="call_1"),
                retry,
            ]
        ),
        ModelResponse(parts=[TextPart("Blue."), ThinkingPart("cut off", **{ENCRYPTED: "sealed-2"})]),
        ModelResponse(parts=[]),
        ModelRequest(parts=[again]),
    ]
    items = responses_input(history, {"aa": "QUFB"}, {"bb": "file-9"})
    assert items == [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,QUFB"},
            ],
        },
        {"type": "reasoning", "summary": [], "encrypted_content": "sealed-1"},
        {"role": "assistant", "content": "Looking now."},
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": '{"path": "a.png"}'},
        {"type": "function_call", "call_id": "call_2", "name": "read", "arguments": '{"path":'},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "a.png: image/png"},
                {"type": "input_image", "detail": "auto", "file_id": "file-9"},
            ],
        },
        {"type": "function_call_output", "call_id": "call_2", "output": retry.model_response()},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": again.model_response()},
    ]


@pytest.mark.parametrize(
    ("ending", "outcome"),
    [
        ({"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}, None),
        ({"type": "response.failed", "response": {"error": {"code": "server_error", "message": "x"}}}, "unavailable"),
        ({"type": "error", "code": "rate_limit_exceeded", "message": "slow"}, "unavailable"),
        ({"type": "error", "code": "invalid_prompt", "message": "no"}, "model_error"),
    ],
)
async def test_how_a_stream_ends(monkeypatch: pytest.MonkeyPatch, ending: dict[str, Any], outcome: str | None) -> None:
    created = {"type": "response.created", "response": {"model": "gpt-test"}}
    text = {"type": "response.output_text.delta", "output_index": 0, "item_id": "m", "content_index": 0, "delta": "a"}
    serve_wire(monkeypatch, answering(sse([created, text, ending]), []))
    model = OpenAIResponsesModel("openai/gpt-test")
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    if outcome is None:
        assert any(d.truncated for d in await deltas(model, go, ModelInfo(settings=KEY)))
    elif outcome == "unavailable":
        with pytest.raises(ModelUnavailable):
            await deltas(model, go, ModelInfo(settings=KEY))
    else:
        with pytest.raises(AidError) as failed:
            await deltas(model, go, ModelInfo(settings=KEY))
        assert failed.value.code == outcome


async def test_openai_names_pick_responses_unless_another_server_is_named() -> None:
    """An `api_base` on `openai/` names a server that speaks OpenAI's API: Chat Completions, as before (#70)."""
    other = {"api_base": "https://example.test/v1"}
    assert isinstance(model_for("openai/gpt-x"), OpenAIResponsesModel)
    assert isinstance(model_for("gpt-x"), OpenAIResponsesModel)
    assert isinstance(model_for("openai/local", other), OpenAIChatModel)
    assert isinstance(model_for("groq/llama"), OpenAIChatModel)
    for name, settings, kind in [
        ("openai/gpt-x", {}, OpenAIResponsesModel),
        ("openai/local", other, OpenAIChatModel),
    ]:
        transport = await resolve(name, {}, settings=settings)
        try:
            assert isinstance(transport, kind)
        finally:
            await transport.aclose()


def test_responses_events_parse_as_the_sdk_parses_them() -> None:
    """Differential: aid's event models against the OpenAI SDK's, on the same JSON; aid's input items validate as
    the SDK's request items."""
    from openai.types.responses import (  # noqa: PLC0415 -- a test dependency only
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseInputParam,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseReasoningSummaryTextDeltaEvent,
        ResponseTextDeltaEvent,
    )
    from pydantic import TypeAdapter  # noqa: PLC0415

    usage = {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 7,
    }
    response: dict[str, Any] = {
        "id": "r",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": usage,
    }
    data = json.dumps({"type": "response.completed", "response": response, "sequence_number": 9})
    completed, sdk_completed = responses_api.event(data), ResponseCompletedEvent.model_validate_json(data)
    assert isinstance(completed, responses_api.ResponseEvent) and completed.response.usage is not None
    ours, theirs = completed.response.usage, sdk_completed.response.usage
    assert theirs is not None and ours.input_tokens_details and ours.output_tokens_details
    assert (completed.response.model, ours.input_tokens, ours.output_tokens) == (
        sdk_completed.response.model,
        theirs.input_tokens,
        theirs.output_tokens,
    )
    assert (ours.input_tokens_details.cached_tokens, ours.output_tokens_details.reasoning_tokens) == (
        theirs.input_tokens_details.cached_tokens,
        theirs.output_tokens_details.reasoning_tokens,
    )

    call = {"type": "function_call", "id": "fc", "call_id": "c", "name": "n", "arguments": "{}", "status": "completed"}
    reasoning: dict[str, Any] = {"type": "reasoning", "id": "rs", "summary": [], "encrypted_content": "e"}
    added: list[tuple[type[ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent], dict[str, Any]]] = [
        (ResponseOutputItemAddedEvent, call),
        (ResponseOutputItemDoneEvent, reasoning),
    ]
    for kind, item in added:
        name = "added" if kind is ResponseOutputItemAddedEvent else "done"
        data = json.dumps(
            {"type": f"response.output_item.{name}", "output_index": 1, "sequence_number": 1, "item": item}
        )
        parsed, sdk_item = responses_api.event(data), kind.model_validate_json(data).item
        assert isinstance(parsed, responses_api.ItemEvent)
        sdk = sdk_item.model_dump()
        assert (parsed.item.type, parsed.item.call_id, parsed.item.name, parsed.item.arguments) == (
            sdk["type"],
            sdk.get("call_id"),
            sdk.get("name"),
            sdk.get("arguments"),
        )
        assert parsed.item.encrypted_content == sdk.get("encrypted_content")

    streamed: list[
        tuple[
            type[
                ResponseTextDeltaEvent
                | ResponseFunctionCallArgumentsDeltaEvent
                | ResponseReasoningSummaryTextDeltaEvent
            ],
            dict[str, Any],
        ]
    ] = [
        (
            ResponseTextDeltaEvent,
            {"type": "response.output_text.delta", "item_id": "m", "content_index": 0, "logprobs": []},
        ),
        (ResponseFunctionCallArgumentsDeltaEvent, {"type": "response.function_call_arguments.delta", "item_id": "fc"}),
        (
            ResponseReasoningSummaryTextDeltaEvent,
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs", "summary_index": 1},
        ),
    ]
    for kind, item in streamed:
        data = json.dumps({**item, "output_index": 2, "delta": "x", "sequence_number": 3})
        parsed, sdk_delta = responses_api.event(data), kind.model_validate_json(data)
        assert isinstance(parsed, responses_api.TextDelta | responses_api.ArgumentsDelta | responses_api.SummaryDelta)
        assert (parsed.type, parsed.output_index, parsed.delta) == (
            sdk_delta.type,
            sdk_delta.output_index,
            sdk_delta.delta,
        )

    data = json.dumps({"type": "error", "code": "server_error", "message": "m", "param": None, "sequence_number": 6})
    failed, sdk_error = responses_api.event(data), ResponseErrorEvent.model_validate_json(data)
    assert isinstance(failed, responses_api.ErrorEvent)
    assert (failed.code, failed.message) == (sdk_error.code, sdk_error.message)

    items = responses_input(
        [
            ModelRequest(parts=[SystemPromptPart("s"), UserPromptPart("u"), AttachmentPart("image/png", "aa")]),
            ModelResponse(parts=[ThinkingPart("", **{ENCRYPTED: "e"}), TextPart("t"), ToolCallPart("n", "{}", "c")]),
            ModelRequest(parts=[ToolReturnPart("n", "r", "c"), AttachmentPart("image/png", "aa", tool_call_id="c")]),
        ],
        {"aa": "QUFB"},
    )
    # The SDK's reasoning item requires an `id`; the server takes one without (measured 2026-11-13), which keeps
    # replay free of OpenAI's stored items.
    sdk_shaped = [item for item in items if item.get("type") != "reasoning"]
    assert len(sdk_shaped) == len(items) - 1
    TypeAdapter[ResponseInputParam](ResponseInputParam).validate_python(sdk_shaped, strict=True)
