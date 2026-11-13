from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import httpx2
import pytest

import aid.backends.native as native_backend
from aid import anthropic_api, chat_api, wire
from aid.agent import Agent, Tool
from aid.discovery import Catalog
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
from aid.model import (
    ANTHROPIC_MAX_TOKENS,
    AnthropicModel,
    GoogleModel,
    ModelInfo,
    OpenAIChatModel,
    OpenAIResponsesModel,
    Provider,
    RequestUsage,
    StreamBroken,
    accumulate,
    anthropic_messages,
    model_for,
    wire_messages,
)
from aid.prices import ENV_PRICES
from aid.protocol import AidError, SessionEvent, Usage
from aid.spec import NativeSpec
from tests.test_native_backend import backend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence
    from pathlib import Path

    from pydantic import BaseModel, JsonValue


def event(delta: dict[str, Any], *, usage: dict[str, Any] | None = None, finish: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    chunk: dict[str, Any] = {
        "id": "response",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "local",
        "choices": [choice],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n".encode()


async def test_openai_transport_tool_loop_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["stream_options"] == {"include_usage": True}
        if len(requests) == 1:
            data = event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":'},
                        }
                    ]
                }
            )
            data += event({"tool_calls": [{"index": 0, "function": {"arguments": '2,"b":3}'}}]})
        else:
            returned = [m for m in body["messages"] if m["role"] == "tool"]
            assert returned == [{"role": "tool", "tool_call_id": "call1", "content": "5"}]
            data = event({"content": "five"})
        data += event({}, usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n")

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_client)

    async def add(a: int, b: int) -> int:
        return a + b

    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent(
        "openai/local", tools=[add], model_settings={"api_key": "test-loop", "api_base": "https://example.test/v1"}
    )
    async with backend(tmp_path, agent) as session:
        result = await session.prompt("add", emit)
    assert result.output == "five"
    usage = [item for item in events if isinstance(item, Usage)]
    assert len(usage) == 1
    assert (usage[0].requests, usage[0].input_tokens, usage[0].output_tokens) == (2, 20, 8)
    assert usage[0].models == ["local"]
    assert usage[0].context_size == native_backend.DEFAULT_DESIRED_MAX
    assert usage[0].context_used is not None and usage[0].context_used > 0
    history = ModelMessagesTypeAdapter.validate_json((tmp_path / "history.json").read_bytes())
    assert all(m.model_name == "local" for m in history if isinstance(m, ModelResponse))
    assert any(isinstance(p, ToolReturnPart) and p.content == 5 for m in history for p in m.parts)
    assert (tmp_path / "history.json").read_bytes().count(b"[Session metadata]") == 1


@pytest.mark.parametrize(("max_context", "gauge"), [(None, 32_000), (5_000, 5_000)])
async def test_the_gauge_is_half_the_model_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_context: int | None, gauge: int
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"openai": {"models": {"local": {"id": "local", "limit": {"context": 64_000}}}}}))
    monkeypatch.setenv(ENV_PRICES, str(catalog))

    def handle(request: httpx2.Request) -> httpx2.Response:
        data = event({"content": "hi"}) + event({}, usage={"prompt_tokens": 10, "completion_tokens": 1})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n")

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_client)
    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent("openai/local", model_settings={"api_key": "test-gauge", "api_base": "https://example.test/v1"})
    async with backend(tmp_path, agent, desired_max=max_context) as session:
        await session.prompt("hello", emit)
    usage = next(item for item in events if isinstance(item, Usage))
    assert usage.context_size == gauge


@pytest.mark.parametrize("breaks", [1, native_backend.STREAM_RETRIES + 1])
async def test_a_broken_stream_goes_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breaks: int) -> None:
    class Breaking(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield event({"reasoning": "thinking"})
            raise httpx2.ReadError("")

    sent = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal sent
        sent += 1
        if sent <= breaks:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Breaking())
        data = event({"content": "done"}) + event({}, usage={"prompt_tokens": 10, "completion_tokens": 1})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n")

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_client)
    monkeypatch.setattr(native_backend, "STREAM_BACKOFF", 0)
    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent("openai/local", model_settings={"api_key": "test-broken", "api_base": "https://example.test/v1"})
    async with backend(tmp_path, agent) as session:
        if breaks > native_backend.STREAM_RETRIES:
            with pytest.raises(AidError) as raised:
                await session.prompt("hello", emit)
            assert raised.value.code == "model_unavailable"
            assert isinstance(raised.value.__cause__, StreamBroken)
        else:
            assert (await session.prompt("hello", emit)).output == "done"
    assert sent == min(breaks + 1, native_backend.STREAM_RETRIES + 1)
    usage = next(item for item in events if isinstance(item, Usage))
    assert usage.requests == sent
    history = ModelMessagesTypeAdapter.validate_json((tmp_path / "history.json").read_bytes())
    # Only the last response stays: the answer, or what the last broken one streamed (a failed turn keeps it).
    [response] = [m for m in history if isinstance(m, ModelResponse)]
    kind = TextPart if breaks <= native_backend.STREAM_RETRIES else ThinkingPart
    assert [type(part) for part in response.parts] == [kind]


@pytest.mark.parametrize(("status", "replayed"), [(503, True), (429, True), (400, False)])
async def test_a_refused_request_goes_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, replayed: bool
) -> None:
    sent = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal sent
        sent += 1
        if sent == 1:
            return httpx2.Response(status, json={"error": {"message": "no available server"}})
        data = event({"content": "done"}) + event({}, usage={"prompt_tokens": 10, "completion_tokens": 1})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n")

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_client)
    monkeypatch.setattr(native_backend, "STREAM_BACKOFF", 0)

    async def emit(item: SessionEvent) -> None:
        pass

    # max_retries 0: the client's own retries would hide the first answer.
    settings: dict[str, JsonValue] = {
        "api_key": "test-refused",
        "api_base": "https://example.test/v1",
        "max_retries": 0,
    }
    async with backend(tmp_path, Agent("openai/local", model_settings=settings)) as session:
        if replayed:
            assert (await session.prompt("hello", emit)).output == "done"
        else:
            with pytest.raises(wire.StatusError):
                await session.prompt("hello", emit)
    assert sent == (2 if replayed else 1)


@pytest.mark.parametrize("name", ["openai/local", "anthropic/claude-test", "google/gemini-test", "responses/gpt-test"])
async def test_a_cancelled_stream_closes_its_transport(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Cancelling a turn mid-response closes the response body on every wire."""
    first = {
        "openai/local": event({"content": "partial"}),
        "anthropic/claude-test": sse(
            [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "partial"}}]
        ),
        "google/gemini-test": b'data: {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}\n\n',
        "responses/gpt-test": sse([{"type": "response.output_text.delta", "output_index": 0, "delta": "partial"}]),
    }[name]

    class WaitingStream(httpx2.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield first
            await anyio.sleep_forever()

        async def aclose(self) -> None:
            self.closed = True

    source = WaitingStream()

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
            ),
            **kwargs,
        )

    monkeypatch.setattr(wire, "client", make_client)
    model = OpenAIResponsesModel(name) if name.startswith("responses/") else model_for(name, {"api_base": "x"})
    try:
        info = ModelInfo(settings={"api_key": "test-cancel", "api_base": "https://example.test/v1"})
        with anyio.fail_after(2):
            with anyio.CancelScope() as scope:
                stream = model.stream([ModelRequest(parts=[UserPromptPart("go")])], info)
                try:
                    async for delta in stream:
                        if delta.part:
                            scope.cancel()
                finally:
                    with anyio.CancelScope(shield=True):
                        await stream.aclose()
        assert source.closed
    finally:
        await model.aclose()


async def test_backend_shutdown_closes_cached_clients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[httpx2.AsyncClient] = []

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=event({"content": "done"}) + b"data: [DONE]\n\n",
                )
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    def load_target(target: str) -> Agent:
        return Agent("openai/local", model_settings={"api_key": "test-shutdown", "api_base": "https://example.test/v1"})

    monkeypatch.setattr(wire, "client", make_client)
    monkeypatch.setattr(native_backend, "_session_catalog", Catalog)
    monkeypatch.setattr(native_backend, "load_target", load_target)

    async def emit(item: SessionEvent) -> None:
        pass

    spec = NativeSpec(cwd=str(tmp_path), target="local:agent", instruction_providers=[], aid_tools=False)
    async with native_backend.open_native(spec, anyio.Path(tmp_path)) as session:
        assert (await session.prompt("go", emit)).output == "done"
        assert clients and not clients[0].is_closed
    assert all(client.is_closed for client in clients)


async def test_open_native_reports_resume_only_for_stored_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh session seeds its system prompt but has no stored conversation: it started, it did not resume."""
    monkeypatch.setattr(native_backend, "_session_catalog", Catalog)

    def load_target(target: str) -> Agent:
        return Agent("openai/local", instructions="You are a test agent.")

    monkeypatch.setattr(native_backend, "load_target", load_target)
    spec = NativeSpec(cwd=str(tmp_path), target="local:agent", instruction_providers=[], aid_tools=False)

    async with native_backend.open_native(spec, anyio.Path(tmp_path)) as fresh:
        assert fresh.started().resumed is False

    (tmp_path / "history.json").write_bytes(
        ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart("earlier")])])
    )
    async with native_backend.open_native(spec, anyio.Path(tmp_path)) as resumed:
        assert resumed.started().resumed is True


async def test_a_seeded_session_starts_from_the_seed_until_it_has_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed is taken byte for byte, its instructions too, so a fork's requests share its parent's prefix."""
    monkeypatch.setattr(native_backend, "_session_catalog", Catalog)

    def load_target(target: str) -> Agent:
        return Agent("openai/local", instructions="Composed instructions.")

    monkeypatch.setattr(native_backend, "load_target", load_target)
    seed = tmp_path / "seed.json"
    seed.write_bytes(
        ModelMessagesTypeAdapter.dump_json(
            [ModelRequest(parts=[SystemPromptPart("Parent instructions.")]), ModelRequest(parts=[UserPromptPart("hi")])]
        )
    )
    state = tmp_path / "fork"
    state.mkdir()
    spec = NativeSpec(
        cwd=str(tmp_path), target="local:agent", instruction_providers=[], aid_tools=False, seed_history=str(seed)
    )

    async with native_backend.open_native(spec, anyio.Path(state)) as forked:
        assert forked.started().resumed is False
    assert (state / "history.json").read_bytes() == seed.read_bytes()

    own = ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart("its own")])])
    (state / "history.json").write_bytes(own)
    async with native_backend.open_native(spec, anyio.Path(state)) as resumed:
        assert resumed.started().resumed is True
    assert (state / "history.json").read_bytes() == own


def test_signed_thinking_and_tool_identity_survive_anthropic_translation() -> None:
    messages = [
        ModelResponse(
            parts=[
                ThinkingPart("thought", signature="signed"),
                TextPart("answer"),
                ToolCallPart("read", {"path": "a"}, "c1"),
            ]
        )
    ]
    system, wired = anthropic_messages(messages)
    assert system == []
    assert dumped(wired[0].content) == [
        {"type": "thinking", "thinking": "thought", "signature": "signed"},
        {"type": "text", "text": "answer"},
        {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "a"}},
    ]


def test_legacy_history_preserves_provider_fields() -> None:
    history = b'[{"kind":"response","parts":[{"part_kind":"thinking","content":"thought","signature":"signed","provider_details":{"opaque":"keep"}}],"provider_response_id":"r1"}]'
    loaded = ModelMessagesTypeAdapter.validate_json(history)
    assert json.loads(ModelMessagesTypeAdapter.dump_json(loaded))[0]["provider_response_id"] == "r1"
    assert json.loads(ModelMessagesTypeAdapter.dump_json(loaded))[0]["parts"][0]["provider_details"] == {
        "opaque": "keep"
    }


async def test_anthropic_stream_preserves_cache_usage_and_separate_signed_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3,
                },
            },
        },
    ]
    for index, content, signature in [(0, "first", "sig1"), (1, "second", "sig2")]:
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": content},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                },
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.extend(
        [
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "redacted_thinking", "data": "opaque"},
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
            {"type": "message_stop"},
        ]
    )
    body = "".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in events).encode()
    serve_wire(monkeypatch, lambda request: httpx2.Response(200, content=body))
    response = ModelResponse(parts=[])
    indexes: dict[int, int] = {}
    usages: list[RequestUsage] = []
    truncated = False
    model = AnthropicModel("anthropic/claude-test")
    try:
        async for delta in model.stream(
            [ModelRequest(parts=[UserPromptPart("think")])],
            ModelInfo(settings={"api_key": "test-anthropic", "api_base": "https://example.test"}),
        ):
            accumulate(response, indexes, delta)
            truncated = truncated or delta.truncated
            if delta.usage:
                usages.append(delta.usage)
        assert truncated
        assert [(p.content, p.signature) for p in response.parts if isinstance(p, ThinkingPart)][:2] == [
            ("first", "sig1"),
            ("second", "sig2"),
        ]
        assert (usages[0].input_tokens, usages[0].cache_read_tokens, usages[0].cache_write_tokens) == (5, 7, 3)
        assert usages[-1].output_tokens == 6
        _, wired = anthropic_messages([response])
        assert dumped(wired[0].content) == [
            {"type": "thinking", "thinking": "first", "signature": "sig1"},
            {"type": "thinking", "thinking": "second", "signature": "sig2"},
            {"type": "redacted_thinking", "data": "opaque"},
        ]
    finally:
        await model.aclose()


@pytest.mark.parametrize("failure", [False, True])
async def test_anthropic_stream_closes_clients_on_cancellation_and_failure(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    class InterruptedStream(httpx2.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            messages: list[dict[str, Any]] = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-test",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
            ]
            yield "".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in messages).encode()
            if failure:
                raise httpx2.ReadError("stream broke")
            await anyio.sleep_forever()

        async def aclose(self) -> None:
            self.closed = True

    source = InterruptedStream()
    serve_wire(monkeypatch, lambda request: httpx2.Response(200, stream=source))

    model = AnthropicModel("anthropic/claude-test")

    async def consume() -> None:
        stream = model.stream(
            [ModelRequest([UserPromptPart("go")])],
            ModelInfo(settings={"api_key": "test-cleanup", "api_base": "https://example.test"}),
        )
        with anyio.CancelScope() as scope:
            try:
                async for delta in stream:
                    if delta.part and not failure:
                        scope.cancel()
            finally:
                with anyio.CancelScope(shield=True):
                    await stream.aclose()

    try:
        with anyio.fail_after(2):
            if failure:
                with pytest.raises(StreamBroken) as raised:
                    await consume()
                assert isinstance(raised.value.__cause__, httpx2.ReadError)
            else:
                await consume()
        assert source.closed
    finally:
        await model.aclose()


def test_wire_skips_an_empty_assistant_message() -> None:
    """Histories stored before failed requests stopped leaving one must still send."""
    messages = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[]),
        ModelRequest(parts=[UserPromptPart("again")]),
    ]
    assert [item["role"] for item in wire_messages(messages)] == ["user", "user"]
    assert [item.role for item in anthropic_messages(messages)[1]] == ["user"]


async def test_model_names_pick_the_sdk_and_credentials_fail_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(model_for("anthropic/claude-x"), AnthropicModel)
    assert isinstance(model_for("groq/llama"), OpenAIChatModel)
    # A bare name is OpenAI's; on OpenAI's own server that is the Responses API (#70).
    assert isinstance(model_for("gpt-x", {"api_base": "https://example.test/v1"}), OpenAIChatModel)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    for name, code, says in [
        ("groq/llama", "missing_credentials", "GROQ_API_KEY"),
        ("anthropic/claude-x", "missing_credentials", "ANTHROPIC_API_KEY"),
        ("nowhere/model", "unknown_provider", "api_base"),
    ]:
        model = model_for(name)
        with pytest.raises(AidError) as caught:
            await anext(model.stream(go, ModelInfo()))
        assert (caught.value.code, says in caught.value.message) == (code, True)
    custom = OpenAIChatModel("nowhere/model")
    assert custom._key({"api_base": "https://example.test/v1", "api_key": "k"}) is not None  # pyright: ignore[reportPrivateUsage]
    local = OpenAIChatModel("ollama/llama")
    assert local._key({}).api_key == "local"  # pyright: ignore[reportPrivateUsage]
    await custom.aclose()
    await local.aclose()


@pytest.mark.parametrize(
    "model", [OpenAIChatModel("groq/llama"), AnthropicModel("anthropic/c"), GoogleModel("google/g")]
)
@pytest.mark.parametrize("settings", [{"api_key": 7}, {"api_key": "k", "max_retries": "many"}])
async def test_a_mistyped_client_setting_fails_before_the_sdk(
    model: OpenAIChatModel | AnthropicModel | GoogleModel, settings: dict[str, JsonValue]
) -> None:
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    with pytest.raises(AidError) as caught:
        await anext(model.stream(go, ModelInfo(settings=settings)))
    assert caught.value.code == "bad_model_settings"
    assert model.model_name in caught.value.message


def serve_wire(monkeypatch: pytest.MonkeyPatch, handle: Callable[[httpx2.Request], httpx2.Response]) -> None:
    """aid's own clients (`aid.wire`) talk to `handle` instead of the server."""

    def client(timeout: float) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), timeout=timeout)

    monkeypatch.setattr(wire, "client", client)


def dumped(blocks: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [block.model_dump(exclude_none=True) for block in blocks]


def sse(items: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in items).encode()


async def test_anthropic_request_is_native_and_passes_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """System on top, one user message for a response's tool results, beta headers through, caching on, and a
    tool that streams no arguments still reads as an empty object."""
    seen: list[httpx2.Request] = []
    body = sse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-test",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t9", "name": "now", "input": {}},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    serve_wire(monkeypatch, handle)

    async def now() -> str:
        return "noon"

    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart("be brief"), UserPromptPart("time?")]),
        ModelResponse(parts=[ToolCallPart("now", "{}", "t1"), ToolCallPart("now", '{"x":', "t2")]),
        ModelRequest(
            parts=[
                ToolReturnPart("now", "noon", "t1"),
                RetryPromptPart("bad arguments", tool_name="now", tool_call_id="t2"),
            ]
        ),
    ]
    model = AnthropicModel("anthropic/claude-test")
    settings: dict[str, JsonValue] = {
        "api_key": "k",
        "api_base": "https://example.test",
        "extra_headers": {"anthropic-beta": "b1"},
    }
    try:
        response = await model.request(history, ModelInfo(tools=[Tool(now)], settings=settings))
    finally:
        await model.aclose()
    assert response.parts == [ToolCallPart("now", "{}", "t9")]
    request = seen[0]
    assert str(request.url) == "https://example.test/v1/messages"
    assert (request.headers["anthropic-beta"], request.headers["x-api-key"]) == ("b1", "k")
    assert request.headers["anthropic-version"] == "2023-06-01"
    sent = json.loads(request.content)
    assert sent["system"] == [{"type": "text", "text": "be brief"}]
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["max_tokens"] == ANTHROPIC_MAX_TOKENS
    assert sent["model"] == "claude-test"
    assert sent["tools"][0]["name"] == "now" and "input_schema" in sent["tools"][0]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][1]["content"][1] == {"type": "tool_use", "id": "t2", "name": "now", "input": {}}
    results = sent["messages"][2]["content"]
    assert [(r["type"], r["tool_use_id"], r.get("is_error")) for r in results] == [
        ("tool_result", "t1", None),
        ("tool_result", "t2", True),
    ]


async def test_provider_defaults_go_under_the_session_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    titles: list[str | None] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        titles.append(request.headers.get("X-Title"))
        body = event({"content": "ok"}) + b"data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_client)
    provider = Provider("https://p.test/v1", "P_KEY", defaults={"max_tokens": 64}, headers={"X-Title": "aid"})
    monkeypatch.setenv("P_KEY", "k")
    model = OpenAIChatModel("p/model", provider)
    go: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("go")])]
    try:
        await model.request(go, ModelInfo())
        await model.request(go, ModelInfo(settings={"max_tokens": 5}))
    finally:
        await model.aclose()
    assert [body["max_tokens"] for body in sent] == [64, 5]
    assert [body["model"] for body in sent] == ["model", "model"]
    assert titles == ["aid", "aid"]


async def test_openai_compatible_stream_keeps_reasoning_and_cache_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # DeepSeek's field, then vLLM's and OpenRouter's: both are one stream's thinking.
    data = event({"reasoning_content": "hm"}) + event({"reasoning": "m"}) + event({"content": "ok"})
    data += event({}, finish="length")
    data += event(
        {},
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 6},
        },
    )

    def make_client(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(
                    200, headers={"content-type": "text/event-stream"}, content=data + b"data: [DONE]\n\n"
                )
            ),
            **kwargs,
        )

    monkeypatch.setattr(wire, "client", make_client)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    model = OpenAIChatModel("groq/llama")
    usages: list[RequestUsage] = []
    response = ModelResponse(parts=[])
    indexes: dict[int, int] = {}
    truncated = False
    try:
        async for delta in model.stream([ModelRequest(parts=[UserPromptPart("go")])], ModelInfo()):
            accumulate(response, indexes, delta)
            truncated = truncated or delta.truncated
            if delta.usage:
                usages.append(delta.usage)
    finally:
        await model.aclose()
    assert truncated
    assert response.parts == [ThinkingPart("hmm"), TextPart("ok")]
    assert (usages[0].input_tokens, usages[0].cache_read_tokens) == (10, 6)


async def test_a_stream_closes_its_events_when_it_stops_early(monkeypatch: pytest.MonkeyPatch) -> None:
    """`[DONE]` ends the read before the body does: the events generator is closed then, not by the collector
    (#67)."""
    closed: list[bool] = []
    events = wire.events

    async def watched(response: httpx2.Response) -> AsyncGenerator[str]:
        try:
            async for data in events(response):
                yield data
        finally:
            closed.append(True)

    body = event({"content": "ok"}) + b"data: [DONE]\n\n" + event({"content": "after"})
    serve_wire(
        monkeypatch,
        lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body),
    )
    monkeypatch.setattr(wire, "events", watched)
    model = OpenAIChatModel("groq/llama")
    try:
        stream = model.stream([ModelRequest(parts=[UserPromptPart("go")])], ModelInfo(settings={"api_key": "k"}))
        parts = [delta.part async for delta in stream if delta.part is not None]
        assert closed == [True]
    finally:
        await model.aclose()
    assert parts == [TextPart("ok")]


@pytest.mark.parametrize("name", ["openai/local", "anthropic/claude-test"])
async def test_a_refused_key_is_named_and_not_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A 401 needs a person to replace the key: one request, then `bad_credentials`, never `model_unavailable`."""
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(401, json={"error": {"type": "authentication_error", "message": "invalid key"}})

    def make_openai(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(wire, "client", make_openai)
    serve_wire(monkeypatch, handle)

    async def emit(item: SessionEvent) -> None:
        pass

    agent = Agent(name, model_settings={"api_key": "refused", "api_base": "https://example.test/v1"})
    async with backend(tmp_path, agent) as session:
        with pytest.raises(AidError) as raised:
            await session.prompt("hello", emit)
    assert raised.value.code == "bad_credentials"
    assert len(sent) == 1


def test_anthropic_events_parse_as_the_sdk_parses_them() -> None:
    """Differential: aid's event models against the Anthropic SDK's, on the same event JSON (#65)."""
    from anthropic.types import RawMessageStreamEvent  # noqa: PLC0415 -- a test dependency only
    from pydantic import TypeAdapter  # noqa: PLC0415

    theirs_adapter = TypeAdapter[RawMessageStreamEvent](RawMessageStreamEvent)
    items: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 0, "cache_read_input_tokens": 7},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "s"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 6}},
    ]
    for item in items:
        data = json.dumps(item)
        ours, theirs = anthropic_api.event(data), theirs_adapter.validate_json(data)
        assert ours.type == theirs.type
        match ours:
            case anthropic_api.MessageStart():
                assert theirs.type == "message_start"
                assert ours.message.model == theirs.message.model
                assert ours.message.usage.model_dump() == {
                    key: getattr(theirs.message.usage, key) for key in ours.message.usage.model_dump()
                }
            case anthropic_api.BlockStart():
                assert theirs.type == "content_block_start"
                assert ours.content_block.model_dump(exclude={"type"}) == theirs.content_block.model_dump(
                    include=set(ours.content_block.model_dump(exclude={"type"}))
                )
            case anthropic_api.BlockDelta():
                assert theirs.type == "content_block_delta"
                assert ours.delta.model_dump() == theirs.delta.model_dump(include=set(ours.delta.model_dump()))
            case anthropic_api.MessageDeltaEvent():
                assert theirs.type == "message_delta"
                assert ours.delta.stop_reason == theirs.delta.stop_reason
                assert ours.usage.output_tokens == theirs.usage.output_tokens
            case _:
                assert isinstance(ours, anthropic_api.BlockStop)


def test_anthropic_output_schema_matches_the_sdk() -> None:
    """Differential: structured output's schema as the SDK's `transform_schema` makes it. Without closed objects the
    API answers 400 (measured 2026-11-13)."""
    from typing import Literal  # noqa: PLC0415

    from anthropic import transform_schema  # noqa: PLC0415 -- a test dependency only
    from pydantic import BaseModel, Field, TypeAdapter  # noqa: PLC0415

    class Finding(BaseModel):
        """One finding."""

        line: int = Field(ge=1)
        kind: Literal["bug", "style"]
        note: str | None = Field(default=None, max_length=80)

    class Report(BaseModel):
        findings: list[Finding] = Field(min_length=1)
        when: str = Field(json_schema_extra={"format": "date"})
        tags: list[str] = Field(min_length=3)
        verdict: Literal["approve"] | Literal["reject"]

    schema = TypeAdapter(Report).json_schema()
    assert anthropic_api.output_schema(schema) == transform_schema(schema)


def test_chat_chunks_and_embeddings_parse_as_the_sdk_parses_them() -> None:
    """Differential: aid's Chat Completions models against the OpenAI SDK's, on the same JSON (#65)."""
    from openai.types import CreateEmbeddingResponse  # noqa: PLC0415 -- a test dependency only
    from openai.types.chat import ChatCompletionChunk  # noqa: PLC0415

    chunks = [
        {"reasoning_content": "hm"},
        {"content": "ok", "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "add", "arguments": '{"a":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
    ]
    for delta in chunks:
        data = event(delta, finish="tool_calls").decode().removeprefix("data: ").strip()
        ours, theirs = chat_api.Chunk.model_validate_json(data), ChatCompletionChunk.model_validate_json(data)
        assert ours.model == theirs.model
        [choice], [expected] = ours.choices, theirs.choices
        assert (choice.finish_reason, choice.delta.content) == (expected.finish_reason, expected.delta.content)
        assert (choice.delta.model_extra or {}) == (expected.delta.model_extra or {})
        calls = [
            (c.index, c.id, c.function and (c.function.name, c.function.arguments))
            for c in choice.delta.tool_calls or []
        ]
        expected_calls = [
            (c.index, c.id, c.function and (c.function.name, c.function.arguments))
            for c in expected.delta.tool_calls or []
        ]
        assert calls == expected_calls
    usage = event(
        {},
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 6},
        },
    )
    data = usage.decode().removeprefix("data: ").strip()
    ours_usage, theirs_usage = (
        chat_api.Chunk.model_validate_json(data).usage,
        ChatCompletionChunk.model_validate_json(data).usage,
    )
    assert ours_usage is not None and theirs_usage is not None
    assert ours_usage.prompt_tokens_details is not None and theirs_usage.prompt_tokens_details is not None
    assert (ours_usage.prompt_tokens, ours_usage.completion_tokens, ours_usage.prompt_tokens_details.cached_tokens) == (
        theirs_usage.prompt_tokens,
        theirs_usage.completion_tokens,
        theirs_usage.prompt_tokens_details.cached_tokens,
    )
    embedded = json.dumps(
        {
            "object": "list",
            "model": "m",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, -1.0]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3, "cost": 1e-7},
        }
    )
    ours_embedded, theirs_embedded = (
        chat_api.Embeddings.model_validate_json(embedded),
        CreateEmbeddingResponse.model_validate_json(embedded),
    )
    assert [(e.index, e.embedding) for e in ours_embedded.data] == [
        (e.index, e.embedding) for e in theirs_embedded.data
    ]
    assert ours_embedded.usage is not None
    assert ours_embedded.usage.prompt_tokens == theirs_embedded.usage.prompt_tokens
    assert (ours_embedded.usage.model_extra or {}).get("cost") == (theirs_embedded.usage.model_extra or {}).get("cost")
