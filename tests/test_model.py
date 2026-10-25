from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic
import anyio
import httpx2
import openai
import pytest

import aid.backends.native as native_backend
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
    Provider,
    RequestUsage,
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
    from pathlib import Path

    from pydantic import JsonValue

pytestmark = pytest.mark.anyio


def event(delta: dict[str, Any], *, usage: dict[str, Any] | None = None, finish: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    chunk: dict[str, Any] = {"id": "response", "model": "local", "choices": [choice]}
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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)

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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
    events: list[SessionEvent] = []

    async def emit(item: SessionEvent) -> None:
        events.append(item)

    agent = Agent("openai/local", model_settings={"api_key": "test-gauge", "api_base": "https://example.test/v1"})
    async with backend(tmp_path, agent, desired_max=max_context) as session:
        await session.prompt("hello", emit)
    usage = next(item for item in events if isinstance(item, Usage))
    assert usage.context_size == gauge


async def test_openai_stream_cancellation_closes_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class WaitingStream(httpx2.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield event({"content": "partial"})
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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
    model = OpenAIChatModel("openai/local")
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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
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
    system, wire = anthropic_messages(messages)
    assert system == []
    assert wire[0]["content"] == [
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
    factory = anthropic.AsyncAnthropic
    clients: list[anthropic.AsyncAnthropic] = []

    def make_client(**kwargs: Any) -> anthropic.AsyncAnthropic:
        client = factory(
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
                )
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", make_client)
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
        _, wire = anthropic_messages([response])
        assert wire[0]["content"] == [
            {"type": "thinking", "thinking": "first", "signature": "sig1"},
            {"type": "thinking", "thinking": "second", "signature": "sig2"},
            {"type": "redacted_thinking", "data": "opaque"},
        ]
        await model.aclose()
        assert clients and all(client.is_closed() for client in clients)
    finally:
        for client in clients:
            await client.close()


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
    factory = anthropic.AsyncAnthropic
    clients: list[anthropic.AsyncAnthropic] = []

    def make_client(**kwargs: Any) -> anthropic.AsyncAnthropic:
        client = factory(
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
                )
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", make_client)

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
                with pytest.raises(httpx2.ReadError):
                    await consume()
            else:
                await consume()
        assert source.closed
        await model.aclose()
        assert clients and all(client.is_closed() for client in clients)
    finally:
        for client in clients:
            await client.close()


def test_wire_skips_an_empty_assistant_message() -> None:
    """Histories stored before failed requests stopped leaving one must still send."""
    messages = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[]),
        ModelRequest(parts=[UserPromptPart("again")]),
    ]
    assert [item["role"] for item in wire_messages(messages)] == ["user", "user"]
    assert [item["role"] for item in anthropic_messages(messages)[1]] == ["user"]


async def test_model_names_pick_the_sdk_and_credentials_fail_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(model_for("anthropic/claude-x"), AnthropicModel)
    assert isinstance(model_for("groq/llama"), OpenAIChatModel)
    assert isinstance(model_for("gpt-x"), OpenAIChatModel)
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
    assert custom._client({"api_base": "https://example.test/v1", "api_key": "k"}) is not None  # pyright: ignore[reportPrivateUsage]
    local = OpenAIChatModel("ollama/llama")
    assert local._client({}).api_key == "local"  # pyright: ignore[reportPrivateUsage]
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
    factory = anthropic.AsyncAnthropic

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    def make_client(**kwargs: Any) -> anthropic.AsyncAnthropic:
        return factory(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)), **kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", make_client)

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
    assert request.headers["anthropic-beta"] == "b1"
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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
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
    data = event({"reasoning_content": "hmm"}) + event({"content": "ok"}) + event({}, finish="length")
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

    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", make_client)
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
