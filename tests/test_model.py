from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic
import anyio
import httpx2
import litelm
import openai
import pytest

import aid.backends.litelm as native_backend
from aid.agent import Agent
from aid.agents import Catalog
from aid.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import LiteLMModel, ModelInfo, RequestUsage, accumulate, wire_messages
from aid.protocol import SessionEvent, Usage
from aid.spec import AgentSpecAdapter, LiteLMSpec
from tests.test_litelm_backend import backend

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


def event(delta: dict[str, Any], *, usage: dict[str, Any] | None = None) -> bytes:
    chunk: dict[str, Any] = {"id": "response", "model": "local", "choices": [{"index": 0, "delta": delta}]}
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n".encode()


async def test_real_litelm_transport_tool_loop_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    try:
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
        assert b"Session metadata" not in (tmp_path / "history.json").read_bytes()
    finally:
        await litelm.close_litelm_async_clients()


async def test_real_litelm_stream_cancellation_closes_transport(monkeypatch: pytest.MonkeyPatch) -> None:
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
    try:
        model = LiteLMModel("openai/local")
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
        await litelm.close_litelm_async_clients()


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

    spec = LiteLMSpec(cwd=str(tmp_path), target="local:agent", instruction_providers=[], aid_tools=False)
    async with native_backend.open_litelm(spec, anyio.Path(tmp_path)) as session:
        assert (await session.prompt("go", emit)).output == "done"
        assert clients and not clients[0].is_closed
    assert all(client.is_closed for client in clients)


async def test_open_litelm_reports_resume_only_for_stored_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh session seeds its system prompt but has no stored conversation: it started, it did not resume."""
    monkeypatch.setattr(native_backend, "_session_catalog", Catalog)

    def load_target(target: str) -> Agent:
        return Agent("openai/local", instructions="You are a test agent.")

    monkeypatch.setattr(native_backend, "load_target", load_target)
    spec = LiteLMSpec(cwd=str(tmp_path), target="local:agent", instruction_providers=[], aid_tools=False)

    async with native_backend.open_litelm(spec, anyio.Path(tmp_path)) as fresh:
        assert fresh.started().resumed is False

    (tmp_path / "history.json").write_bytes(
        ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart("earlier")])])
    )
    async with native_backend.open_litelm(spec, anyio.Path(tmp_path)) as resumed:
        assert resumed.started().resumed is True


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
    wire = wire_messages(messages, anthropic=True)
    assert wire[0]["content"][0] == {"type": "thinking", "thinking": "thought", "signature": "signed"}
    assert wire[0]["tool_calls"][0]["id"] == "c1"


def test_legacy_session_kind_and_history_preserve_provider_fields() -> None:
    spec = AgentSpecAdapter.validate_python({"kind": "pydantic-ai", "cwd": "/", "target": "agents:echo"})
    assert isinstance(spec, LiteLMSpec)
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
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
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
    try:
        model = LiteLMModel("anthropic/claude-test")
        async for delta in model.stream(
            [ModelRequest(parts=[UserPromptPart("think")])],
            ModelInfo(settings={"api_key": "test-anthropic", "api_base": "https://example.test"}),
        ):
            accumulate(response, indexes, delta)
            if delta.usage:
                usages.append(delta.usage)
        assert [(p.content, p.signature) for p in response.parts if isinstance(p, ThinkingPart)][:2] == [
            ("first", "sig1"),
            ("second", "sig2"),
        ]
        assert (usages[0].input_tokens, usages[0].cache_read_tokens, usages[0].cache_write_tokens) == (5, 7, 3)
        assert usages[-1].output_tokens == 6
        wire = wire_messages([response], anthropic=True)
        assert wire[0]["content"] == [
            {"type": "thinking", "thinking": "first", "signature": "sig1"},
            {"type": "thinking", "thinking": "second", "signature": "sig2"},
            {"type": "redacted_thinking", "data": "opaque"},
        ]
        assert all(client.is_closed() for client in clients)
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

    async def consume() -> None:
        model = LiteLMModel("anthropic/claude-test")
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
                with pytest.raises(litelm.APIConnectionError):
                    await consume()
            else:
                await consume()
        assert source.closed
        assert clients and all(client.is_closed() for client in clients)
    finally:
        for client in clients:
            await client.close()


@pytest.mark.parametrize("anthropic", [False, True])
def test_wire_skips_an_empty_assistant_message(anthropic: bool) -> None:
    """Histories stored before failed requests stopped leaving one must still send."""
    messages = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[]),
        ModelRequest(parts=[UserPromptPart("again")]),
    ]
    assert [item["role"] for item in wire_messages(messages, anthropic=anthropic)] == ["user", "user"]
