"""Images in a native session's requests (#66): the store, and each protocol's request mapping."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import TYPE_CHECKING

import httpx2
import pytest

from aid.agent import Agent
from aid.backends.acp import AcpBackend
from aid.backends.claude_sdk import ClaudeSdkBackend
from aid.backends.claude_tty import ClaudeTtyBackend
from aid.images import MAX_BYTES, MAX_PROMPT_BYTES, ImageStore, decoded, refs
from aid.messages import (
    ImagePart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import AnthropicModel, ModelInfo, StreamDelta, anthropic_messages, google_contents, wire_messages
from aid.protocol import AidError
from tests.models import TestModel
from tests.test_model import serve_wire, sse
from tests.test_native_backend import backend, stored

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from aid.backends.base import Backend
    from aid.messages import ModelMessage
    from aid.protocol import SessionEvent


def png(rgb: tuple[int, int, int] = (255, 0, 0), size: int = 8) -> bytes:
    """A square of one colour."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b"")
    )


def test_the_store_keeps_an_image_once_by_its_hash(tmp_path: Path) -> None:
    store = ImageStore(tmp_path)
    first = store.put(png())
    assert store.put(png()) == first
    assert (first.media_type, first.tool_call_id) == ("image/png", None)
    assert [p.name for p in (tmp_path / "images").iterdir()] == [first.sha256]
    assert store.read(first) == png()


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
        (b"GIF89a....", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
def test_the_store_names_each_format(tmp_path: Path, data: bytes, kind: str) -> None:
    assert ImageStore(tmp_path).put(data).media_type == kind


@pytest.mark.parametrize("data", [b"not an image", b"\x89PNG\r\n\x1a\n" + bytes(MAX_BYTES)])
def test_the_store_refuses_what_no_model_takes(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(AidError) as refused:
        ImageStore(tmp_path).put(data)
    assert refused.value.code == "bad_image"
    assert not (tmp_path / "images").exists()


def test_a_prompts_images_arrive_as_base64() -> None:
    sent = base64.b64encode(png()).decode()
    assert decoded([sent]) == [png()]
    (ref,) = refs([sent])
    assert ref.media_type == "image/png"


@pytest.mark.parametrize(
    "images",
    [
        ["not base64!"],
        [base64.b64encode(b"text").decode()],
        [base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(MAX_BYTES - 8)).decode()] * (MAX_PROMPT_BYTES // MAX_BYTES + 1),
    ],
    ids=["not-base64", "not-an-image", "too-much"],
)
def test_a_prompts_images_are_checked(images: list[str]) -> None:
    with pytest.raises(AidError) as refused:
        decoded(images)
    assert refused.value.code == "bad_image"


def test_a_missing_or_changed_image_is_named(tmp_path: Path) -> None:
    store = ImageStore(tmp_path)
    part = store.put(png())
    (tmp_path / "images" / part.sha256).write_bytes(png((0, 0, 255)))
    with pytest.raises(AidError) as changed:
        store.read(part)
    with pytest.raises(AidError) as missing:
        store.read(ImagePart("image/png", "0" * 64))
    assert (changed.value.code, missing.value.code) == ("image_missing", "image_missing")


async def test_a_fork_takes_its_parents_images(tmp_path: Path) -> None:
    parent = ImageStore(tmp_path / "parent")
    part = parent.put(png(), tool_call_id="c1")
    fork = ImageStore(tmp_path / "fork")
    await fork.take(parent, [ModelRequest([UserPromptPart("see"), part])])
    assert fork.read(part) == png()


async def test_a_request_without_the_image_bytes_fails_by_name() -> None:
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("see"), ImagePart("image/png", "0" * 64)])]
    for mapping in (wire_messages, anthropic_messages, google_contents):
        with pytest.raises(AidError) as missing:
            mapping(history)
        assert missing.value.code == "image_missing"
    assert await ModelInfo().encoded(history) == {}


@pytest.fixture
def history(tmp_path: Path) -> tuple[list[ModelMessage], dict[str, str]]:
    """A prompt with an image, then a tool whose answer carries one."""
    store = ImageStore(tmp_path)
    red, blue = store.put(png()), store.put(png((0, 0, 255)), tool_call_id="c1")
    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart("what colour?"), red]),
        ModelResponse([ToolCallPart("look", "{}", "c1"), ToolCallPart("look", "{}", "c2")]),
        ModelRequest([ToolReturnPart("look", "a picture", "c1"), blue, ToolReturnPart("look", "none", "c2")]),
    ]
    encoded = {p.sha256: base64.b64encode(store.read(p)).decode() for p in (red, blue)}
    return messages, encoded


def test_chat_completions_put_images_in_user_content(history: tuple[list[ModelMessage], dict[str, str]]) -> None:
    """A tool message takes text only: the call's image follows the request's last tool message."""
    messages, encoded = history
    red, blue = (f"data:image/png;base64,{data}" for data in encoded.values())
    wired = wire_messages(messages, images=encoded)
    assert wired[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "what colour?"}, {"type": "image_url", "image_url": {"url": red}}],
    }
    assert [(w["role"], w.get("tool_call_id")) for w in wired[2:]] == [("tool", "c1"), ("tool", "c2"), ("user", None)]
    assert wired[4]["content"] == [{"type": "image_url", "image_url": {"url": blue}}]


def test_anthropic_puts_a_tools_image_in_its_result(history: tuple[list[ModelMessage], dict[str, str]]) -> None:
    messages, encoded = history
    red, blue = encoded.values()
    _, wired = anthropic_messages(messages, encoded)
    sent = [m.model_dump(exclude_none=True) for m in wired]
    assert sent[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": red},
    }
    first, second = sent[2]["content"]
    assert first["content"] == [
        {"type": "text", "text": "a picture"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": blue}},
    ]
    assert second["content"] == "none"


def test_gemini_sends_images_inline(history: tuple[list[ModelMessage], dict[str, str]]) -> None:
    messages, encoded = history
    red, blue = encoded.values()
    _, contents = google_contents(messages, encoded)
    sent = [json.loads(c.model_dump_json(exclude_none=True)) for c in contents]
    assert sent[0]["parts"] == [{"text": "what colour?"}, {"inlineData": {"mimeType": "image/png", "data": red}}]
    assert [list(p) for p in sent[2]["parts"]] == [["functionResponse"], ["inlineData"], ["functionResponse"]]
    assert sent[2]["parts"][1]["inlineData"]["data"] == blue


async def test_a_transport_reads_the_bytes_from_the_sessions_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx2.Request] = []
    body = sse(
        [
            {
                "type": "message_start",
                "message": {"model": "claude-test", "usage": {"input_tokens": 1, "output_tokens": 0}},
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "red"}},
            {"type": "message_stop"},
        ]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    serve_wire(monkeypatch, handle)
    store = ImageStore(tmp_path)
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("what colour?"), store.put(png())])]
    model = AnthropicModel("anthropic/claude-test")
    try:
        await model.request(history, ModelInfo(settings={"api_key": "k"}, images=store))
    finally:
        await model.aclose()
    image = json.loads(seen[0].content)["messages"][0]["content"][1]
    assert base64.b64decode(image["source"]["data"]) == png()


class Looking(TestModel):
    """Keeps what each request carried: its messages and their images' bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[list[ModelMessage], dict[str, str]]] = []

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        self.seen.append((messages, dict(await info.encoded(messages))))
        yield StreamDelta(0, TextPart("red"))


async def emit(event: SessionEvent) -> None:
    pass


async def test_a_native_prompt_sends_its_images_and_keeps_them(tmp_path: Path) -> None:
    model = Looking()
    store = ImageStore(tmp_path)
    async with backend(tmp_path, Agent(model), images=store) as session:
        assert (await session.prompt("what colour?", emit, [png()])).output == "red"
    ((messages, encoded),) = model.seen
    prompt = messages[-1].parts
    assert [p.part_kind for p in prompt[:2]] == ["user-prompt", "image"]
    image = prompt[1]
    assert isinstance(image, ImagePart)
    assert base64.b64decode(encoded[image.sha256]) == png()
    kept = await stored(tmp_path)
    assert image in kept[0].parts
    assert store.read(image) == png()


async def test_a_bad_image_fails_before_the_turn(tmp_path: Path) -> None:
    model = Looking()
    async with backend(tmp_path, Agent(model), images=ImageStore(tmp_path)) as session:
        with pytest.raises(AidError) as refused:
            await session.prompt("what colour?", emit, [b"text"])
    assert (refused.value.code, model.seen) == ("bad_image", [])


@pytest.mark.parametrize("engine", [AcpBackend, ClaudeSdkBackend, ClaudeTtyBackend])
async def test_other_engines_refuse_images(engine: type[Backend]) -> None:
    session = object.__new__(engine)
    with pytest.raises(AidError) as refused:
        await session.prompt("see", emit, [png()])
    assert refused.value.code == "images_unsupported"
