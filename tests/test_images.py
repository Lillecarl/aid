"""Images in a native session's requests (#66): the store, and each protocol's request mapping."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import TYPE_CHECKING

import httpx2
import pytest

from aid.images import MAX_BYTES, ImageStore
from aid.messages import ImagePart, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from aid.model import AnthropicModel, ModelInfo, anthropic_messages, google_contents, wire_messages
from aid.protocol import AidError
from tests.test_model import serve_wire, sse

if TYPE_CHECKING:
    from pathlib import Path

    from aid.messages import ModelMessage


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


def test_a_missing_or_changed_image_is_named(tmp_path: Path) -> None:
    store = ImageStore(tmp_path)
    part = store.put(png())
    (tmp_path / "images" / part.sha256).write_bytes(png((0, 0, 255)))
    with pytest.raises(AidError) as changed:
        store.read(part)
    with pytest.raises(AidError) as missing:
        store.read(ImagePart("image/png", "0" * 64))
    assert (changed.value.code, missing.value.code) == ("image_missing", "image_missing")


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
