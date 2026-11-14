"""Images in a native session's requests (#66): the store, and each protocol's request mapping."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import httpx2
import pytest

from aid.agent import Agent
from aid.attachments import (
    MAX_BYTES,
    MAX_PDF_BYTES,
    MAX_PROMPT_BYTES,
    AttachmentStore,
    ToolAttachment,
    decoded,
    file_name,
    refs,
)
from aid.backends.acp import AcpBackend
from aid.backends.claude_sdk import ClaudeSdkBackend
from aid.backends.claude_tty import ClaudeTtyBackend
from aid.backends.permissions import PermissionWaits
from aid.coding import CODING, Coding, read
from aid.messages import (
    AttachmentPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid.model import (
    AnthropicModel,
    GoogleModel,
    ModelInfo,
    OpenAIResponsesModel,
    Provider,
    StreamDelta,
    anthropic_messages,
    google_contents,
    responses_input,
    wire_messages,
)
from aid.prices import ENV_PRICES
from aid.protocol import AidError, AttachmentRef, Prompt, PromptEntry, ToolCall
from aid.spec import PermissionMode
from tests.models import TestModel
from tests.test_google_model import chunk
from tests.test_model import serve_wire, sse
from tests.test_native_backend import backend, stored

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from pydantic import JsonValue

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


def pdf(text: str = "The secret word is PINEAPPLE.") -> bytes:
    """A one-page PDF with `text` in Helvetica: what the PDF probes and live tests read (#72)."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_a_pdf_is_an_attachment_with_its_own_limit(tmp_path: Path) -> None:
    """A PDF goes up to `MAX_PDF_BYTES`, past the image limit; anything else stays refused."""
    store = AttachmentStore(tmp_path)
    assert store.put(pdf()).media_type == "application/pdf"
    big = b"%PDF-1.4\n" + bytes(MAX_BYTES)
    assert store.put(big).media_type == "application/pdf"
    with pytest.raises(AidError) as refused:
        store.put(b"%PDF-1.4\n" + bytes(MAX_PDF_BYTES))
    assert refused.value.code == "bad_attachment" and "pdf" in refused.value.message
    assert file_name(store.put(pdf())).endswith(".pdf")


def test_the_store_keeps_an_image_once_by_its_hash(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path)
    first = store.put(png())
    assert store.put(png()) == first
    assert (first.media_type, first.tool_call_id) == ("image/png", None)
    assert [p.name for p in (tmp_path / "attachments").iterdir()] == [first.sha256]
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
    assert AttachmentStore(tmp_path).put(data).media_type == kind


@pytest.mark.parametrize("data", [b"not an image", b"\x89PNG\r\n\x1a\n" + bytes(MAX_BYTES)])
def test_the_store_refuses_what_no_model_takes(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(AidError) as refused:
        AttachmentStore(tmp_path).put(data)
    assert refused.value.code == "bad_attachment"
    assert not (tmp_path / "attachments").exists()


async def test_a_session_from_before_the_rename_still_loads(tmp_path: Path) -> None:
    """#72 renamed images to attachments; what older sessions stored still reads: the history's `image` parts (kept as
    stored, so the resent prefix keeps its bytes), the transcript's `images`, the `images/` dir and its upload table."""
    data = png()
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / "images" / "uploads").mkdir(parents=True)
    (tmp_path / "images" / sha).write_bytes(data)
    (tmp_path / "images" / "uploads" / "scope.json").write_text(json.dumps({sha: {"id": "file_old"}}))
    stored_part = {"media_type": "image/png", "sha256": sha, "tool_call_id": None, "part_kind": "image"}
    history = json.dumps([{"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "see"}, stored_part]}])
    [message] = ModelMessagesTypeAdapter.validate_json(history)
    part = message.parts[1]
    assert isinstance(part, AttachmentPart) and part.part_kind == "image"
    assert json.loads(ModelMessagesTypeAdapter.dump_json([message]))[0]["parts"][1] == stored_part
    store = AttachmentStore(tmp_path)
    assert store.read(part) == data
    assert (await store.uploads("scope"))[sha].id == "file_old"
    entry = PromptEntry.model_validate(
        {"type": "prompt", "text": "see", "images": [{"media_type": "image/png", "sha256": sha}]}
    )
    call = ToolCall.model_validate({"tool_call_id": "c", "images": [{"media_type": "image/png", "sha256": sha}]})
    assert entry.attachments == call.attachments == [AttachmentRef(media_type="image/png", sha256=sha)]
    assert Prompt.model_validate({"session": "s", "text": "t", "images": ["QUFB"]}).attachments == ["QUFB"]
    assert AttachmentStore(tmp_path).put(data).part_kind == "attachment"


def test_a_prompts_images_arrive_as_base64() -> None:
    sent = base64.b64encode(png()).decode()
    assert decoded([sent]) == [png()]
    (ref,) = refs([sent])
    assert ref.media_type == "image/png"


@pytest.mark.parametrize(
    "attachments",
    [
        ["not base64!"],
        [base64.b64encode(b"text").decode()],
        [base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(MAX_BYTES - 8)).decode()] * (MAX_PROMPT_BYTES // MAX_BYTES + 1),
    ],
    ids=["not-base64", "not-an-image", "too-much"],
)
def test_a_prompts_images_are_checked(attachments: list[str]) -> None:
    with pytest.raises(AidError) as refused:
        decoded(attachments)
    assert refused.value.code == "bad_attachment"


def test_a_missing_or_changed_image_is_named(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path)
    part = store.put(png())
    (tmp_path / "attachments" / part.sha256).write_bytes(png((0, 0, 255)))
    with pytest.raises(AidError) as changed:
        store.read(part)
    with pytest.raises(AidError) as missing:
        store.read(AttachmentPart("image/png", "0" * 64))
    assert (changed.value.code, missing.value.code) == ("attachment_missing", "attachment_missing")


async def test_a_fork_takes_its_parents_images(tmp_path: Path) -> None:
    parent = AttachmentStore(tmp_path / "parent")
    part = parent.put(png(), tool_call_id="c1")
    fork = AttachmentStore(tmp_path / "fork")
    await fork.take(parent, [ModelRequest([UserPromptPart("see"), part])])
    assert fork.read(part) == png()


async def test_a_request_without_the_image_bytes_fails_by_name() -> None:
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("see"), AttachmentPart("image/png", "0" * 64)])]
    for mapping in (wire_messages, anthropic_messages, google_contents):
        with pytest.raises(AidError) as missing:
            mapping(history)
        assert missing.value.code == "attachment_missing"
    assert await ModelInfo().encoded(history) == {}


@pytest.fixture
def history(tmp_path: Path) -> tuple[list[ModelMessage], dict[str, str]]:
    """A prompt with an image, then a tool whose answer carries one."""
    store = AttachmentStore(tmp_path)
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
    wired = wire_messages(messages, attachments=encoded)
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
    """A prompt's image is a user part; a tool's goes inside its own function response."""
    messages, encoded = history
    red, blue = encoded.values()
    _, contents = google_contents(messages, encoded)
    sent = [json.loads(c.model_dump_json(exclude_none=True)) for c in contents]
    assert sent[0]["parts"] == [{"text": "what colour?"}, {"inlineData": {"mimeType": "image/png", "data": red}}]
    first, second = (p["functionResponse"] for p in sent[2]["parts"])
    assert first["parts"] == [{"inlineData": {"mimeType": "image/png", "data": blue}}]
    assert "parts" not in second


@pytest.fixture
def documents(tmp_path: Path) -> tuple[list[ModelMessage], dict[str, str]]:
    """A prompt with a PDF, then a tool whose answer carries one."""
    store = AttachmentStore(tmp_path)
    sent, read_back = store.put(pdf()), store.put(pdf("Another page."), tool_call_id="c1")
    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart("the secret word?"), sent]),
        ModelResponse([ToolCallPart("read", "{}", "c1")]),
        ModelRequest([ToolReturnPart("read", "doc.pdf", "c1"), read_back]),
    ]
    return messages, {p.sha256: base64.b64encode(store.read(p)).decode() for p in (sent, read_back)}


def test_each_wire_sends_a_pdf_as_its_api_names_it(documents: tuple[list[ModelMessage], dict[str, str]]) -> None:
    """Measured shapes (#72): Chat Completions a `file` part (a tool's after the tool messages), Anthropic a `document`
    block (a tool's inside its result), Gemini the media type as it is, Responses an `input_file`."""
    messages, encoded = documents
    (first, data), (second, tool_data) = encoded.items()
    url, tool_url = f"data:application/pdf;base64,{data}", f"data:application/pdf;base64,{tool_data}"
    chat = wire_messages(messages, attachments=encoded)
    assert chat[0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "the secret word?"},
            {"type": "file", "file": {"filename": f"{first}.pdf", "file_data": url}},
        ],
    }
    assert chat[-1] == {
        "role": "user",
        "content": [{"type": "file", "file": {"filename": f"{second}.pdf", "file_data": tool_url}}],
    }
    _, wired = anthropic_messages(messages, encoded)
    sent = [m.model_dump(exclude_none=True) for m in wired]
    assert sent[0]["content"][1] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }
    assert sent[2]["content"][0]["content"][1]["type"] == "document"
    _, contents = google_contents(messages, encoded)
    google = [json.loads(c.model_dump_json(exclude_none=True)) for c in contents]
    assert google[0]["parts"][1] == {"inlineData": {"mimeType": "application/pdf", "data": data}}
    assert google[2]["parts"][0]["functionResponse"]["parts"] == [
        {"inlineData": {"mimeType": "application/pdf", "data": tool_data}}
    ]
    items = responses_input(messages, encoded, {second: "file-9"})
    assert items[0] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "the secret word?"},
            {"type": "input_file", "filename": f"{first}.pdf", "file_data": url},
        ],
    }
    assert items[-1] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": [{"type": "input_text", "text": "doc.pdf"}, {"type": "input_file", "file_id": "file-9"}],
    }


ANSWER: Final = sse(
    [
        {
            "type": "message_start",
            "message": {"model": "claude-test", "usage": {"input_tokens": 1, "output_tokens": 0}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "red"}},
        {"type": "message_stop"},
    ]
)


class FilesServer:
    """Anthropic's Messages and Files APIs (#69): numbered uploads; a deleted id answers 404 as measured."""

    def __init__(self, upload_status: int = 200) -> None:
        self.upload_status = upload_status
        self.uploads: list[httpx2.Request] = []
        self.sent: list[dict[str, str]] = []
        self.deleted: set[str] = set()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/files":
            self.uploads.append(request)
            if self.upload_status != 200:
                return httpx2.Response(self.upload_status, json={"error": {"message": "no files here"}})
            return httpx2.Response(200, json={"type": "file", "id": f"file_{len(self.uploads)}"})
        source: dict[str, str] = json.loads(request.content)["messages"][0]["content"][1]["source"]
        self.sent.append(source)
        if (named := source.get("file_id")) in self.deleted:
            error = {"type": "not_found_error", "message": f"File `{named}` not found."}
            return httpx2.Response(404, json={"type": "error", "error": error})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=ANSWER)


async def ask_twice(model: AnthropicModel, store: AttachmentStore, server: FilesServer) -> None:
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("what colour?"), store.put(png())])]
    try:
        await model.request(history, ModelInfo(settings={"api_key": "k", "max_retries": 0}, attachments=store))
        server.deleted.add("file_1")
        await model.request(history, ModelInfo(settings={"api_key": "k", "max_retries": 0}, attachments=store))
    finally:
        await model.aclose()


async def test_anthropic_sends_an_image_by_file_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One upload per image and key; a deleted file is uploaded again and the request sent again."""
    serve_wire(monkeypatch, server := FilesServer())
    await ask_twice(AnthropicModel("anthropic/claude-test"), AttachmentStore(tmp_path), server)
    assert server.sent == [
        {"type": "file", "file_id": "file_1"},
        {"type": "file", "file_id": "file_1"},
        {"type": "file", "file_id": "file_2"},
    ]
    assert len(server.uploads) == 2
    assert server.uploads[0].headers["x-api-key"] == "k"
    assert png() in server.uploads[0].content


async def test_an_image_goes_as_bytes_where_files_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed upload costs nothing but the bytes: the request carries them, and the next one tries again."""
    serve_wire(monkeypatch, server := FilesServer(upload_status=500))
    await ask_twice(AnthropicModel("anthropic/claude-test"), AttachmentStore(tmp_path), server)
    assert [base64.b64decode(s["data"]) for s in server.sent] == [png(), png()]
    assert len(server.uploads) == 2


class OpenAIFiles:
    """OpenAI's Responses and Files APIs (#70): numbered uploads; a deleted id answers 404 as measured."""

    def __init__(self) -> None:
        self.uploads: list[httpx2.Request] = []
        self.sent: list[dict[str, str]] = []
        self.deleted: set[str] = set()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/files":
            self.uploads.append(request)
            return httpx2.Response(200, json={"object": "file", "id": f"file-{len(self.uploads)}"})
        image: dict[str, str] = json.loads(request.content)["input"][0]["content"][1]
        self.sent.append(image)
        if (named := image.get("file_id")) in self.deleted:
            error = {"message": f"Files [{named}] were not found", "type": "invalid_request_error"}
            return httpx2.Response(404, json={"error": error})
        done = {"type": "response.completed", "response": {"status": "completed"}}
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse([done]))


async def test_openai_sends_an_image_by_file_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Responses API takes an image by file id where Chat Completions does not (#70): one upload, a deleted file
    uploaded again and the request sent again."""
    serve_wire(monkeypatch, server := OpenAIFiles())
    store = AttachmentStore(tmp_path)
    model = OpenAIResponsesModel("openai/gpt-test")
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("what colour?"), store.put(png())])]
    info = ModelInfo(settings={"api_key": "k", "max_retries": 0}, attachments=store)
    try:
        await model.request(history, info)
        server.deleted.add("file-1")
        await model.request(history, info)
    finally:
        await model.aclose()
    assert server.sent == [
        {"type": "input_image", "detail": "auto", "file_id": "file-1"},
        {"type": "input_image", "detail": "auto", "file_id": "file-1"},
        {"type": "input_image", "detail": "auto", "file_id": "file-2"},
    ]
    assert len(server.uploads) == 2
    assert server.uploads[0].headers["authorization"] == "Bearer k"
    assert b'name="purpose"\r\n\r\nvision' in server.uploads[0].content and png() in server.uploads[0].content
    # OpenAI takes an image's format from the file name's extension (measured: none → 400).
    assert f'filename="{hashlib.sha256(png()).hexdigest()}.png"'.encode() in server.uploads[0].content


class GeminiFiles:
    """Gemini's Files API (resumable upload) and streamGenerateContent (#69): a deleted URI answers 403 as measured."""

    def __init__(self, expires: datetime) -> None:
        self.expires = expires
        self.uploads: list[httpx2.Request] = []
        self.sent: list[dict[str, dict[str, str]]] = []
        self.deleted: set[str] = set()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/upload/v1beta/files":
            return httpx2.Response(200, headers={"x-goog-upload-url": "https://upload.example/session"})
        if request.url.host == "upload.example":
            self.uploads.append(request)
            name = f"files/f{len(self.uploads)}"
            stored = {
                "name": name,
                "uri": f"https://generativelanguage.googleapis.com/v1beta/{name}",
                "state": "ACTIVE",
                "expirationTime": self.expires.isoformat(),
            }
            return httpx2.Response(200, json={"file": stored})
        image: dict[str, dict[str, str]] = json.loads(request.content)["contents"][0]["parts"][1]
        self.sent.append(image)
        if (uri := image.get("fileData", {}).get("fileUri", "")) in self.deleted:
            message = f"You do not have permission to access the File {uri.rsplit('/', 1)[1]} or it may not exist."
            return httpx2.Response(
                403, json={"error": {"code": 403, "message": message, "status": "PERMISSION_DENIED"}}
            )
        body = chunk([{"text": "red"}], finish="STOP", usage={"promptTokenCount": 1, "candidatesTokenCount": 1})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def ask_gemini(store: AttachmentStore, server: GeminiFiles, rounds: int = 2) -> None:
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("what colour?"), store.put(png())])]
    model = GoogleModel("google/gemini-test")
    try:
        for _ in range(rounds):
            await model.request(history, ModelInfo(settings={"api_key": "k", "max_retries": 0}, attachments=store))
            server.deleted.add("https://generativelanguage.googleapis.com/v1beta/files/f1")
    finally:
        await model.aclose()


async def test_gemini_sends_an_image_by_file_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploaded once; a file Gemini no longer holds (403) is uploaded again and the request sent again."""
    serve_wire(monkeypatch, server := GeminiFiles(datetime.now(UTC) + timedelta(hours=48)))
    await ask_gemini(AttachmentStore(tmp_path), server)
    assert [s["fileData"]["fileUri"].rsplit("/", 1)[1] for s in server.sent] == ["f1", "f1", "f2"]
    assert (len(server.uploads), server.uploads[0].content) == (2, png())


async def test_a_gemini_file_near_its_expiry_goes_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serve_wire(monkeypatch, server := GeminiFiles(datetime.now(UTC) + timedelta(minutes=30)))
    await ask_gemini(AttachmentStore(tmp_path), server, rounds=1)
    await ask_gemini(AttachmentStore(tmp_path), server, rounds=1)
    assert [s["fileData"]["fileUri"].rsplit("/", 1)[1] for s in server.sent] == ["f1", "f2"]


async def test_another_messages_server_gets_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serve_wire(monkeypatch, server := FilesServer())
    model = AnthropicModel("zen/claude-test", Provider("https://zen.example/", "ZEN_KEY"))
    await ask_twice(model, AttachmentStore(tmp_path), server)
    assert (len(server.uploads), [s["type"] for s in server.sent]) == (0, ["base64", "base64"])


class Looking(TestModel):
    """Keeps what each request carried: its messages and their attachments' bytes."""

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
    store = AttachmentStore(tmp_path)
    async with backend(tmp_path, Agent(model), attachments=store) as session:
        assert (await session.prompt("what colour?", emit, [png()])).output == "red"
    ((messages, encoded),) = model.seen
    prompt = messages[-1].parts
    assert [p.part_kind for p in prompt[:2]] == ["user-prompt", "attachment"]
    image = prompt[1]
    assert isinstance(image, AttachmentPart)
    assert base64.b64decode(encoded[image.sha256]) == png()
    kept = await stored(tmp_path)
    assert image in kept[0].parts
    assert store.read(image) == png()


async def test_a_bad_image_fails_before_the_turn(tmp_path: Path) -> None:
    model = Looking()
    async with backend(tmp_path, Agent(model), attachments=AttachmentStore(tmp_path)) as session:
        with pytest.raises(AidError) as refused:
            await session.prompt("what colour?", emit, [b"text"])
    assert (refused.value.code, model.seen) == ("bad_attachment", [])


@pytest.mark.parametrize(
    ("takes", "code"), [(["text"], "attachments_unsupported"), (["text", "image"], None), (None, None)]
)
async def test_models_dev_input_decides_whether_a_model_sees_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, takes: list[str] | None, code: str | None
) -> None:
    """A model models.dev lists without `image` input is refused before any request; one it does not describe is
    asked."""
    entry: dict[str, JsonValue] = {"id": "local"}
    if takes is not None:
        entry["modalities"] = {"input": [*takes], "output": ["text"]}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"openai": {"models": {"local": entry}}}))
    monkeypatch.setenv(ENV_PRICES, str(catalog))
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(400, json={"error": {"message": "stop here"}})

    serve_wire(monkeypatch, handle)
    agent = Agent(
        "openai/local", model_settings={"api_key": "k", "api_base": "https://example.test/v1", "max_retries": 0}
    )
    async with backend(tmp_path, agent, attachments=AttachmentStore(tmp_path)) as session:
        with pytest.raises(Exception) as refused:  # the asked model's 400 stops the turn
            await session.prompt("what colour?", emit, [png()])
    if code is None:
        assert len(seen) == 1
    else:
        assert isinstance(refused.value, AidError)
        assert (refused.value.code, seen) == (code, [])
        assert "models.dev input: text" in refused.value.message


@pytest.mark.parametrize(
    ("name", "settings", "takes", "refused"),
    [
        ("openai/local", {"api_base": "https://example.test/v1"}, ["text", "image"], True),
        ("openai/local", {"api_base": "https://example.test/v1"}, ["text", "image", "pdf"], False),
        # OpenAI reads PDFs on its image models; models.dev does not list them (gpt-5-nano, measured).
        ("openai/local", {}, ["text", "image"], False),
        ("openai/local", {}, ["text"], True),
    ],
)
async def test_models_dev_decides_whether_a_model_reads_pdfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    settings: dict[str, JsonValue],
    takes: list[str],
    refused: bool,
) -> None:
    catalog = tmp_path / "catalog.json"
    entry = {"id": "local", "modalities": {"input": takes, "output": ["text"]}}
    catalog.write_text(json.dumps({"openai": {"models": {"local": entry}}}))
    monkeypatch.setenv(ENV_PRICES, str(catalog))
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if not request.url.path.endswith("/files"):  # OpenAI's own route uploads first
            seen.append(request)
        return httpx2.Response(400, json={"error": {"message": "stop here"}})

    serve_wire(monkeypatch, handle)
    agent = Agent(name, model_settings={"api_key": "k", "max_retries": 0, **settings})
    async with backend(tmp_path, agent, attachments=AttachmentStore(tmp_path)) as session:
        with pytest.raises(Exception) as failed:  # the asked model's 400 stops the turn
            await session.prompt("the secret word?", emit, [pdf()])
    if refused:
        assert isinstance(failed.value, AidError)
        assert (failed.value.code, seen) == ("attachments_unsupported", [])
        assert "takes no pdf" in failed.value.message
    else:
        assert len(seen) == 1


async def test_read_hands_a_pdf_to_the_model(tmp_path: Path) -> None:
    """A PDF can be plain ASCII, so `read` checks the bytes before it decodes them as text."""
    (tmp_path / "doc.pdf").write_bytes(pdf())
    coding = Coding(
        cwd=tmp_path,
        store=tmp_path / "runs",
        outputs=tmp_path / "outputs",
        mode=PermissionMode.DENY,
        timeout=10,
        waits=PermissionWaits(),
    )
    token = CODING.set(coding)
    try:
        result = await read("doc.pdf")
    finally:
        CODING.reset(token)
    assert isinstance(result, ToolAttachment) and result.data == pdf()
    assert result.text == f"doc.pdf: application/pdf, {len(pdf())} bytes"


class Reading(Looking):
    """Reads `pic.png` first, then answers."""

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        self.seen.append((messages, dict(await info.encoded(messages))))
        if len(self.seen) == 1:
            yield StreamDelta(0, ToolCallPart("read", {"path": "pic.png"}, "c1"))
        else:
            yield StreamDelta(0, TextPart("blue"))


@pytest.mark.parametrize("keeps", [True, False])
async def test_read_hands_an_image_file_to_the_model(tmp_path: Path, keeps: bool) -> None:
    """The image follows its tool answer in the same request, tied to the call, and the call's event names it for
    the transcript; without a store the model gets the note alone."""
    (tmp_path / "pic.png").write_bytes(png((0, 0, 255)))
    model = Reading()
    events: list[SessionEvent] = []

    async def record(event: SessionEvent) -> None:
        events.append(event)

    async with backend(
        tmp_path, Agent(model, tools=[read]), attachments=AttachmentStore(tmp_path) if keeps else None
    ) as session:
        await session.prompt("what colour is pic.png?", record)
    returned = [e for e in events if isinstance(e, ToolCall) and e.status == "completed"]
    messages, encoded = model.seen[1]
    answer = messages[-1].parts
    assert isinstance(answer[0], ToolReturnPart)
    if keeps:
        assert answer[0].content == f"pic.png: image/png, {len(png((0, 0, 255)))} bytes"
        image = answer[1]
        assert isinstance(image, AttachmentPart)
        assert (image.tool_call_id, base64.b64decode(encoded[image.sha256])) == ("c1", png((0, 0, 255)))
        assert image in (await stored(tmp_path))[-2].parts
        assert [e.attachments for e in returned] == [[AttachmentRef(media_type="image/png", sha256=image.sha256)]]
    else:
        assert str(answer[0].content).endswith("; this model takes no images")
        assert (len(answer), encoded) == (1, {})
        assert [e.attachments for e in returned] == [[]]


@pytest.mark.parametrize("engine", [AcpBackend, ClaudeSdkBackend, ClaudeTtyBackend])
async def test_other_engines_refuse_images(engine: type[Backend]) -> None:
    session = object.__new__(engine)
    with pytest.raises(AidError) as refused:
        await session.prompt("see", emit, [png()])
    assert refused.value.code == "attachments_unsupported"
