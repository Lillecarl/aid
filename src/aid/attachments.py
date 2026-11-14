"""Images a session sends to its model (#66): content-addressed files in the session dir, named by `AttachmentPart`."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import anyio.to_thread
from pydantic import BaseModel, TypeAdapter, ValidationError

from aid.messages import AttachmentPart
from aid.protocol import AidError, AttachmentRef

if TYPE_CHECKING:
    from collections.abc import Container, Mapping, Sequence
    from pathlib import Path

    from aid.messages import ModelMessage

ATTACHMENTS_DIR: Final = "attachments"
LEGACY_DIR: Final = "images"
"""Where sessions before #72 kept them: read, never written."""


class Upload(BaseModel):
    """An image a provider holds (#69): its file id or URI, and when the provider deletes it (None: never)."""

    id: str
    expires: datetime | None = None


UPLOADS: Final = TypeAdapter(dict[str, Upload])
UPLOADS_DIR: Final = "uploads"
"""`attachments/uploads/<scope>.json`: sha256 → a provider's file id, per provider and key (`scope`)."""
MAX_BYTES: Final = 5 * 1024 * 1024
"""Anthropic's limit per image, the lowest of the three APIs."""
MAX_PROMPT_BYTES: Final = 20 * 1024 * 1024
"""One prompt's attachments together: what one protocol message may carry. Each request resends every image of the
history, so this does not keep a request under a provider's size limit."""

SIGNATURES: Final = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
"""The formats every API aid speaks takes. WebP's RIFF header needs its own check (`media_type`)."""


def media_type(data: bytes) -> str | None:
    """The image format by its magic bytes, None for anything else."""
    for magic, kind in SIGNATURES:
        if data.startswith(magic):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def checked(data: bytes) -> str:
    """The image's media type, or `bad_attachment` for what no model takes."""
    if len(data) > MAX_BYTES:
        raise AidError("bad_attachment", f"the image is {len(data)} bytes, over the limit of {MAX_BYTES}")
    if (kind := media_type(data)) is None:
        raise AidError("bad_attachment", "the image is not PNG, JPEG, GIF or WebP")
    return kind


@dataclass(frozen=True)
class ToolAttachment:
    """A tool's answer that is an image: the native loop stores `data` and sends it after `text`, or sends `text` alone
    where the model takes no attachments."""

    data: bytes
    text: str


def decoded(attachments: Sequence[str]) -> list[bytes]:
    """A prompt's base64 attachments as bytes, each checked, `bad_attachment` when one is not."""
    if (total := sum(len(image) // 4 * 3 for image in attachments)) > MAX_PROMPT_BYTES:
        raise AidError("bad_attachment", f"the attachments are {total} bytes, over the limit of {MAX_PROMPT_BYTES}")
    try:
        data = [base64.b64decode(image, validate=True) for image in attachments]
    except binascii.Error as error:
        raise AidError("bad_attachment", f"an image is not base64: {error}") from None
    for one in data:
        checked(one)
    return data


def refs(attachments: Sequence[str]) -> list[AttachmentRef]:
    """What the transcript records of a prompt's attachments."""
    return [AttachmentRef(media_type=checked(d), sha256=hashlib.sha256(d).hexdigest()) for d in decoded(attachments)]


class AttachmentStore:
    """`root/attachments/<sha256>`, written once. A part names bytes the store checks on read."""

    def __init__(self, root: Path) -> None:
        self._dir = root / ATTACHMENTS_DIR
        self._legacy = root / LEGACY_DIR

    def _path(self, sha256: str) -> Path:
        path = self._dir / sha256
        if not path.exists() and (legacy := self._legacy / sha256).exists():
            return legacy
        return path

    def put(self, data: bytes, tool_call_id: str | None = None) -> AttachmentPart:
        kind = checked(data)
        digest = hashlib.sha256(data).hexdigest()
        path = self._dir / digest
        if not path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f".{digest}.{os.getpid()}")
            partial.write_bytes(data)
            partial.replace(path)
        return AttachmentPart(kind, digest, tool_call_id)

    def read(self, part: AttachmentPart) -> bytes:
        try:
            data = self._path(part.sha256).read_bytes()
        except FileNotFoundError:
            raise AidError("attachment_missing", f"image {part.sha256} is not in {self._dir}") from None
        if hashlib.sha256(data).hexdigest() != part.sha256:
            raise AidError("attachment_missing", f"image {part.sha256} in {self._dir} does not match its name")
        return data

    def _encoded(self, parts: Sequence[AttachmentPart]) -> Encoded:
        return {part.sha256: base64.b64encode(self.read(part)).decode() for part in parts}

    async def encoded(self, messages: Sequence[ModelMessage], skip: Container[str] = ()) -> Encoded:
        """Every image of `messages` as base64 by sha256, read in a worker thread; `skip` names those sent by id."""
        parts = [p for m in messages for p in m.parts if isinstance(p, AttachmentPart) and p.sha256 not in skip]
        return await anyio.to_thread.run_sync(self._encoded, parts) if parts else {}

    async def bytes_of(self, part: AttachmentPart) -> bytes:
        return await anyio.to_thread.run_sync(self.read, part)

    def _uploads_file(self, scope: str) -> Path:
        return self._dir / UPLOADS_DIR / f"{scope}.json"

    def _uploads(self, scope: str) -> dict[str, Upload]:
        path = self._uploads_file(scope)
        if not path.exists():
            path = self._legacy / UPLOADS_DIR / path.name
        try:
            return UPLOADS.validate_json(path.read_bytes())
        except FileNotFoundError, ValidationError:
            return {}

    def _write_uploads(self, scope: str, ids: Mapping[str, Upload]) -> None:
        path = self._uploads_file(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{os.getpid()}")
        partial.write_bytes(UPLOADS.dump_json(dict(ids)))
        partial.replace(path)

    async def uploads(self, scope: str) -> dict[str, Upload]:
        """The file ids `scope` holds for this store's attachments: an unreadable table is an empty one."""
        return await anyio.to_thread.run_sync(self._uploads, scope)

    async def save_uploads(self, scope: str, ids: Mapping[str, Upload]) -> None:
        await anyio.to_thread.run_sync(self._write_uploads, scope, ids)

    def _take(self, other: AttachmentStore, parts: Sequence[AttachmentPart]) -> None:
        for part in parts:
            if not self._path(part.sha256).exists():
                self.put(other.read(part), part.tool_call_id)

    async def take(self, other: AttachmentStore, messages: Sequence[ModelMessage]) -> None:
        """Copy the attachments of `messages` from `other`: a seeded history (a fork) names its parent's."""
        parts = [p for m in messages for p in m.parts if isinstance(p, AttachmentPart)]
        if parts:
            await anyio.to_thread.run_sync(self._take, other, parts)

    async def add(self, data: bytes, tool_call_id: str | None = None) -> AttachmentPart:
        return await anyio.to_thread.run_sync(self.put, data, tool_call_id)


type Encoded = Mapping[str, str]
"""Images' base64 by sha256: what a transport's request mapping reads."""

NO_ATTACHMENTS: Final[Encoded] = MappingProxyType({})


def encoded(attachments: Encoded, part: AttachmentPart) -> str:
    try:
        return attachments[part.sha256]
    except KeyError:
        raise AidError("attachment_missing", f"image {part.sha256} has no bytes for this request") from None
