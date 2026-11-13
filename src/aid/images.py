"""Images a session sends to its model (#66): content-addressed files in the session dir, named by `ImagePart`."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import anyio.to_thread

from aid.messages import ImagePart
from aid.protocol import AidError, ImageRef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from aid.messages import ModelMessage

IMAGES_DIR: Final = "images"
MAX_BYTES: Final = 5 * 1024 * 1024
"""Anthropic's limit per image, the lowest of the three APIs."""
MAX_PROMPT_BYTES: Final = 20 * 1024 * 1024
"""One prompt's images together: what one protocol message may carry. Each request resends every image of the
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
    """The image's media type, or `bad_image` for what no model takes."""
    if len(data) > MAX_BYTES:
        raise AidError("bad_image", f"the image is {len(data)} bytes, over the limit of {MAX_BYTES}")
    if (kind := media_type(data)) is None:
        raise AidError("bad_image", "the image is not PNG, JPEG, GIF or WebP")
    return kind


def decoded(images: Sequence[str]) -> list[bytes]:
    """A prompt's base64 images as bytes, each checked, `bad_image` when one is not."""
    if (total := sum(len(image) // 4 * 3 for image in images)) > MAX_PROMPT_BYTES:
        raise AidError("bad_image", f"the images are {total} bytes, over the limit of {MAX_PROMPT_BYTES}")
    try:
        data = [base64.b64decode(image, validate=True) for image in images]
    except binascii.Error as error:
        raise AidError("bad_image", f"an image is not base64: {error}") from None
    for one in data:
        checked(one)
    return data


def refs(images: Sequence[str]) -> list[ImageRef]:
    """What the transcript records of a prompt's images."""
    return [ImageRef(media_type=checked(d), sha256=hashlib.sha256(d).hexdigest()) for d in decoded(images)]


class ImageStore:
    """`root/images/<sha256>`, written once. A part names bytes the store checks on read."""

    def __init__(self, root: Path) -> None:
        self._dir = root / IMAGES_DIR

    def put(self, data: bytes, tool_call_id: str | None = None) -> ImagePart:
        kind = checked(data)
        digest = hashlib.sha256(data).hexdigest()
        path = self._dir / digest
        if not path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f".{digest}.{os.getpid()}")
            partial.write_bytes(data)
            partial.replace(path)
        return ImagePart(kind, digest, tool_call_id)

    def read(self, part: ImagePart) -> bytes:
        try:
            data = (self._dir / part.sha256).read_bytes()
        except FileNotFoundError:
            raise AidError("image_missing", f"image {part.sha256} is not in {self._dir}") from None
        if hashlib.sha256(data).hexdigest() != part.sha256:
            raise AidError("image_missing", f"image {part.sha256} in {self._dir} does not match its name")
        return data

    def _encoded(self, parts: Sequence[ImagePart]) -> Encoded:
        return {part.sha256: base64.b64encode(self.read(part)).decode() for part in parts}

    async def encoded(self, messages: Sequence[ModelMessage]) -> Encoded:
        """Every image of `messages` as base64 by sha256, read in a worker thread."""
        parts = [p for m in messages for p in m.parts if isinstance(p, ImagePart)]
        return await anyio.to_thread.run_sync(self._encoded, parts) if parts else {}

    def _take(self, other: ImageStore, parts: Sequence[ImagePart]) -> None:
        for part in parts:
            if not (self._dir / part.sha256).exists():
                self.put(other.read(part), part.tool_call_id)

    async def take(self, other: ImageStore, messages: Sequence[ModelMessage]) -> None:
        """Copy the images of `messages` from `other`: a seeded history (a fork) names its parent's."""
        parts = [p for m in messages for p in m.parts if isinstance(p, ImagePart)]
        if parts:
            await anyio.to_thread.run_sync(self._take, other, parts)

    async def add(self, data: bytes, tool_call_id: str | None = None) -> ImagePart:
        return await anyio.to_thread.run_sync(self.put, data, tool_call_id)


type Encoded = Mapping[str, str]
"""Images' base64 by sha256: what a transport's request mapping reads."""

NO_IMAGES: Final[Encoded] = MappingProxyType({})


def encoded(images: Encoded, part: ImagePart) -> str:
    try:
        return images[part.sha256]
    except KeyError:
        raise AidError("image_missing", f"image {part.sha256} has no bytes for this request") from None
