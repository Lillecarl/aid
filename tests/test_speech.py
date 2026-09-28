"""Speech to text against a real model: AID_TEST_SPEECH_MODEL, which the dev shell and the Nix tests set."""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import numpy as np
import pytest

from aid.speech import Heard, Transcription, load

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from aid.speech import Recognizer

MODEL = Path(os.environ["AID_TEST_SPEECH_MODEL"]) if os.environ.get("AID_TEST_SPEECH_MODEL") else None
pytestmark = pytest.mark.skipif(MODEL is None, reason="AID_TEST_SPEECH_MODEL is not set")

# What the default model's test_wavs/0.wav says.
SAID = "Ask not what your country can do for you. Ask what you can do for your country"


def speech() -> tuple[int, NDArray[np.float32]]:
    assert MODEL is not None
    with wave.open(str(MODEL / "test_wavs" / "0.wav")) as f:
        rate, frames = f.getframerate(), f.readframes(f.getnframes())
    return rate, np.frombuffer(frames, dtype=np.int16).astype(np.float32) / np.float32(32768)


@pytest.fixture(scope="module")
def recognizer() -> Recognizer:
    assert MODEL is not None
    return load(MODEL)


def transcribe(recognizer: Recognizer) -> list[Heard]:
    rate, samples = speech()
    transcription = Transcription(recognizer)
    heard: list[Heard] = []
    step = rate // 10
    for start in range(0, len(samples), step):
        heard += transcription.feed(rate, samples[start : start + step])
    return heard + transcription.finish(rate)


def said(heard: list[Heard]) -> str:
    return " ".join(h.text for h in heard if h.final)


def test_speech_becomes_text(recognizer: Recognizer) -> None:
    heard = transcribe(recognizer)
    assert said(heard) == SAID
    assert [h for h in heard if not h.final], "no guess arrived before the end"
    assert heard[-1].final


@pytest.mark.anyio
async def test_two_speakers_share_one_recognizer(recognizer: Recognizer) -> None:
    results: list[str] = []

    async def one() -> None:
        results.append(said(await anyio.to_thread.run_sync(transcribe, recognizer)))

    async with anyio.create_task_group() as tg:
        for _ in range(2):
            tg.start_soon(one)
    assert results == [SAID, SAID]


def test_a_directory_without_a_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"encoder\.onnx"):
        load(tmp_path)
