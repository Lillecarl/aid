"""Speech to text with sherpa-onnx, for aid-web: a streaming transducer that hears audio as it arrives.

A model directory holds `encoder.onnx`, `decoder.onnx`, `joiner.onnx` and `tokens.txt`, as the streaming
transducers of k2-fsa's `asr-models` release do. Measured with sherpa-onnx 1.13.3 and
`sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`: 1.3 s to load, 0.07 s to decode 3.8 s of speech on two
threads, cased and punctuated text, and `accept_waveform` resamples any input rate to the model's.

Everything here blocks; aid-web runs it in worker threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np
import sherpa_onnx  # pyright: ignore[reportMissingTypeStubs] -- untyped; `Recognizer` is the part aid uses

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class Stream(Protocol):
    def accept_waveform(self, sample_rate: int, waveform: NDArray[np.float32]) -> None: ...

    def input_finished(self) -> None: ...


class Recognizer(Protocol):
    def create_stream(self) -> Stream: ...

    def is_ready(self, s: Stream) -> bool: ...

    def decode_stream(self, s: Stream) -> None: ...

    def is_endpoint(self, s: Stream) -> bool: ...

    def get_result(self, s: Stream) -> str: ...

    def reset(self, s: Stream) -> None: ...


MODEL_FILES: Final = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
THREADS: Final = 2
# Silence fed after the last audio: a streaming model holds back its last words until it hears more.
TAIL_SECONDS: Final = 0.5


@dataclass(frozen=True)
class Heard:
    text: str
    final: bool
    """The end of an utterance: the text will not change. Otherwise it is a guess that the next audio may revise."""


def load(model: Path) -> Recognizer:
    if missing := [name for name in MODEL_FILES if not (model / name).is_file()]:
        raise FileNotFoundError(f"{model} is not a streaming transducer: it lacks {', '.join(missing)}")
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(  # pyright: ignore[reportUnknownMemberType]
        tokens=str(model / "tokens.txt"),
        encoder=str(model / "encoder.onnx"),
        decoder=str(model / "decoder.onnx"),
        joiner=str(model / "joiner.onnx"),
        num_threads=THREADS,
        enable_endpoint_detection=True,
    )
    return cast("Recognizer", recognizer)


class Transcription:
    """One speaker's audio, in order, turned into what was heard."""

    def __init__(self, recognizer: Recognizer) -> None:
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()
        self._partial = ""

    def feed(self, rate: int, samples: NDArray[np.float32]) -> list[Heard]:
        """Mono samples in [-1, 1] at `rate` Hz. Returns what changed: a new guess, or finished utterances."""
        self._stream.accept_waveform(rate, samples)
        return self._decode()

    def finish(self, rate: int) -> list[Heard]:
        """The speaker stopped. Returns the last words, as final."""
        self._stream.accept_waveform(rate, np.zeros(int(rate * TAIL_SECONDS), dtype=np.float32))
        self._stream.input_finished()
        heard = self._decode()
        text = self._recognizer.get_result(self._stream).strip()
        return [*heard, Heard(text, final=True)] if text else heard

    def _decode(self) -> list[Heard]:
        heard: list[Heard] = []
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
            if self._recognizer.is_endpoint(self._stream):
                if text := self._recognizer.get_result(self._stream).strip():
                    heard.append(Heard(text, final=True))
                self._recognizer.reset(self._stream)
                self._partial = ""
        text = self._recognizer.get_result(self._stream).strip()
        if text != self._partial:
            self._partial = text
            heard.append(Heard(text, final=False))
        return heard
