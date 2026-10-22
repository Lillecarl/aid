"""Bounded, engine-independent loop signals. Evidence asks for supervision, never authorizes cancellation.

This first detector compares completed tool calls exactly. Semantic detectors can add signals later without
turning embeddings into either a dependency of the daemon or an automatic stop policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from aid.protocol import ToolCall

WINDOW: Final = 24
REPETITIONS: Final = 3
PENDING_CALLS: Final = 128


@dataclass(frozen=True)
class Step:
    call_id: str
    title: str
    fingerprint: bytes


@dataclass(frozen=True)
class LoopEvidence:
    period: int
    steps: tuple[Step, ...]

    def describe(self) -> str:
        sequence = " → ".join(step.title for step in self.steps[: self.period])
        ids = ", ".join(step.call_id for step in self.steps)
        return (
            f"{REPETITIONS} repeated cycles of {self.period} completed tool calls: {sequence}.\n"
            "Their arguments, statuses, and recorded outputs are identical across cycles. "
            "This is a suspicion, not proof of non-progress (polling and repetitive work can be legitimate).\n"
            f"Tool call ids for history inspection: {ids}"
        )


def repeated_suffix(steps: tuple[Step, ...]) -> LoopEvidence | None:
    """The shortest sequence repeated at least three times at the live end of the window."""
    for period in range(1, len(steps) // REPETITIONS + 1):
        tail = steps[-period * REPETITIONS :]
        fingerprints = tuple(step.fingerprint for step in tail)
        if fingerprints == fingerprints[:period] * REPETITIONS:
            return LoopEvidence(period=period, steps=tail)
    return None


class ToolLoopDetector:
    """One live turn's incremental calls, kept bounded. Report an episode once, not on every repeated call;
    the rotations of an A/B loop are the same episode. A full quiet window rearms the same pattern."""

    def __init__(self) -> None:
        self._pending: dict[str, ToolCall] = {}
        self._steps: deque[Step] = deque(maxlen=WINDOW)
        self._reported: tuple[bytes, ...] | None = None
        self._quiet = 0

    def observe(self, call: ToolCall) -> LoopEvidence | None:
        if any(step.call_id == call.tool_call_id for step in self._steps):
            return None  # Duplicate terminal updates never count as another execution.
        previous = self._pending.pop(call.tool_call_id, None)
        if previous is not None:
            call = previous.model_copy(
                update={key: value for key, value in call.model_dump().items() if value is not None and value != []}
            )
        if call.status not in {"completed", "failed"}:
            self._pending[call.tool_call_id] = call
            if len(self._pending) > PENDING_CALLS:
                self._pending.pop(next(iter(self._pending)))
            return None
        if not call.title or call.input is None or call.output is None:
            # An unobserved result/argument may be progress between two known calls. Do not splice the
            # known steps together and falsely describe them as a consecutive repeated sequence.
            self._steps.clear()
            self._quiet = 0
            return None
        body = json.dumps([call.title, call.input, call.status, call.output], sort_keys=True, ensure_ascii=False)
        self._steps.append(
            Step(call_id=call.tool_call_id, title=call.title, fingerprint=hashlib.sha256(body.encode()).digest())
        )
        evidence = repeated_suffix(tuple(self._steps))
        if evidence is None:
            self._quiet += 1
            if self._quiet >= WINDOW:
                self._reported = None
            return None
        self._quiet = 0
        pattern = tuple(step.fingerprint for step in evidence.steps[: evidence.period])
        key = min(pattern[offset:] + pattern[:offset] for offset in range(len(pattern)))
        if key == self._reported:
            return None
        self._reported = key
        return evidence
