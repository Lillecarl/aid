from __future__ import annotations

import pytest

from aid.protocol import ToolCall
from plugins.supervision.loops import REPETITIONS, WINDOW, ToolLoopDetector


def complete(detector: ToolLoopDetector, number: int, title: str = "read", *, output: str = "unchanged") -> bool:
    detector.observe(ToolCall(tool_call_id=str(number), title=title, status="in_progress", input={"path": "app.py"}))
    return detector.observe(ToolCall(tool_call_id=str(number), status="completed", output=output)) is not None


def test_three_identical_calls_raise_one_suspicion() -> None:
    detector = ToolLoopDetector()
    assert [complete(detector, n) for n in range(12)] == [False, False, True] + [False] * 9


@pytest.mark.parametrize("period", [2, 3, 8])
def test_repeated_sequences_are_phase_independent(period: int) -> None:
    detector = ToolLoopDetector()
    alerts = [complete(detector, n, title=f"tool{n % period}") for n in range(period * (REPETITIONS + 2))]
    assert sum(alerts) == 1
    assert alerts[period * REPETITIONS - 1]


def test_changing_results_are_not_exact_repetition() -> None:
    detector = ToolLoopDetector()
    assert not any(complete(detector, n, output=f"progress {n}") for n in range(60))


def test_changing_arguments_are_not_exact_repetition() -> None:
    detector = ToolLoopDetector()
    for n in range(10):
        assert (
            detector.observe(
                ToolCall(tool_call_id=str(n), title="read", status="completed", input={"path": str(n)}, output="same")
            )
            is None
        )


def test_missing_output_does_not_count_as_repetition() -> None:
    detector = ToolLoopDetector()
    for n in range(10):
        assert detector.observe(ToolCall(tool_call_id=str(n), title="read", status="completed")) is None


def test_duplicate_completion_does_not_count_as_execution() -> None:
    detector = ToolLoopDetector()
    call = ToolCall(tool_call_id="one", title="read", status="completed", input={}, output="same")
    for _ in range(20):
        assert detector.observe(call) is None


def test_failed_calls_can_raise_a_suspicion() -> None:
    detector = ToolLoopDetector()
    evidence = None
    for n in range(3):
        evidence = detector.observe(
            ToolCall(tool_call_id=str(n), title="edit", status="failed", input={}, output="anchor missing")
        )
    assert evidence is not None
    assert "suspicion, not proof" in evidence.describe()


def test_missing_telemetry_breaks_a_sequence_instead_of_hiding_progress() -> None:
    detector = ToolLoopDetector()
    assert not complete(detector, 0)
    assert not complete(detector, 1)
    assert detector.observe(ToolCall(tool_call_id="unknown", title="write", status="completed")) is None
    assert not complete(detector, 2)
    assert not complete(detector, 3)
    assert complete(detector, 4)


def test_missing_arguments_do_not_count_as_identical_arguments() -> None:
    detector = ToolLoopDetector()
    for n in range(10):
        assert detector.observe(ToolCall(tool_call_id=str(n), title="read", status="completed", output="same")) is None


def test_quiet_window_rearms_an_old_pattern() -> None:
    detector = ToolLoopDetector()
    assert complete(detector, 0) is False
    assert complete(detector, 1) is False
    assert complete(detector, 2) is True
    for n in range(3, WINDOW + 3):
        assert not complete(detector, n, output=f"progress {n}")
    assert not complete(detector, WINDOW + 3)
    assert not complete(detector, WINDOW + 4)
    assert complete(detector, WINDOW + 5)
