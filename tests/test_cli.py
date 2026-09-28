from __future__ import annotations

import pytest

from aid.cli import _split_agent_command  # pyright: ignore[reportPrivateUsage] -- the parsing rule under test


@pytest.mark.parametrize(
    ("argv", "head", "tail"),
    [
        (
            ["new-acp", "x", "--cwd", "/tmp", "--", "agent", "--flag", "--"],
            ["new-acp", "x", "--cwd", "/tmp"],
            ["agent", "--flag", "--"],
        ),
        (["new-acp", "x"], ["new-acp", "x"], []),
        (["prompt", "x", "--", "-text"], ["prompt", "x", "--", "-text"], []),
    ],
)
def test_split_agent_command(argv: list[str], head: list[str], tail: list[str]) -> None:
    assert _split_agent_command(argv) == (head, tail)
