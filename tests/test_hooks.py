from __future__ import annotations

import pytest
from pydantic import ValidationError

from aid.spec import AcpSpec, ClaudeTtySpec, Hook, LiteLMSpec, match_hook


def test_valid_hook() -> None:
    hook = Hook(tool="python", match=r"rm\s+-rf", message="no destructive deletes")
    assert hook.tool == "python"
    assert hook.match == r"rm\s+-rf"
    assert hook.message == "no destructive deletes"


def test_invalid_regex_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid hook match regex"):
        Hook(tool="python", match="(unclosed", message="blocked")


def test_hook_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        Hook(tool="*", match="x", message="m", extra="no")  # type: ignore[call-arg]


def test_hook_is_frozen() -> None:
    hook = Hook(tool="*", match="x", message="m")
    with pytest.raises(ValidationError):
        hook.tool = "python"  # type: ignore[misc]


def test_specs_carry_hooks_default_empty() -> None:
    acp = AcpSpec(cwd="/", command=["a"])
    pyd = LiteLMSpec(cwd="/", agent="myagent")
    tty = ClaudeTtySpec(cwd="/")
    assert acp.hooks == []
    assert pyd.hooks == []
    assert tty.hooks == []
    hook = Hook(tool="*", match="secret", message="blocked")
    assert acp.model_copy(update={"hooks": [hook]}).hooks == [hook]
    assert pyd.model_copy(update={"hooks": [hook]}).hooks == [hook]
    assert tty.model_copy(update={"hooks": [hook]}).hooks == [hook]


def test_first_match_wins() -> None:
    first = Hook(tool="python", match="evil", message="first")
    second = Hook(tool="python", match="evil", message="second")
    assert match_hook([first, second], "python", "this is evil") is first


def test_star_matches_any_tool() -> None:
    hook = Hook(tool="*", match="evil", message="blocked")
    assert match_hook([hook], "python", "evil here") is hook
    assert match_hook([hook], "shell", "evil here") is hook
    assert match_hook([hook], "apply_edits", "evil here") is hook


def test_tool_must_fit() -> None:
    hook = Hook(tool="python", match="evil", message="blocked")
    assert match_hook([hook], "shell", "evil here") is None
    assert match_hook([hook], "python", "evil here") is hook


def test_no_match_returns_none() -> None:
    assert match_hook([], "python", "anything") is None
    hook = Hook(tool="python", match="evil", message="blocked")
    assert match_hook([hook], "python", "innocent text") is None


def test_matcher_is_total() -> None:
    hook = Hook(tool="python", match="evil", message="blocked")
    assert match_hook([hook], None, "evil") is None
    assert match_hook([hook], "python", None) is None
    assert match_hook(None, "python", "evil") is None  # type: ignore[arg-type]
    broken = Hook.model_construct(tool="python", match="(unclosed", message="blocked")
    assert match_hook([broken], "python", "evil") is None
    assert match_hook(["not-a-hook"], "python", "evil") is None  # type: ignore[list-item]
