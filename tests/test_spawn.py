from __future__ import annotations

import pytest

from aid.protocol import AidError
from aid.spawn import MAX_CHILDREN, MAX_DEPTH, SpawnRequest, child_name, child_spec
from aid.spec import AcpSpec, NativeSpec, PermissionMode, Sandbox

OFFERED = ["ls", "read", "outline", "python", "python:read", "python:write", "send_message", "spawn"]


def parent(**fields: object) -> NativeSpec:
    return NativeSpec.model_validate(
        {
            "cwd": "/work",
            "agent": "coder",
            "model": "deepseek/deepseek-flash",
            "model_settings": {"max_tokens": 900},
            "max_context": 50_000,
            "permission": PermissionMode.ASK,
            "tool_permissions": {"python": PermissionMode.SUPERVISOR},
            "sandbox": Sandbox(network=False),
            "supervisor": "boss",
            "tags": {"team": "a"},
            **fields,
        }
    )


def request(**fields: object) -> SpawnRequest:
    return SpawnRequest.model_validate({"label": "reader", "instructions": "Read.", "tools": ["ls|read"], **fields})


def test_a_child_gets_the_names_its_patterns_match_as_literals() -> None:
    child = child_spec("p", parent(), OFFERED, request(tools=["ls|read", "python.*"]), depth=0, children=0)
    assert child.tools == ["ls", "python", "python:read", "python:write", "read"]
    later = ["ls", "read", "read_more"]
    assert [name for name in later if child.offers(name)] == ["ls", "read"]


def test_a_child_inherits_what_bounds_its_parent() -> None:
    child = child_spec("p", parent(), OFFERED, request(), depth=0, children=0)
    assert (child.agent, child.instructions, child.runs, child.parent) == (None, "Read.", "instructions", "p")
    assert (child.permission, child.tool_permissions) == (PermissionMode.ASK, {"python": PermissionMode.SUPERVISOR})
    assert (child.sandbox, child.supervisor, child.cwd) == (Sandbox(network=False), "boss", "/work")
    assert (child.model, child.model_settings, child.max_context) == (
        "deepseek/deepseek-flash",
        {"max_tokens": 900},
        50_000,
    )
    assert child.tags == {}


def test_a_child_with_no_tools_is_offered_none() -> None:
    child = child_spec("p", parent(), OFFERED, request(tools=[]), depth=0, children=0)
    assert child.tools == [] and not child.offers("ls")


def test_a_pattern_matching_none_of_the_parents_tools_is_refused() -> None:
    with pytest.raises(AidError, match="shell"):
        child_spec("p", parent(), OFFERED, request(tools=["read", "shell"]), depth=0, children=0)


@pytest.mark.parametrize(
    ("spawn_models", "model", "allowed"),
    [
        (None, None, True),
        (None, "deepseek/deepseek-flash", True),
        (None, "vllm/glm-5.3-flash", False),
        (["anthropic/claude-haiku-.*"], "anthropic/claude-haiku-5-5", True),
        (["anthropic/claude-haiku-.*"], "anthropic/claude-opus-5-5", False),
    ],
)
def test_a_child_runs_on_its_parents_model_or_one_a_person_allowed(
    spawn_models: list[str] | None, model: str | None, allowed: bool
) -> None:
    spec = parent(spawn_models=spawn_models)
    if not allowed:
        with pytest.raises(AidError, match="may not run on"):
            child_spec("p", spec, OFFERED, request(model=model), depth=0, children=0)
        return
    child = child_spec("p", spec, OFFERED, request(model=model), depth=0, children=0)
    assert child.model == (model or spec.model)
    assert child.spawn_models == spawn_models


def test_settings_follow_only_within_the_provider() -> None:
    spec = parent(spawn_models=["anthropic/.*"])
    child = child_spec("p", spec, OFFERED, request(model="anthropic/claude-haiku-5-5"), depth=0, children=0)
    assert (child.model_settings, child.max_context) == ({}, None)


def test_a_parent_on_its_agents_model_names_the_childs() -> None:
    with pytest.raises(AidError, match="name the child's model"):
        child_spec("p", parent(model=None), OFFERED, request(), depth=0, children=0)
    spec = parent(model=None, spawn_models=["deepseek/.*"])
    assert child_spec("p", spec, OFFERED, request(model="deepseek/x"), depth=0, children=0).model == "deepseek/x"


def test_depth_and_children_are_capped() -> None:
    child_spec("p", parent(), OFFERED, request(), depth=MAX_DEPTH - 1, children=MAX_CHILDREN - 1)
    with pytest.raises(AidError, match="ancestors"):
        child_spec("p", parent(), OFFERED, request(), depth=MAX_DEPTH, children=0)
    with pytest.raises(AidError, match="children"):
        child_spec("p", parent(), OFFERED, request(), depth=0, children=MAX_CHILDREN)


def test_only_native_sessions_spawn() -> None:
    with pytest.raises(AidError, match="native"):
        child_spec("p", AcpSpec(cwd="/", command=["x"]), OFFERED, request(), depth=0, children=0)


def test_a_child_name_fits_a_session_name() -> None:
    assert len(child_name("x" * 64, "a" * 16)) == 64
    assert child_name("p", "reader") == "p.reader"
