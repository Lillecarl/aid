from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from plugins.campaign.subsystems import (
    OVERVIEW_LINES,
    MapProblem,
    Subsystem,
    SubsystemMap,
    assign,
    integration_prompt,
    overview,
    parse_map,
    subsystem_context,
)
from plugins.campaign.units import FileInfo

if TYPE_CHECKING:
    from collections.abc import Mapping


def files(*paths: str) -> list[FileInfo]:
    return [FileInfo(path=path, lines=10, size=100) for path in paths]


def test_a_model_writes_globs_three_ways_and_each_covers_the_directory() -> None:
    tree = files("src/auth/login.py", "src/auth/sub/token.py", "src/api.py", "deploy/k8s.yaml", "README.md")
    parts = [
        Subsystem(name="auth", globs=["./src/auth/"]),
        Subsystem(name="api", globs=["src/*.py"]),
        Subsystem(name="deploy", globs=["deploy/**"]),
    ]
    groups = assign(tree, parts)
    assert [[f.path for f in group] for group in groups] == [
        ["src/auth/login.py", "src/auth/sub/token.py"],
        ["src/api.py"],
        ["deploy/k8s.yaml"],
        ["README.md"],
    ]
    assert assign(files("src/auth/x.py"), [Subsystem(name="a", globs=["src/auth/*"])])[0] != []


def test_the_first_matching_subsystem_takes_a_file() -> None:
    groups = assign(files("lib/x.py"), [Subsystem(name="a", globs=["lib"]), Subsystem(name="b", globs=["**"])])
    assert [len(group) for group in groups] == [1, 0, 0]


def answer(*parts: Mapping[str, object]) -> str:
    return f"Here is the map:\n```json\n{json.dumps({'subsystems': list(parts)})}\n```"


def test_a_map_leaving_too_much_unmatched_is_refused_with_what_it_missed() -> None:
    tree = files(*(f"a/{n}.py" for n in range(6)), "b/x.py", "c/y.py", "d/z.py")
    with pytest.raises(MapProblem, match=r"3 of 9 files .*\n- b/x.py\n- c/y.py\n- d/z.py"):
        parse_map(answer({"name": "a", "globs": ["a/**"]}, {"name": "b", "globs": ["nothing/**"]}), tree)
    mapped, groups = parse_map(answer({"name": "a", "globs": ["a"]}, {"name": "rest", "globs": ["b", "c"]}), tree)
    assert [s.name for s in mapped.subsystems] == ["a", "rest"]
    assert [len(g) for g in groups] == [6, 2, 1]


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        answer({"name": "only", "globs": ["**"]}),
        answer({"name": "a", "globs": []}, {"name": "b", "globs": ["**"]}),
        answer(*({"name": f"s{n}", "globs": [f"s{n}"]} for n in range(13))),
    ],
)
def test_a_malformed_map_is_refused(text: str) -> None:
    with pytest.raises(MapProblem, match="not valid"):
        parse_map(text, files("x.py"))


def test_a_bare_json_answer_parses() -> None:
    text = json.dumps({"subsystems": [{"name": "a", "globs": ["**"]}, {"name": "b", "globs": ["b"]}]})
    mapped, _ = parse_map(f"Map: {text} done.", files("x.py"))
    assert len(mapped.subsystems) == 2


def test_the_overview_sums_deep_trees_until_it_fits() -> None:
    deep = files(*(f"d{a}/e{b}/f.py" for a in range(30) for b in range(30)))
    text = overview(deep)
    assert len(text.splitlines()) <= OVERVIEW_LINES + 1
    assert "d0/  30 files, 300 lines" in text and "summed into their parents" in text
    assert overview(files("a/b.py", "top.py")) == "./  1 files, 10 lines\na/  1 files, 10 lines".replace("./", ".")


def test_prompts_name_the_subsystem_and_clip_answers() -> None:
    mapped = SubsystemMap(
        subsystems=[Subsystem(name="auth", globs=["auth"], purpose="logs in"), Subsystem(name="api", globs=["api"])]
    )
    assert subsystem_context(mapped, 0).startswith("Subsystem: auth\nPurpose: logs in")
    assert "- auth: logs in" in subsystem_context(mapped, 1)
    assert subsystem_context(mapped, 2).startswith("Subsystem: other")
    prompt = integration_prompt("A brief.", mapped, [("auth", "x" * 5000)] * 100)
    assert "x" * 1500 + "…" in prompt and "more unit answers left out" in prompt
    assert len(prompt) < 70_000
