from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio
import apsw
import pytest

from aid.protocol import AidError
from aid.removals import (
    Entry,
    EntryKind,
    GitState,
    KeptReason,
    Outcome,
    PlanStore,
    Removal,
    execute,
    git_state,
    plan,
    removable,
)

if TYPE_CHECKING:
    from pathlib import Path


def tree(tmp_path: Path) -> Path:
    """build/ with a nested file, a file to keep, a link to it, and a link out of the directory."""
    work = tmp_path / "work"
    (work / "build" / "sub").mkdir(parents=True)
    (work / "build" / "a.txt").write_text("aa")
    (work / "build" / "sub" / "b.txt").write_text("bbb")
    (work / "keep.txt").write_text("keep")
    (work / "link").symlink_to("keep.txt")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("x")
    (work / "out").symlink_to(outside)
    return work.resolve()


def rows(entries: list[Entry]) -> list[tuple[str, bool, str | None]]:
    return [(e.path, e.top, e.within) for e in entries]


def test_removable_takes_only_entries_inside_the_directory(tmp_path: Path) -> None:
    root = tree(tmp_path)
    assert removable(root, "out") == root / "out"
    assert removable(root, "link") == root / "link"
    for path, code in [
        ("out/x.txt", "outside"),
        ("..", "outside"),
        (".", "outside"),
        ("../outside", "outside"),
        ("nope", "not_found"),
    ]:
        with pytest.raises(AidError) as caught:
            removable(root, path)
        assert caught.value.code == code


def test_a_matched_directory_goes_whole(tmp_path: Path) -> None:
    root = tree(tmp_path)
    planned = plan(root, Removal(paths=["build/sub", "link"], globs=["build"]), None)
    assert rows(planned.entries) == [
        ("build", True, None),
        ("build/a.txt", False, "build"),
        ("build/sub", False, "build"),
        ("build/sub/b.txt", False, "build"),
        ("link", True, None),
    ]
    assert (planned.files, planned.directories, planned.bytes) == (3, 2, 2 + 3 + len("keep.txt"))
    assert planned.entries[0].matched == "glob build"


def test_a_glob_inside_a_directory_takes_its_contents_only(tmp_path: Path) -> None:
    root = tree(tmp_path)
    planned = plan(root, Removal(globs=["build/**"]), None)
    assert rows(planned.entries) == [
        ("build/a.txt", True, None),
        ("build/sub", True, None),
        ("build/sub/b.txt", False, "build/sub"),
    ]


def test_an_exclude_splits_its_directory(tmp_path: Path) -> None:
    root = tree(tmp_path)
    (root / "build" / "sub" / "c.txt").write_text("c")
    planned = plan(root, Removal(paths=["build"], excludes=["build/sub/b.txt"]), None)
    assert rows(planned.entries) == [("build/a.txt", True, None), ("build/sub/c.txt", True, None)]
    assert [(k.path, k.reason) for k in planned.kept] == [("build/sub/b.txt", KeptReason.EXCLUDED)]


def test_an_excluded_directory_keeps_everything_in_it(tmp_path: Path) -> None:
    root = tree(tmp_path)
    planned = plan(root, Removal(globs=["build/**"], excludes=["build/sub"]), None)
    assert rows(planned.entries) == [("build/a.txt", True, None)]
    assert [k.path for k in planned.kept] == ["build/sub"]


def test_patterns_skip_git_and_never_follow_links(tmp_path: Path) -> None:
    root = tree(tmp_path)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")
    (root / ".hidden").write_text("h")
    planned = plan(root, Removal(globs=["**"], kind="file"), None)
    assert [e.path for e in planned.entries] == [".hidden", "build/a.txt", "build/sub/b.txt", "keep.txt"]
    links = plan(root, Removal(globs=["*"], kind="link"), None)
    assert [(e.path, e.kind) for e in links.entries] == [("link", EntryKind.LINK), ("out", EntryKind.LINK)]


def test_a_regex_searches_the_relative_path(tmp_path: Path) -> None:
    root = tree(tmp_path)
    planned = plan(root, Removal(regexes=[r"\.txt$", "nothing-here"]), None)
    assert [e.path for e in planned.entries] == ["build/a.txt", "build/sub/b.txt", "keep.txt"]
    assert planned.unmatched == ["regex nothing-here"]
    with pytest.raises(AidError) as caught:
        plan(root, Removal(regexes=["("]), None)
    assert caught.value.code == "invalid_regex"


def test_patterns_stay_inside_the_directory(tmp_path: Path) -> None:
    root = tree(tmp_path)
    for pattern in ["../outside/*", "/etc/*"]:
        with pytest.raises(AidError):
            plan(root, Removal(globs=[pattern]), None)
    assert plan(root, Removal(globs=["out/*"]), None).unmatched == ["glob out/*"]


def test_age_keeps_newer_files(tmp_path: Path) -> None:
    root = tree(tmp_path)
    old = 1_000_000.0
    os.utime(root / "build" / "a.txt", (old, old))
    planned = plan(root, Removal(paths=["build"], older_than_days=1), None, now=old + 2 * 86400)
    assert rows(planned.entries) == [("build/a.txt", True, None)]
    assert [(k.path, k.reason) for k in planned.kept] == [("build/sub/b.txt", KeptReason.NEWER)]


def test_untracked_only_keeps_what_git_tracks(tmp_path: Path) -> None:
    root = tree(tmp_path)
    git = GitState(
        tracked_files=frozenset({"build/sub/b.txt", "keep.txt"}),
        tracked_dirs=frozenset({"build", "build/sub"}),
        ignored=frozenset(),
    )
    planned = plan(root, Removal(globs=["*"], untracked_only=True), git)
    assert rows(planned.entries) == [("build/a.txt", True, None), ("link", True, None), ("out", True, None)]
    assert [(k.path, k.reason) for k in planned.kept] == [
        ("build/sub/b.txt", KeptReason.TRACKED),
        ("keep.txt", KeptReason.TRACKED),
    ]
    with pytest.raises(AidError):
        plan(root, Removal(globs=["*"], untracked_only=True), None)


@pytest.mark.anyio
async def test_git_state_reads_tracked_and_ignored(tmp_path: Path) -> None:
    root = tree(tmp_path)
    assert await git_state(root) is None
    (root / ".gitignore").write_text("build/\n")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    await anyio.run_process(["git", "init", "-q"], cwd=root, env=env)
    await anyio.run_process(["git", "add", "keep.txt", ".gitignore"], cwd=root, env=env)
    git = await git_state(root)
    assert git is not None
    assert git.tracked("keep.txt", EntryKind.FILE)
    assert not git.tracked("link", EntryKind.LINK)
    assert git.is_ignored("build/sub/b.txt")
    assert not git.is_ignored("keep.txt")
    planned = plan(root, Removal(globs=["**"], kind="file", ignored_only=True), git)
    assert [e.path for e in planned.entries] == ["build/a.txt", "build/sub/b.txt"]


def test_execution_removes_the_plan_not_a_new_match(tmp_path: Path) -> None:
    root = tree(tmp_path)
    planned = plan(root, Removal(paths=["build", "link", "keep.txt"]), None)
    (root / "build" / "late.txt").write_text("late")
    (root / "link").unlink()
    (root / "keep.txt").unlink()
    (root / "keep.txt").mkdir()
    assert execute(root, planned.tops) == {
        "build": Outcome.REMOVED,
        "keep.txt": Outcome.CHANGED,
        "link": Outcome.MISSING,
    }
    assert sorted(p.name for p in root.iterdir()) == ["keep.txt", "out"]


@pytest.mark.anyio
async def test_the_store_records_entries_and_outcomes(tmp_path: Path) -> None:
    root = tree(tmp_path)
    removal = Removal(paths=["build"], excludes=["build/a.txt"])
    planned = plan(root, removal, None)
    store = await PlanStore.open(tmp_path / "plans.db")
    try:
        plan_id = await store.record(removal, planned, "call-1", "req-1")
        await store.outcomes(plan_id, {"build/sub": Outcome.REMOVED})
    finally:
        await store.aclose()
    db = apsw.Connection(str(tmp_path / "plans.db"))
    assert db.execute("SELECT id, request_id, status, request FROM removal_plans").fetchall() == [
        (plan_id, "req-1", "asked", '{"paths":["build"],"excludes":["build/a.txt"]}')
    ]
    assert db.execute("SELECT path, top, outcome FROM removal_entries").fetchall() == [
        ("build/sub", 1, "removed"),
        ("build/sub/b.txt", 0, None),
    ]
    assert db.execute("SELECT path, reason FROM removal_kept").fetchall() == [("build/a.txt", "excluded")]
