from __future__ import annotations

import json

from plugins.prreview.review import (
    LineComment,
    Pr,
    angle_prompt,
    comment_problems,
    diff_lines,
    rebuilds,
)

DIFF = """\
diff --git a/a.nix b/a.nix
--- a/a.nix
+++ b/a.nix
@@ -10,4 +10,4 @@ let
 keep
--- a removed line that starts with two dashes
+++ an added line that starts with two pluses
 tail
\\ No newline at end of file
diff --git a/gone.nix b/gone.nix
deleted file mode 100644
--- a/gone.nix
+++ /dev/null
@@ -1,2 +0,0 @@
-one
-two
diff --git a/new.nix b/new.nix
new file mode 100644
--- /dev/null
+++ b/new.nix
@@ -0,0 +1 @@
+only
diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
"""


def test_the_lines_a_diff_shows_follow_hunk_counts_not_prefixes() -> None:
    shown = diff_lines(DIFF)
    assert (shown["a.nix"].left, shown["a.nix"].right) == ({10, 11, 12}, {10, 11, 12})
    assert (shown["gone.nix"].left, shown["gone.nix"].right) == ({1, 2}, set())
    assert (shown["new.nix"].left, shown["new.nix"].right) == (set(), {1})
    assert "logo.png" not in shown


def test_a_comment_off_the_diff_is_named() -> None:
    shown = diff_lines(DIFF)
    comments = [
        LineComment(path="a.nix", line=11, body="ok"),
        LineComment(path="gone.nix", line=2, side="LEFT", body="ok"),
        LineComment(path="a.nix", line=13, body="no"),
        LineComment(path="other.nix", line=1, body="no"),
        LineComment(path="gone.nix", line=1, body="no"),
    ]
    assert comment_problems(comments, shown) == [
        "comment 3: the diff shows no RIGHT line 13 of a.nix",
        "comment 4: other.nix is not in the diff",
        "comment 5: the diff shows no RIGHT line 1 of gone.nix",
    ]


def test_rebuilds_are_ci_s_list_for_the_platform() -> None:
    changed = {"rebuildsByPlatform": {"x86_64-linux": ["b", "a", "a"], "aarch64-linux": ["c"]}, "labels": {}}
    assert rebuilds(json.dumps(changed)) == ["a", "b"]
    assert rebuilds(json.dumps({"attrdiff": {}})) == []


def test_a_long_diff_is_cut_and_says_so() -> None:
    pr = Pr(number=1, head="h", base="master", title="t", body="b", author="a", url="u")
    prompt = angle_prompt(pr, "x" * 2000, [], None, 1000)
    assert 'truncated="at 1000 characters' in prompt
    assert "x" * 1001 not in prompt
    assert "CI lists nothing to rebuild." in prompt
