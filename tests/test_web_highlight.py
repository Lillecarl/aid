from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aid.web.highlight import highlight, token_class
from tests.conftest import needs_grammars

if TYPE_CHECKING:
    from aid.web.highlight import Grammars

pytestmark = needs_grammars


def test_every_query_compiles(grammars: Grammars) -> None:
    assert len(grammars.languages) >= 20
    for language in grammars.languages:
        assert grammars.compile(language).classes, language


def test_later_pattern_wins_on_one_node(grammars: Grammars) -> None:
    text = "def f(x): return print(x)\nclass A: pass\n"
    spans = grammars.highlight("a.py", text)
    assert spans is not None
    by_text = {text[start:end]: cls for start, end, cls in spans}
    assert by_text["def"] == "k"
    assert by_text["f"] == "nf"  # @function after @variable
    assert by_text["print"] == "nb"  # @function.builtin after @function and @variable
    assert by_text["A"] == "no"  # @constant (all capitals) after @constructor


def test_spans_are_sorted_and_disjoint(grammars: Grammars) -> None:
    text = 'x = f"a{b!r}c"  # note\n'
    spans = grammars.highlight("a.py", text)
    assert spans is not None
    ends = [0, *(end for _start, end, _cls in spans)]
    assert all(start >= end for (start, _end, _cls), end in zip(spans, ends, strict=False))


def test_offsets_count_utf16_units(grammars: Grammars) -> None:
    # é is 2 UTF-8 bytes and 1 UTF-16 unit; 😀 is 4 bytes and 2 units.
    text = 's = "é😀"\nn = 42\n'
    spans = grammars.highlight("a.py", text)
    assert spans is not None
    number = len(text[: text.index("42")].encode("utf-16-le")) // 2
    assert number == text.index("42") + 1
    assert (number, number + 2, "m") in spans
    assert (4, 9, "s") in spans


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/a.py", "python"),
        ("web/x.tsx", "tsx"),
        ("Makefile", "make"),
        ("dir/.envrc", "bash"),
        ("default.nix", "nix"),
        ("README", None),
        ("notes.md", None),
    ],
)
def test_language_by_name_then_extension(grammars: Grammars, path: str, language: str | None) -> None:
    assert grammars.language_of(path) == language


def test_capture_names_fall_back_to_their_prefix() -> None:
    assert token_class("keyword.return") == "k"
    assert token_class("function.builtin") == "nb"
    assert token_class("string.special.symbol") == "ss"
    assert token_class("spell") is None


def test_a_fenced_tag_names_a_grammar_by_name_or_file_type(grammars: Grammars) -> None:
    assert grammars.named("python") == grammars.named("py") == grammars.named("PY") == "python"
    assert highlight(grammars, "py", None, "def f(): pass\n") == grammars.highlight("a.py", "def f(): pass\n")


@pytest.mark.parametrize("grammars_or_none", [True, False])
def test_pygments_takes_what_no_grammar_knows(grammars: Grammars, grammars_or_none: bool) -> None:
    text = "[core]\n  é = 😀\n"
    found = highlight(grammars if grammars_or_none else None, "ini", None, text)
    assert found is not None
    assert (0, 6, "k") in found or any(cls == "k" for _s, _e, cls in found)
    # Ends count UTF-16 units: the emoji takes two.
    assert max(end for _start, end, _cls in found) <= len(text.encode("utf-16-le")) // 2
    assert highlight(None, "no-such-language", None, text) is None
    assert highlight(None, None, None, text) is None


def _classes(spans: list[tuple[int, int, str]] | None, text: str) -> dict[str, str]:
    assert spans is not None
    return {text[start:end]: cls for start, end, cls in spans}


def test_a_script_in_html_is_highlighted_as_javascript(grammars: Grammars) -> None:
    text = "<p>let</p><script>let x = 1;</script>\n"
    classes = _classes(grammars.highlight("a.html", text), text)
    assert classes["let"] == "k"
    assert classes["1"] == "m"


def test_a_nix_builder_script_is_bash_and_its_plain_text_stays_string(grammars: Grammars) -> None:
    text = "{ x = runCommand \"a\" { } ''\n  echo \"$out\" | grep -v foo\n''; }\n"
    spans = grammars.highlight("a.nix", text)
    classes = _classes(spans, text)
    assert classes["echo"] == "nf"
    assert spans is not None
    at = text.index("foo")
    assert [cls for start, end, cls in spans if start <= at < end] == ["s"]


def test_a_comment_names_the_injected_language(grammars: Grammars) -> None:
    text = "{ y = /* python */ ''\n  def f(): pass\n''; }\n"
    assert _classes(grammars.highlight("a.nix", text), text)["def"] == "k"


def test_an_unknown_injected_language_leaves_the_host(grammars: Grammars) -> None:
    text = "{ y = /* klingon */ ''\n  def f(): pass\n''; }\n"
    spans = grammars.highlight("a.nix", text)
    assert spans is not None
    at = text.index("def")
    assert [cls for start, end, cls in spans if start <= at < end] == ["s"]
