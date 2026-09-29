from __future__ import annotations

import pytest

from aid.web.theme import Theme, css


@pytest.mark.parametrize(
    ("spec", "scheme"),
    [
        ("pygments:catppuccin-mocha", "dark"),
        ("pygments:default", "light"),
        ("base16:solarized-dark", "dark"),
        ("base16:solarized-light", "light"),
    ],
)
def test_theme_colors_the_page_and_code(spec: str, scheme: str) -> None:
    sheet = css(Theme.parse(spec))
    assert f"color-scheme: {scheme};" in sheet
    assert "--surface: #" in sheet
    assert ".hl .k {" in sheet
    assert ".hl .s {" in sheet


def test_catppuccin_is_pymuxs() -> None:
    sheet = css(Theme.parse("pygments:catppuccin-mocha"))
    assert "--surface: #181825;" in sheet
    assert ".hl .k { color: #CBA6F7 }" in sheet


def test_base16_keywords_are_base0e() -> None:
    theme = Theme.parse("base16:solarized-dark")
    assert f".hl .k {{ color: {theme.roles()['color-5'].upper()} }}" in css(theme)


def test_without_a_theme_code_follows_the_browser() -> None:
    sheet = css(None)
    assert "html:root" not in sheet
    assert "@media (prefers-color-scheme: dark)" in sheet
    assert sheet.count(".hl .k {") == 2


@pytest.mark.parametrize("spec", ["catppuccin-mocha", "pygments:", "vim:desert", "pygments:nope", "base16:nope"])
def test_unknown_theme_is_refused(spec: str) -> None:
    with pytest.raises(ValueError, match=r"expected|no such"):
        Theme.parse(spec)
