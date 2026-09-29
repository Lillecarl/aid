"""aid web's colours, from the theme pymux draws with: `pygments:<name>` or `base16:<name>`.

The page reads CSS variables from pymux's roles. Code carries Pygments' short token classes (`k`, `s`, `nf`, …)
under `.hl`, for the server's highlighter and the browser's alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from pygments.formatters import HtmlFormatter
from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from pymux.style_base16 import base16_roles  # pyright: ignore[reportMissingTypeStubs] -- annotated, no py.typed
from pymux.style_pygments import pygments_roles  # pyright: ignore[reportMissingTypeStubs]

CODE: Final = ".hl"
# The page without a theme: its own light and dark, and these two for code.
UNTHEMED: Final = ("default", "one-dark")


@dataclass(frozen=True)
class Theme:
    kind: str
    name: str

    @classmethod
    def parse(cls, spec: str) -> Theme:
        """A theme as pymux's `theme` option spells it. An unknown one raises ValueError."""
        kind, _, name = spec.partition(":")
        if kind not in ("pygments", "base16") or not name:
            raise ValueError(f"{spec!r}: expected pygments:<name> or base16:<name>")
        theme = cls(kind, name)
        try:
            theme.roles()
        except KeyError:
            raise ValueError(f"{spec!r}: no such {kind} theme") from None
        return theme

    def roles(self) -> dict[str, str]:
        return pygments_roles(self.name) if self.kind == "pygments" else base16_roles(self.name)

    def style(self) -> type[Style] | str:
        return self.name if self.kind == "pygments" else _base16_style(self.roles())


def css(theme: Theme | None) -> str:
    if theme is None:
        light, dark = (_code(style) for style in UNTHEMED)
        return f"{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n"
    return f"{_page(theme.roles())}\n{_code(theme.style())}\n"


def _code(style: type[Style] | str) -> str:
    formatter: HtmlFormatter[str] = HtmlFormatter(style=style)
    # The stubs leave both untyped; each returns a list of rules.
    background = cast("list[str]", formatter.get_background_style_defs(CODE))  # pyright: ignore[reportUnknownMemberType]
    tokens = cast("list[str]", formatter.get_token_style_defs(CODE))  # pyright: ignore[reportUnknownMemberType]
    return "\n".join([*background, *tokens])


def _page(roles: dict[str, str]) -> str:
    variables = {
        "color-scheme": "dark" if _luminance(roles["surface"]) < _luminance(roles["text"]) else "light",
        "--surface": roles["surface"],
        "--surface-raised": roles["surface-raised"],
        "--text": roles["text"],
        "--muted": roles["text-muted"],
        "--line": roles["border"],
        "--accent": roles["accent"],
        "--code-bg": roles["surface-raised"],
        "--ok": roles["color-2"],
        "--bad": roles["danger"],
    }
    body = "".join(f"  {key}: {value};\n" for key, value in variables.items())
    # html:root outranks app.css's :root, whichever stylesheet the browser reads last.
    return f"html:root {{\n{body}}}\nbody {{\n  background: var(--surface);\n  color: var(--text);\n}}"


def _luminance(color: str) -> float:
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _base16_style(roles: dict[str, str]) -> type[Style]:
    """A base16 scheme as a Pygments style, after the base16 styling guidelines.
    pymux's roles carry the letters: color-1 is base08, color-2 base0B, warn-bright base09, and so on."""
    base03, base05 = roles["color-8"], roles["text"]
    base08, base09, base0a = roles["color-1"], roles["warn-bright"], roles["color-3"]
    base0b, base0c, base0d, base0e = roles["color-2"], roles["color-6"], roles["color-4"], roles["color-5"]
    styles = {
        Token: base05,
        Comment: f"italic {base03}",
        Keyword: base0e,
        Keyword.Constant: base09,
        Keyword.Type: base0a,
        Name.Attribute: base0d,
        Name.Builtin: base0c,
        Name.Class: base0a,
        Name.Constant: base09,
        Name.Decorator: base0c,
        Name.Exception: base08,
        Name.Function: base0d,
        Name.Namespace: base0a,
        Name.Tag: base08,
        Name.Variable: base08,
        Number: base09,
        Operator: base05,
        Punctuation: base05,
        String: base0b,
        String.Escape: base0c,
        String.Regex: base0c,
        Generic.Deleted: base08,
        Generic.Inserted: base0b,
        Generic.Heading: f"bold {base0d}",
        Generic.Subheading: f"bold {base0d}",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Error: base08,
    }
    return type(
        "Base16",
        (Style,),
        {"background_color": roles["surface"], "highlight_color": roles["search-match"], "styles": styles},
    )
