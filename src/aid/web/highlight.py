"""Syntax highlighting for the file viewer, by tree-sitter, as Pygments short token classes (`theme.CODE`).

The grammars are a directory (`nix/tree-sitter-grammars.nix`): `languages.json`, and per language a `parser`
shared object and its `highlights.scm`. Spans come back as UTF-16 offsets into the text, which is what a browser
string counts.

Precedence follows tree-sitter-highlight (crates/highlight/src/highlight.rs, 0.25): of the patterns that
capture one node, the last one wins, and a node's capture is drawn over its parent's.
"""

from __future__ import annotations

import ctypes
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from tree_sitter import Language, Parser, Query, QueryCursor

ENV_GRAMMARS: Final = "AID_TREE_SITTER_GRAMMARS"

type Span = tuple[int, int, str]
"""Start and end in UTF-16 code units, and the Pygments class."""

# Capture names as the grammars' queries spell them. A name missing here falls back to its prefix
# (`keyword.return` to `keyword`); one with no known prefix draws nothing. "" draws plain text over a parent.
CLASSES: Final = {
    "attribute": "na",
    "boolean": "kc",
    "character": "sc",
    "character.special": "se",
    "comment": "c",
    "comment.documentation": "sd",
    "conditional": "k",
    "constant": "no",
    "constant.builtin": "kc",
    "constructor": "nc",
    "delimiter": "p",
    "diff.delta": "gu",
    "diff.minus": "gd",
    "diff.plus": "gi",
    "embedded": "",
    "escape": "se",
    "exception": "k",
    "field": "py",
    "float": "mf",
    "function": "nf",
    "function.builtin": "nb",
    "function.macro": "fm",
    "function.method.builtin": "nb",
    "function.special": "fm",
    "include": "kn",
    "keyword": "k",
    "keyword.directive": "cp",
    "keyword.function": "kd",
    "keyword.import": "kn",
    "keyword.operator": "ow",
    "label": "nl",
    "method": "nf",
    "module": "nn",
    "namespace": "nn",
    "none": "",
    "number": "m",
    "number.float": "mf",
    "operator": "o",
    "parameter": "n",
    "preproc": "cp",
    "property": "py",
    "punctuation": "p",
    "repeat": "k",
    "storageclass": "kd",
    "string": "s",
    "string.escape": "se",
    "string.regex": "sr",
    "string.special": "ss",
    "string.special.key": "nt",
    "string.special.path": "sx",
    "string.special.regex": "sr",
    "string.special.uri": "sx",
    "tag": "nt",
    "tag.error": "err",
    "type": "nc",
    "type.builtin": "kt",
    "type.qualifier": "kd",
    "variable": "n",
    "variable.builtin": "bp",
    "variable.member": "py",
}


def token_class(capture: str) -> str | None:
    name = capture
    while name:
        if name in CLASSES:
            return CLASSES[name]
        name = name.rpartition(".")[0]
    return None


_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.restype = ctypes.py_object
_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _language(parser: Path, symbol: str) -> Language:
    function = getattr(ctypes.CDLL(str(parser)), symbol)
    function.restype = ctypes.c_void_p
    # A capsule: py-tree-sitter 0.25 deprecates a bare pointer.
    capsule: object = _capsule(function(), b"tree_sitter.Language", None)
    return Language(capsule)


@dataclass(frozen=True)
class _Compiled:
    language: Language
    query: Query
    classes: dict[str, str | None]
    """By capture name."""


class Grammars:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._types: dict[str, str] = {}
        for name, language in json.loads((root / "languages.json").read_text()).items():
            for file_type in language["file_types"]:
                self._types.setdefault(file_type, name)
        self._symbols = {name: f"tree_sitter_{name}" for name in set(self._types.values())}
        self._compiled: dict[str, _Compiled] = {}
        self._lock = threading.Lock()

    @property
    def languages(self) -> list[str]:
        return sorted(self._symbols)

    def language_of(self, path: str) -> str | None:
        """By whole file name first, then by extension, as tree-sitter.json's file types match."""
        name = PurePosixPath(path).name
        return self._types.get(name) or self._types.get(name.rpartition(".")[2] if "." in name else "")

    def compile(self, language: str) -> _Compiled:
        with self._lock:
            if (found := self._compiled.get(language)) is None:
                directory = self.root / language
                lang = _language(directory / "parser", self._symbols[language])
                query = Query(lang, (directory / "highlights.scm").read_text())
                # A property at runtime; the 0.25 stub calls it a method.
                count: int = query.capture_count  # pyright: ignore[reportAssignmentType]
                names = (query.capture_name(i) for i in range(count))
                classes = {name: token_class(name) for name in names}
                found = self._compiled[language] = _Compiled(lang, query, classes)
            return found

    def highlight(self, path: str, text: str) -> list[Span] | None:
        """None when no grammar knows the file."""
        if (language := self.language_of(path)) is None:
            return None
        return spans(self.compile(language), text)


def spans(compiled: _Compiled, text: str) -> list[Span]:
    data = text.encode()
    tree = Parser(compiled.language).parse(data)
    captured: list[tuple[int, int, int, str]] = []
    for pattern, captures in QueryCursor(compiled.query).matches(tree.root_node):
        for capture, nodes in captures.items():
            if (cls := compiled.classes[capture]) is not None:
                captured.extend((node.start_byte, node.end_byte, pattern, cls) for node in nodes)
    # Outer before inner, and for one range the earlier pattern first, so what is painted last wins.
    captured.sort(key=lambda c: (c[0], -c[1], c[2]))
    ids: dict[str, int] = {"": 0}
    paint = bytearray(len(data))
    for start, end, _pattern, cls in captured:
        index = ids.setdefault(cls, len(ids))
        paint[start:end] = bytes((index,)) * (end - start)
    by_id = {index: cls for cls, index in ids.items()}
    runs = [(m.start(), m.end(), by_id[m.group()[0]]) for m in _RUNS.finditer(paint)]
    return _utf16(data, runs) if not text.isascii() else runs


# A run of one nonzero class id. Ids stay under 256: a query has far fewer classes than that.
_RUNS: Final = re.compile(rb"([^\x00])\1*")


def _utf16(data: bytes, runs: list[Span]) -> list[Span]:
    """Byte offsets to UTF-16 ones. Node boundaries fall between characters, so every slice decodes."""
    offsets: dict[int, int] = {0: 0}
    position = units = 0
    for boundary in sorted({b for start, end, _cls in runs for b in (start, end)}):
        units += len(data[position:boundary].decode().encode("utf-16-le")) // 2
        offsets[boundary] = units
        position = boundary
    return [(offsets[start], offsets[end], cls) for start, end, cls in runs]
