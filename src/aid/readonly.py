"""Rules for tool calls that plainly only read: a shell command parsed with shlex, a program's argv, or an agent's
own read tool. Pure: no IO, no aid state. `reads` says True only when every part is known to read; anything it
cannot tell (command substitution, a variable it did not see set, a redirect into a file, an unknown program or
subcommand, a credential-shaped path) is False, for a person, a classifier or a supervisor to judge.
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

READERS: Final = frozenset(
    {
        *("ls", "cat", "head", "tail", "wc", "pwd", "rg", "grep", "egrep", "fgrep", "stat", "file", "du", "df"),
        *("tree", "echo", "printf", "which", "type", "whoami", "id", "date", "uname", "realpath", "readlink"),
        *("dirname", "basename", "sort", "uniq", "cut", "tr", "column", "diff", "cmp", "comm", "nl", "true"),
        *("false", "test", "[", "md5sum", "sha1sum", "sha256sum", "b2sum", "hexdump", "od", "strings", "nproc"),
        *("free", "uptime", "ps", "lsblk", "man", "tldr", "jq", "fd", "bat", "cd", "pidof", "getent", "seq"),
        *("tac", "rev", "fold", "fmt", "expand", "paste", "join", "look", "zcat", "xxd", "lscpu", "findmnt"),
    }
)
"""Programs that read and print, whatever their arguments. Not `find` (-delete, -exec), `sed` (-i), `env`
(prints secrets), `xargs` (runs programs) or `tee` (writes)."""

SEPARATORS: Final = frozenset({"|", "||", "&&", ";", "\n"})
QUIET_TARGETS: Final = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout"})
NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ASSIGNMENT: Final = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)
EXPANSION: Final = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
SENSITIVE: Final = re.compile(
    r"(^|/)(\.ssh|\.gnupg|\.aws|\.kube|\.docker|\.password-store|\.netrc|\.privrc|\.env(\..*)?|\.pgpass"
    r"|auth\.json|credentials(\.json)?|secrets?|shadow|gshadow|id_[a-z0-9]+|.*\.(key|pem|p12|age))($|/)",
    re.IGNORECASE,
)
"""Paths a read must not touch unasked: keys, tokens and credential stores."""

GIT_GLOBAL_VALUED: Final = frozenset({"-C", "--git-dir", "--work-tree"})
GIT_READS: Final = frozenset(
    {"status", "log", "show", "diff", "blame", "rev-parse", "ls-files", "ls-tree", "cat-file", "describe"}
    | {"shortlog", "grep", "rev-list", "merge-base", "show-ref", "for-each-ref", "name-rev", "whatchanged"}
)
JJ_GLOBAL_VALUED: Final = frozenset({"-R", "--repository", "--color", "--at-op", "--at-operation"})
JJ_READS: Final = frozenset({"log", "diff", "show", "status", "st", "evolog", "obslog", "interdiff", "root"})
JJ_NESTED_READS: Final = {
    "file": frozenset({"show", "list", "annotate"}),
    "op": frozenset({"log", "show", "diff"}),
    "operation": frozenset({"log", "show", "diff"}),
    "bookmark": frozenset({"list", "l"}),
    "b": frozenset({"list", "l"}),
    "config": frozenset({"get", "list", "path"}),
    "workspace": frozenset({"list", "root"}),
    "tag": frozenset({"list", "l"}),
}
NIX_READS: Final = frozenset({"eval", "path-info", "search", "log", "why-depends", "show-derivation", "hash"})
NIX_NESTED_READS: Final = {
    "derivation": frozenset({"show"}),
    "flake": frozenset({"show", "metadata", "info"}),
    "store": frozenset({"ls", "cat", "diff-closures", "path-from-hash-part"}),
    "config": frozenset({"show"}),
}
SYSTEMCTL_READS: Final = frozenset(
    {"status", "show", "cat", "list-units", "list-unit-files", "list-timers", "is-active", "is-enabled", "is-failed"}
)
FIELDS: Final = TypeAdapter(dict[str, JsonValue])
CLAUDE_READ_TOOLS: Final = frozenset({"Read", "Grep", "Glob", "LS"})
"""Claude Code's own tools that only read."""


def _subcommand(argv: list[str], valued: frozenset[str]) -> list[str]:
    """The words after a program's global options; None-like (empty) when an unknown option comes first."""
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        option = rest.pop(0)
        if option in valued:
            if not rest:
                return []
            rest.pop(0)
        elif option.split("=", 1)[0] in valued:
            continue
        elif option not in {"--no-pager", "--ignore-working-copy", "--quiet", "--no-optional-locks"}:
            return []
    return rest


def _git(argv: list[str]) -> bool:
    words = _subcommand(argv, GIT_GLOBAL_VALUED)
    if not words:
        return False
    match words[0]:
        case verb if verb in GIT_READS:
            return "--output" not in words and not any(w.startswith("--output=") for w in words)
        case "branch" | "tag":
            return all(w.startswith("-") for w in words[1:]) and not {"-d", "-D", "-m", "-M", "-c", "-C", "-f"} & {
                *words[1:]
            }
        case "remote":
            return words[1:] in ([], ["-v"], ["show"]) or (len(words) == 3 and words[1] in {"get-url", "show"})
        case "stash":
            return words[1:2] in (["list"], ["show"])
        case "config":
            return bool(words[1:]) and words[1] in {"--get", "--get-all", "--list", "-l", "--get-regexp"}
        case _:
            return False


def _jj(argv: list[str]) -> bool:
    words = _subcommand(argv, JJ_GLOBAL_VALUED)
    if not words:
        return False
    if words[0] in JJ_READS:
        return True
    nested = JJ_NESTED_READS.get(words[0])
    return nested is not None and len(words) > 1 and words[1] in nested


def _nix(argv: list[str]) -> bool:
    words = _subcommand(argv, frozenset({"--file", "-f", "--option", "--extra-experimental-features"}))
    if not words or "--write-to" in words:
        return False
    if words[0] in NIX_READS:
        return True
    nested = NIX_NESTED_READS.get(words[0])
    return nested is not None and len(words) > 1 and words[1] in nested


def _sed(argv: list[str]) -> bool:
    for arg in argv[1:]:
        if arg.startswith("--in-place") or (arg.startswith("-") and not arg.startswith("--") and "i" in arg[1:]):
            return False
        # The w, W and e commands write files or run programs.
        if not arg.startswith("-") and re.search(r"(^|[;{}\s\d/])[wWe](\s|$)", arg):
            return False
    return True


def _find(argv: list[str]) -> bool:
    acting = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
    return not acting & {*argv[1:]}


def _awk(argv: list[str]) -> bool:
    return not any(re.search(r"system|getline|[>|]", arg) for arg in argv[1:] if not arg.startswith("-"))


def _systemctl(argv: list[str]) -> bool:
    words = [w for w in argv[1:] if not w.startswith("-")]
    return bool(words) and words[0] in SYSTEMCTL_READS


def _journalctl(argv: list[str]) -> bool:
    return not any(a.startswith(("--vacuum", "--rotate", "--flush", "--sync", "--relinquish")) for a in argv[1:])


PROGRAMS: Final[dict[str, Callable[[list[str]], bool]]] = {
    "git": _git,
    "jj": _jj,
    "nix": _nix,
    "pynix": _nix,
    "sed": _sed,
    "find": _find,
    "awk": _awk,
    "gawk": _awk,
    "systemctl": _systemctl,
    "journalctl": _journalctl,
}


def argv_reads(argv: list[str]) -> bool:
    """One program with its arguments, already split, only reads."""
    if not argv or any(SENSITIVE.search(arg) for arg in argv):
        return False
    program = argv[0].rsplit("/", 1)[-1]
    if program in {"sh", "bash"} and len(argv) == 3 and argv[1] == "-c":
        return command_reads(argv[2])
    if argv[1:] in (["--help"], ["--version"], ["-h"]):
        return True
    if program in READERS:
        return True
    if (rule := PROGRAMS.get(program)) is not None:
        return rule(argv)
    return False


def _expand(word: str, known: dict[str, str]) -> str | None:
    """`$name` and `${name}` replaced by a value set earlier in the command; None for one never set."""
    missing = False

    def value(match: re.Match[str]) -> str:
        nonlocal missing
        name = match.group(1) or match.group(2)
        if name not in known:
            missing = True
            return ""
        return known[name]

    expanded = EXPANSION.sub(value, word)
    return None if missing else expanded


def command_reads(command: str) -> bool:
    """A shell command whose every program only reads. Pipes and `;`, `&&`, `||` join programs; `NAME=literal`
    statements set variables later words may use; output may go to /dev/null or another descriptor only."""
    if "`" in command or "$(" in command or "<(" in command or ">(" in command:
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    known: dict[str, str] = {}
    segment: list[str] = []
    tokens.append(";")
    at = 0
    while at < len(tokens):
        token = tokens[at]
        at += 1
        if token in SEPARATORS:
            if segment and not _segment_reads(segment, known):
                return False
            segment = []
            continue
        if token in {">", ">>", "&>", ">|"}:
            if segment and segment[-1].isdigit():
                segment.pop()
            if at >= len(tokens) or tokens[at] not in QUIET_TARGETS:
                return False
            at += 1
            continue
        if token == ">&":
            if segment and segment[-1].isdigit():
                segment.pop()
            if at >= len(tokens) or not tokens[at].isdigit():
                return False
            at += 1
            continue
        if token == "<":
            continue
        if not token or token[0] in "&|<>()":
            return False
        expanded = _expand(token, known)
        if expanded is None:
            return False
        segment.append(expanded)
    return True


def _segment_reads(words: list[str], known: dict[str, str]) -> bool:
    while words and (assigned := ASSIGNMENT.fullmatch(words[0])) is not None:
        known[assigned.group(1)] = assigned.group(2)
        words = words[1:]
    return not words or argv_reads(words)


def reads(tool_name: str | None, tool_input: JsonValue) -> bool:
    """A permission request's call only reads: a shell `command`, a program's `argv`, or Claude's read tools on a
    path that is not credential-shaped."""
    try:
        fields = FIELDS.validate_python(tool_input)
    except ValidationError:
        return False
    if tool_name in CLAUDE_READ_TOOLS:
        return not any(isinstance(v, str) and SENSITIVE.search(v) for v in fields.values())
    match fields.get("argv"), fields.get("command"):
        case list() as argv, None if all(isinstance(a, str) for a in argv):
            return argv_reads([str(a) for a in argv])
        case None, str() as command:
            return command_reads(command)
        case _:
            return False
