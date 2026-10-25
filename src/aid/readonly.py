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
        *("ls", "cat", "head", "tail", "wc", "pwd", "grep", "egrep", "fgrep", "stat", "du", "df", "echo"),
        *("which", "type", "whoami", "id", "uname", "realpath", "readlink", "dirname", "basename", "cut", "tr"),
        *("column", "diff", "cmp", "comm", "nl", "true", "false", "test", "[", "md5sum", "sha1sum", "sha256sum"),
        *("b2sum", "hexdump", "od", "strings", "nproc", "free", "uptime", "ps", "lsblk", "tldr", "jq", "cd"),
        *("pidof", "getent", "seq", "tac", "rev", "fold", "fmt", "expand", "paste", "join", "look", "zcat"),
        *("lscpu", "findmnt", "nixos-version", "locale", "tty", "groups", "users", "who", "w", "last"),
    }
)
"""Programs that read and print, whatever their arguments. Not `find` (-delete, -exec), `sed` (-i), `env`
(prints secrets), `xargs` (runs programs) or `tee` (writes); readers with a writing or program-running option
have a rule in `PROGRAMS`."""

SEPARATORS: Final = frozenset({"|", "||", "&&", ";", "\n"})
QUIET_TARGETS: Final = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout"})
NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ASSIGNMENT: Final = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)
EXPANSION: Final = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
RUNS_CODE: Final = re.compile(
    r"PATH|ENV|BASH_ENV|IFS|PS4|PROMPT_COMMAND|SHELLOPTS|EDITOR|VISUAL|BROWSER|\w*PAGER|LESS\w*|LD_\w+"
    r"|GIT_\w+|JJ_\w+|NIX_\w+|PYTHON\w*|PERL\w*|RUBY\w*|NODE_\w+|BAT_\w+|MAN\w*|\w+_proxy|\w+_PROXY"
)
"""Environment variables that make the programs reading them run code or load it, or reroute their traffic."""
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
GIT_LISTING: Final = frozenset(
    {"-a", "-r", "-l", "-v", "-vv", "--all", "--remotes", "--list", "--show-current", "--verbose", "--color"}
    | {"--no-color", "--sort", "--format", "--column", "--no-column", "-n"}
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
JJ_GIT_READS: Final = (["remote", "list"],)
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
            # `git grep -O CMD` opens the matches in a program.
            return _without("--output", "--open-files-in-pager", "--ext-diff", short="O" if verb == "grep" else "")(
                ["git", *words[1:]]
            )
        case "branch" | "tag":
            # Patterns only with a list flag: a bare name creates one.
            listing = {"-l", "--list"} & {*words}
            return all(w.split("=", 1)[0] in GIT_LISTING if w.startswith("-") else listing for w in words[1:])
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
    # Global options also follow the subcommand: `--config ui.pager=…` and `--tool` run programs.
    if not words or not _without("--config", "--config-toml", "--config-file", "--tool")(argv):
        return False
    if words[0] in JJ_READS:
        return True
    if words[0] == "git":
        return words[1:3] in JJ_GIT_READS
    nested = JJ_NESTED_READS.get(words[0])
    return nested is not None and len(words) > 1 and words[1] in nested


def _nix(argv: list[str]) -> bool:
    words = _subcommand(argv, frozenset({"--file", "-f", "--extra-experimental-features"}))
    # Every Nix setting is also a flag: `--allow-unsafe-native-code-during-evaluation` runs programs.
    if not words or "--write-to" in words or any("unsafe" in w or w.endswith("-hook") for w in argv):
        return False
    if words[0] in NIX_READS:
        return True
    nested = NIX_NESTED_READS.get(words[0])
    return nested is not None and len(words) > 1 and words[1] in nested


def _sed(argv: list[str]) -> bool:
    if not _without("--file", short="f")(argv):
        return False
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
    if not _without("--file", "--include", "--load", "--exec", short="fiEl")(argv):
        return False
    return not any(re.search(r"system|getline|[>|]", arg) for arg in argv[1:] if not arg.startswith("-"))


def _systemctl(argv: list[str]) -> bool:
    words = [w for w in argv[1:] if not w.startswith("-")]
    return bool(words) and words[0] in SYSTEMCTL_READS


def _journalctl(argv: list[str]) -> bool:
    return not any(a.startswith(("--vacuum", "--rotate", "--flush", "--sync", "--relinquish")) for a in argv[1:])


def _timeout(argv: list[str]) -> bool:
    """`timeout [options] DURATION COMMAND…`: the command decides."""
    rest = argv[1:]
    while rest and rest[0].startswith("-"):
        option = rest.pop(0)
        if option in {"-s", "--signal", "-k", "--kill-after"} and rest:
            rest.pop(0)
    return len(rest) > 1 and argv_reads(rest[1:])


def _ip(argv: list[str]) -> bool:
    words = [w for w in argv[1:] if not w.startswith("-")]
    objects = {"addr", "address", "a", "route", "r", "link", "l", "neigh", "n", "rule"}
    return bool(words) and words[0] in objects and (len(words) == 1 or words[1] in {"show", "list", "ls", "get"})


def _without(*options: str, short: str = "") -> Callable[[list[str]], bool]:
    """A reader but for these options: long ones also as `--name=value`, `short` letters also inside a cluster."""

    def rule(argv: list[str]) -> bool:
        for arg in argv[1:]:
            if arg == "--":
                return True
            if arg.split("=", 1)[0] in options:
                return False
            if short and arg.startswith("-") and not arg.startswith("--") and set(short) & set(arg[1:]):
                return False
        return True

    return rule


def _operands(argv: list[str], valued: frozenset[str]) -> list[str]:
    """The words that are no option nor an option's value."""
    words: list[str] = []
    rest = iter(argv[1:])
    for arg in rest:
        if arg in valued:
            next(rest, None)
        elif not arg.startswith("-") or arg == "-":
            words.append(arg)
    return words


def _man(argv: list[str]) -> bool:
    """Pages by name and section, `-k` and `-f` searches: `-P` and `--html` run programs."""
    return all(not arg.startswith("-") or arg in {"-k", "-f", "-a", "-w"} for arg in argv[1:])


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
    "timeout": _timeout,
    "ip": _ip,
    "mount": lambda argv: argv == ["mount"],
    "command": lambda argv: len(argv) > 2 and argv[1] in {"-v", "-V"},
    "sort": _without("--output", "--compress-program", short="o"),
    "uniq": lambda argv: len(_operands(argv, frozenset({"-f", "-s", "-w"}))) < 2,
    "tree": _without("-o"),
    "fd": _without("--exec", "--exec-batch", short="xX"),
    "rg": _without("--pre"),
    "file": _without("--compile", short="C"),
    "xxd": lambda argv: len(_operands(argv, frozenset({"-c", "-g", "-l", "-s", "-o", "-n", "-C"}))) < 2,
    "bat": lambda argv: _without("--pager")(argv) and argv[1:2] != ["cache"],
    "man": _man,
    "printf": _without("-v"),
    "date": lambda argv: (
        _without("--set", short="s")(argv)
        and all(
            w.startswith("+") for w in _operands(argv, frozenset({"-d", "--date", "-f", "--file", "-r", "--reference"}))
        )
    ),
    "hostname": lambda argv: all(a.startswith("-") and a not in {"-F", "--file", "-b", "--boot"} for a in argv[1:]),
    "hostnamectl": lambda argv: [w for w in argv[1:] if not w.startswith("-")] in ([], ["status"]),
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
    """`NAME=value` alone sets a shell variable; before a program it is that program's environment, where PAGER,
    LD_PRELOAD and GIT_* run code. Assigning a variable already exported changes later programs' environment too:
    names that make programs run code are never set."""
    assigned = 0
    while assigned < len(words) and (match := ASSIGNMENT.fullmatch(words[assigned])) is not None:
        if RUNS_CODE.fullmatch(match.group(1)):
            return False
        known[match.group(1)] = match.group(2)
        assigned += 1
    if assigned and assigned < len(words):
        return False
    return assigned == len(words) or argv_reads(words)


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
