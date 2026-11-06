"""`gh` and `nix` for the PR review tests: `python prreview_fakes.py gh|nix FIXTURES ARGS...`. Answers come from files
in FIXTURES (`pr.json`, `diff`, `changed-paths.json`; `related.json` {"open": [N], "merged": [N]} with `pr-N.json`
and `diff-N` per related PR); each nix call is logged to `nix.log`, one JSON argv a line."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def gh(fixtures: Path, args: list[str]) -> int:
    match args:
        case ["search", "prs", *rest] if "--checks" in rest:
            print(json.dumps([{"number": json.loads((fixtures / "pr.json").read_text())["number"], "updatedAt": "x"}]))
        case ["search", "prs", *rest]:
            file = fixtures / "related.json"
            related: dict[str, list[int]] = json.loads(file.read_text()) if file.exists() else {}
            found = related.get("merged" if "--merged" in rest else "open", [])
            print(json.dumps([{"number": n, "updatedAt": "x"} for n in found]))
        case ["pr", "view", number, *_]:
            own = fixtures / f"pr-{number}.json"
            print((own if own.exists() else fixtures / "pr.json").read_text())
        case ["pr", "diff", number, *_]:
            own = fixtures / f"diff-{number}"
            sys.stdout.write((own if own.exists() else fixtures / "diff").read_text())
        case ["api", *_]:
            print(json.dumps({"workflow_runs": [{"id": 11, "name": "PR", "conclusion": "success"}]}))
        case ["run", "download", *rest]:
            into = Path(rest[rest.index("--dir") + 1])
            into.mkdir(parents=True, exist_ok=True)
            (into / "changed-paths.json").write_text((fixtures / "changed-paths.json").read_text())
        case _:
            print(f"fake gh: {args}", file=sys.stderr)
            return 1
    return 0


def nix(fixtures: Path, args: list[str]) -> int:
    with (fixtures / "nix.log").open("a") as log:
        log.write(json.dumps(args) + "\n")
    match args:
        case ["build", *rest]:
            attr = rest[rest.index("--no-link") - 1]
            if attr == "broken":
                print("error: builder for broken failed", file=sys.stderr)
                return 1
            print(f"/nix/store/fake-{attr}")
        case ["path-info", *rest]:
            print(json.dumps({path: {"closureSize": 2**20} for path in rest if path.startswith("/")}))
        case ["store", "delete", *_]:
            pass
        case _:
            return 1
    return 0


if __name__ == "__main__":
    program, fixtures, *rest = sys.argv[1:]
    sys.exit((gh if program == "gh" else nix)(Path(fixtures), rest))
