"""`gh` and `pynix-review` for the PR review tests: `python prreview_fakes.py gh|pynix-review FIXTURES ARGS...`.
Answers come from files in FIXTURES (`pr.json`, `diff`, `changed-paths.json`; `related.json` {"open": [N],
"merged": [N]} with `pr-N.json` and `diff-N` per related PR); each pynix-review call is logged to
`pynix-review.log`, one JSON object a line: its argv and the `GITHUB_TOKEN` it got."""

from __future__ import annotations

import json
import os
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
        case ["auth", "token"]:
            print("fake-token")
        case ["api", *_]:
            print(json.dumps({"workflow_runs": [{"id": 11, "name": "PR", "conclusion": "success"}]}))
        case ["run", "download", *_] if not (fixtures / "changed-paths.json").exists():
            print("no valid artifacts found to download", file=sys.stderr)
            return 1
        case ["run", "download", *rest]:
            into = Path(rest[rest.index("--dir") + 1])
            into.mkdir(parents=True, exist_ok=True)
            (into / "changed-paths.json").write_text((fixtures / "changed-paths.json").read_text())
        case _:
            print(f"fake gh: {args}", file=sys.stderr)
            return 1
    return 0


def pynix_review(fixtures: Path, args: list[str]) -> int:
    """`pr N --out DIR ...`: one changed derivation per attribute CI rebuilds, `broken` failing; the head is
    `head` from FIXTURES when that file exists, else the PR's."""
    with (fixtures / "pynix-review.log").open("a") as log:
        log.write(json.dumps({"argv": args, "token": os.environ.get("GITHUB_TOKEN")}) + "\n")
    if args[:1] != ["pr"]:
        return 1
    out = Path(args[args.index("--out") + 1])
    limit = int(args[args.index("--max-derivations") + 1])
    moved = fixtures / "head"
    view = json.loads((fixtures / "pr.json").read_text())
    head = moved.read_text().strip() if moved.exists() else view["headRefOid"]
    attrs = sorted(json.loads((fixtures / "changed-paths.json").read_text())["rebuildsByPlatform"]["x86_64-linux"])
    pr = {"head": head, "merge_base": "base0", "attrs": attrs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "pr.json").write_text(json.dumps(pr))
    print("evaluating", file=sys.stderr)
    if len(attrs) > limit:
        print(json.dumps({"pr": pr, "skipped": f"{len(attrs)} changed derivations, more than {limit}"}))
        return 0
    failed = [a for a in attrs if a == "broken"]
    report = {
        "results": [{"pname": a, "status": "failed" if a in failed else "built"} for a in attrs],
        "built": len(attrs) - len(failed),
        "failed": len(failed),
        "unchanged_input_failed": 0,
        "out_of_budget": 0,
        "wall_seconds": 4.0,
    }
    built = out / "built"
    built.mkdir()
    (built / "built.json").write_text(json.dumps(report))
    link = "https://example.test/#(" + "x" * 300 + ")"
    (built / "built.md").write_text(
        f"### pynix-review build\n\n[Graph]({link})\n\n| result | derivations |\n| --- | --- |\n"
        f"| built | {report['built']} |\n| failed | {report['failed']} |\n"
    )
    print(json.dumps({"pr": pr, "skipped": None, "bundle": {}, "built": {"files": [], "view": link}}))
    return 0


if __name__ == "__main__":
    program, fixtures, *rest = sys.argv[1:]
    sys.exit((gh if program == "gh" else pynix_review)(Path(fixtures), rest))
