"""What a PR review runs: `gh` for the PR and CI's artifact, `git` for the checkout, `nix` for the builds. The plugin
runs these, never a model.
"""

from __future__ import annotations

import codecs
import contextlib
import shutil
import time
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
import structlog
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from aid.protocol import AidError

from .duplicates import Candidate
from .review import Build, Pr

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

log = structlog.get_logger(__name__)

REPO: Final = "NixOS/nixpkgs"
CI_WORKFLOW: Final = "PR"
"""The workflow whose `comparison` artifact lists each platform's rebuilds."""
ARTIFACT: Final = "comparison"
CHANGED_PATHS: Final = "changed-paths.json"
HEAD_REF: Final = "refs/prreview/head"
LOG_LINES: Final = 60
SAY_EVERY: Final = 0.5
"""Seconds between live-output sends while a command writes."""
type Say = Callable[[str], Awaitable[None]]
PR_FIELDS: Final = "number,headRefOid,baseRefName,title,body,author,url,updatedAt"
CANDIDATE_FIELDS: Final = "number,title,state,updatedAt,files"


class Commands(BaseModel):
    """The programs, as argv prefixes: tests put a script in each."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gh: list[str] = Field(default_factory=lambda: ["gh"])
    git: list[str] = Field(default_factory=lambda: ["git"])
    nix: list[str] = Field(default_factory=lambda: ["nix"])


class _Hit(BaseModel):
    number: int
    updatedAt: str = ""


class _Author(BaseModel):
    login: str = ""


class _PrView(BaseModel):
    number: int
    headRefOid: str
    baseRefName: str
    title: str = ""
    body: str | None = None
    author: _Author | None = None
    url: str = ""
    updatedAt: str = ""


class _File(BaseModel):
    path: str


class _CandidateView(BaseModel):
    number: int
    title: str = ""
    state: str
    updatedAt: str = ""
    files: list[_File] = Field(default_factory=list[_File])


class _WorkflowRun(BaseModel):
    id: int
    name: str = ""
    conclusion: str | None = None


class _Runs(BaseModel):
    workflow_runs: list[_WorkflowRun] = Field(default_factory=list[_WorkflowRun])


class _PathInfo(BaseModel):
    path: str | None = None
    closureSize: int = 0


_HITS: Final = TypeAdapter(list[_Hit])
type _PathInfos = dict[str, _PathInfo | None] | list[_PathInfo]
_PATH_INFO: Final[TypeAdapter[_PathInfos]] = TypeAdapter(_PathInfos)
"""`nix path-info --json`: an object keyed by path since Nix 2.19, a list before."""


async def _quiet(text: str) -> None:
    pass


async def streamed(argv: Sequence[str], say: Say = _quiet, cwd: Path | None = None) -> tuple[int, bytes, bytes]:
    """Run `argv`, passing its stderr to `say` as it comes, at most every `SAY_EVERY` seconds; then its exit code,
    stdout and stderr. A cancel kills it (anyio's `Process.aclose`)."""
    out, err = bytearray(), bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending: list[str] = []
    said = time.monotonic()

    async def flush() -> None:
        nonlocal said
        if pending:
            text = "".join(pending)
            pending.clear()
            said = time.monotonic()
            await say(text)

    async with await anyio.open_process(list(argv), cwd=cwd) as process:
        assert process.stdout is not None and process.stderr is not None

        async def pump_out() -> None:
            async for chunk in process.stdout:  # pyright: ignore[reportOptionalIterable] -- asserted above
                out.extend(chunk)

        async with anyio.create_task_group() as tg:
            tg.start_soon(pump_out)
            async for chunk in process.stderr:
                err.extend(chunk)
                pending.append(decoder.decode(chunk))
                if time.monotonic() - said >= SAY_EVERY:
                    await flush()
        pending.append(decoder.decode(b"", final=True))
        await flush()
        code = await process.wait()
    return code, bytes(out), bytes(err)


async def _run(argv: Sequence[str], what: str, cwd: Path | None = None, say: Say = _quiet) -> str:
    code, stdout, stderr = await streamed(argv, say, cwd)
    if code != 0:
        tail = stderr.decode(errors="replace").strip()[-2000:]
        raise AidError("command_failed", f"{what} exited {code}: {tail}")
    return stdout.decode(errors="replace")


async def search(commands: Commands, query: Sequence[str], limit: int, sort: str, order: str) -> list[int]:
    """Open PRs the search finds, in GitHub's `sort` and `order`. GitHub sorts, never this: its default order is
    best match, and `limit` cuts that (measured 2026-11-08: over 1000 matches, the first 30 all reviewed, so every
    search came back empty)."""
    argv = [*commands.gh, "search", "prs", "--repo", REPO, *query, "--sort", sort, "--order", order]
    out = await _run([*argv, "--limit", str(limit), "--json", "number,updatedAt"], "gh search prs")
    return [hit.number for hit in _HITS.validate_json(out)]


async def pr(commands: Commands, number: int) -> Pr:
    out = await _run([*commands.gh, "pr", "view", str(number), "--repo", REPO, "--json", PR_FIELDS], "gh pr view")
    view = _PrView.model_validate_json(out)
    return Pr(
        number=view.number,
        head=view.headRefOid,
        base=view.baseRefName,
        title=view.title,
        body=view.body or "",
        author=view.author.login if view.author else "",
        url=view.url,
        updated=view.updatedAt,
    )


async def related(commands: Commands, terms: Sequence[str], merged_since: str, limit: int) -> list[int]:
    """PRs whose title names one of `terms`: open ones, and ones merged since `merged_since` (YYYY-MM-DD)."""
    found: list[int] = []
    for term in terms:
        for state in (["--state", "open"], ["--merged", "--merged-at", f">={merged_since}"]):
            out = await _run(
                [
                    *commands.gh,
                    "search",
                    "prs",
                    "--repo",
                    REPO,
                    f"{term} in:title",
                    *state,
                    "--limit",
                    str(limit),
                    "--json",
                    "number,updatedAt",
                ],
                "gh search prs",
            )
            found.extend(hit.number for hit in _HITS.validate_json(out) if hit.number not in found)
    return found


async def candidate(commands: Commands, number: int, files: set[str]) -> Candidate | None:
    """PR `number` as a possible duplicate: None unless it is open or merged and touches one of `files`."""
    out = await _run(
        [*commands.gh, "pr", "view", str(number), "--repo", REPO, "--json", CANDIDATE_FIELDS], "gh pr view"
    )
    view = _CandidateView.model_validate_json(out)
    if view.state not in {"OPEN", "MERGED"} or not files & {f.path for f in view.files}:
        return None
    return Candidate(
        number=view.number,
        title=view.title,
        state="OPEN" if view.state == "OPEN" else "MERGED",
        updated=view.updatedAt,
        diff=await diff(commands, number),
    )


async def diff(commands: Commands, number: int) -> str:
    """What GitHub shows: against the merge base, so review comments land on its lines."""
    return await _run([*commands.gh, "pr", "diff", str(number), "--repo", REPO], "gh pr diff")


async def rebuild_list(commands: Commands, head: str, into: Path) -> str | None:
    """`changed-paths.json` of the head's CI run, None when no run or artifact holds one (artifacts expire)."""
    out = await _run([*commands.gh, "api", f"repos/{REPO}/actions/runs?head_sha={head}"], "gh api runs")
    runs = [r for r in _Runs.model_validate_json(out).workflow_runs if r.name == CI_WORKFLOW]
    runs = [r for r in runs if r.conclusion == "success"] or runs
    if not runs:
        return None
    try:
        await _run(
            [
                *commands.gh,
                "run",
                "download",
                str(runs[0].id),
                "--repo",
                REPO,
                "--name",
                ARTIFACT,
                "--dir",
                str(into),
            ],
            "gh run download",
        )
    except AidError as failure:
        log.info("no_artifact", head=head, error=failure.message)
        return None
    found = into / CHANGED_PATHS
    return await anyio.Path(found).read_text() if await anyio.Path(found).exists() else None


async def clone(commands: Commands, path: Path) -> None:
    """A bare repository of the plugin's own: it holds only what one PR's checkout fetched."""
    if not await anyio.Path(path / "HEAD").exists():
        await _run([*commands.git, "init", "--bare", "--quiet", str(path)], "git init")


async def checkout(commands: Commands, repo: Path, remote: str, pr: Pr, worktree: Path, say: Say = _quiet) -> None:
    """The PR's head, alone (`--depth 1`), as a detached worktree. A head that moved since `pr` refuses. A worktree
    left behind (a daemon killed mid-PR, measured 2026-11-07) goes first: `worktree add` refuses a path that exists."""
    git = [*commands.git, "-C", str(repo)]
    if await anyio.Path(worktree).exists():
        log.warning("stale_worktree", path=str(worktree))
        with contextlib.suppress(AidError):
            await _run([*git, "worktree", "remove", "--force", str(worktree)], "git worktree remove")
        if await anyio.Path(worktree).exists():
            await anyio.to_thread.run_sync(shutil.rmtree, worktree)
    await _run([*git, "worktree", "prune"], "git worktree prune")
    ref = f"+refs/pull/{pr.number}/head:{HEAD_REF}"
    await _run([*git, "fetch", "--progress", "--no-tags", "--depth", "1", remote, ref], "git fetch", say=say)
    fetched = (await _run([*git, "rev-parse", HEAD_REF], "git rev-parse")).strip()
    if fetched != pr.head:
        raise AidError("head_moved", f"the PR's head is {fetched}, not {pr.head}")
    await _run([*git, "worktree", "add", "--quiet", "--force", "--detach", str(worktree), pr.head], "git worktree")


async def clean(commands: Commands, repo: Path, worktree: Path) -> None:
    """The worktree goes, and every object only it needed: the disk holds one PR at a time."""
    git = [*commands.git, "-C", str(repo)]
    for argv, what in (
        ([*git, "worktree", "remove", "--force", str(worktree)], "git worktree remove"),
        ([*git, "update-ref", "-d", HEAD_REF], "git update-ref"),
        ([*git, "gc", "--quiet", "--prune=now"], "git gc"),
    ):
        try:
            await _run(argv, what)
        except AidError as failure:
            log.warning("clean_failed", step=what, error=failure.message)


async def build(
    commands: Commands, worktree: Path, attr: str, allowed: float, nixpkgs_config: str, say: Say = _quiet
) -> Build:
    """One attribute of the worktree, outside any nixpkgs config of the daemon's user; past `allowed` seconds it is
    stopped and recorded as a timeout."""
    argv = [
        *commands.nix,
        "build",
        "--file",
        str(worktree),
        "--arg",
        "config",
        nixpkgs_config,
        "--arg",
        "overlays",
        "[ ]",
        attr,
        "--no-link",
        "--print-out-paths",
        "--print-build-logs",
    ]
    started = time.monotonic()
    result = None
    with anyio.move_on_after(allowed):
        result = await streamed(argv, say)
    seconds = time.monotonic() - started
    if result is None:
        return Build(attr=attr, outcome="timeout", seconds=seconds, log_tail="")
    code, stdout, stderr = result
    tail = "\n".join(stderr.decode(errors="replace").splitlines()[-LOG_LINES:])
    if code != 0:
        return Build(attr=attr, outcome="failed", seconds=seconds, log_tail=tail)
    paths = tuple(line for line in stdout.decode().split() if line.startswith("/"))
    return Build(
        attr=attr,
        outcome="built",
        seconds=seconds,
        log_tail=tail,
        out_paths=paths,
        closure_size=await closure_size(commands, paths),
    )


async def closure_size(commands: Commands, paths: Sequence[str]) -> int | None:
    if not paths:
        return None
    try:
        out = await _run([*commands.nix, "path-info", "--closure-size", "--json", *paths], "nix path-info")
    except AidError:
        return None
    info = _PATH_INFO.validate_json(out)
    rows = info.values() if isinstance(info, dict) else info
    return max((row.closureSize for row in rows if row is not None), default=None)


async def delete(commands: Commands, paths: Sequence[str]) -> None:
    """Only the PR's own outputs, all in one call (they may refer to each other); what is still alive stays."""
    if paths:
        try:
            await _run([*commands.nix, "store", "delete", *paths], "nix store delete")
        except AidError as failure:
            log.warning("delete_failed", paths=len(paths), error=failure.message)
