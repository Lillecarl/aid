"""Findings: sessions report issues they find into a local database, for review passes after them
(docs/audit-campaigns.md).

`report_finding` writes one row and nothing else: the place it names is checked inside the caller's directory,
which the plugin reads from the caller's spec. The operator queries with `aid findings query`. Bundled and on by
default (`autostart`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import anyio
import anyio.to_thread
from pydantic import ValidationError

import aid
from aid.paths import default_paths
from aid.protocol import AidError, GetSpec
from aid.spec import AgentSpecAdapter, NativeSpec
from aid.sql import ask

from .store import FINDINGS_FILE, FindingStore, Report, Verdict, check

if TYPE_CHECKING:
    from pydantic import JsonValue

PLUGIN: Final = "findings"
AI_REVIEW: Final = "ai"
"""`verdicts.pass` of a model's judgement; a person's is "human"."""


@aid.mcptool
async def report_finding(
    path: str,
    category: str,
    severity: str,
    title: str,
    detail: str,
    line_start: int | None = None,
    line_end: int | None = None,
    evidence: str | None = None,
) -> str:
    """Record one issue you found, so people review it later. Report each issue once, as soon as you are sure of it.

    `path`: the file (or directory) relative to your working directory. `category`: security, secret, bug, config,
    dependency, quality or other. `severity`: critical, high, medium, low or info. `title`: one line. `detail`: what
    is wrong, why it matters and how to fix it. `line_start`/`line_end`: the lines it is on. `evidence`: the code
    that shows it, quoted exactly from the file; a quote that is not in the file is refused."""
    async with aid.connect() as client:
        result = await client.call_plugin(
            PLUGIN,
            "report",
            path=path,
            category=category,
            severity=severity,
            title=title,
            detail=detail,
            line_start=line_start,
            line_end=line_end,
            evidence=evidence,
        )
    return str(result)


@aid.mcptool
async def record_verdict(
    finding: int,
    status: str,
    note: str,
    severity: str | None = None,
    duplicate_of: int | None = None,
) -> str:
    """Judge one finding after reading the code it names. `status`: confirmed (a real issue as described), rejected
    (not an issue: say why), duplicate (the same issue as `duplicate_of`, another finding's id) or unsure. `note`:
    the reason, briefly. `severity`: critical, high, medium, low or info, only when the finding's is wrong."""
    async with aid.connect() as client:
        await client.call_plugin(
            PLUGIN,
            "verdict",
            finding=finding,
            status=status,
            note=note,
            severity=severity,
            duplicate_of=duplicate_of,
        )
    return f"Recorded the verdict on finding {finding}."


class Findings(aid.Plugin):
    """The findings database's one writer: sessions report, the operator queries."""

    grants = frozenset({aid.Grant.SUPERVISE, aid.Grant.PRIVATE})

    def __init__(self) -> None:
        self._client: aid.Client | None = None
        self._store: FindingStore | None = None
        self._file: Path = default_paths().plugin_dir(self.plugin_name()) / FINDINGS_FILE

    async def run(self, client: aid.Client) -> None:
        self._client = client
        self._store = await FindingStore.open(self._file)
        try:
            await anyio.sleep_forever()
        finally:
            store, self._store = self._store, None
            await store.aclose()

    async def call(self, caller: str | None, method: str, args: dict[str, JsonValue]) -> JsonValue:
        if self._client is None or self._store is None:
            raise AidError("not_running", "findings is not running")
        match method:
            case "report":
                if caller is None:
                    raise AidError("forbidden", "a session reports findings in its own directory")
                return await self._report(self._client, self._store, caller, args)
            case "verdict":
                if caller is None:
                    raise AidError("forbidden", "a session judges findings in its own directory")
                return await self._verdict(self._client, self._store, caller, args)
            case "query":
                if caller is not None:
                    raise AidError("forbidden", "only the operator queries findings")
                sql = args.get("sql")
                if not isinstance(sql, str) or not sql.strip():
                    raise AidError("empty_query", "a query needs SQL")
                return (await ask({"findings": self._file}, sql)).model_dump(mode="json")
            case _:
                raise AidError("unknown_method", f"findings has no method {method!r}")

    @staticmethod
    async def _verdict(client: aid.Client, store: FindingStore, caller: str, args: dict[str, JsonValue]) -> None:
        try:
            verdict = Verdict.model_validate({key: value for key, value in args.items() if value is not None})
        except ValidationError as error:
            raise AidError("bad_args", str(error)) from None
        spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session=caller)))
        repo = str(await anyio.Path(spec.cwd).resolve())
        await store.judge(repo, verdict, session=caller, review=AI_REVIEW)

    @staticmethod
    async def _report(client: aid.Client, store: FindingStore, caller: str, args: dict[str, JsonValue]) -> str:
        try:
            report = Report.model_validate({key: value for key, value in args.items() if value is not None})
        except ValidationError as error:
            raise AidError("bad_args", str(error)) from None
        spec = AgentSpecAdapter.validate_python(await client.call(GetSpec(session=caller)))
        repo, path = await anyio.to_thread.run_sync(check, Path(spec.cwd), report)
        model = spec.model if isinstance(spec, NativeSpec) else None
        found, first = await store.insert(repo, path, report, session=caller, model=model)
        if first is not None:
            return f"Recorded finding {found}; finding {first} has the same place and title."
        return f"Recorded finding {found}."
