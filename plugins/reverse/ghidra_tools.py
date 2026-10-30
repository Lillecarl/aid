"""The `ghidra_*` tools: one headless Ghidra per session (`ghidra_helper.py`), started on the first call inside the
session's sandbox and kept for the worker's life, so the JVM starts once and opened programs stay open.

The project lives in the workspace under `ghidra/`, with Ghidra's own settings and caches: a sandbox writes
nothing else, and a restarted worker finds its analysis where it left it.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

import anyio
import anyio.to_thread
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import JsonValue, TypeAdapter

import aid
from aid.coding import CODING, spill
from aid.protocol import AidError

if TYPE_CHECKING:
    from anyio.abc import Process

PYTHON: Final = "aid-ghidra-python"
"""pyghidra's Python, from the `reverse-tools` environment on the session's path."""
HELPER: Final = Path(__file__).with_name("ghidra_helper.py").read_text()
"""Passed with `-c`, not by path: the sandbox's private /tmp may hide where the plugin lives."""
STATE: Final = "ghidra"
"""Not `.ghidra`: Ghidra refuses a project path with an element starting with a dot (measured, 12.1.4)."""
START_SECONDS: Final = 180.0
IMPORT_SECONDS: Final = 3600.0
"""An import analyses the whole program: minutes for a large firmware image."""
CALL_SECONDS: Final = 600.0
LINE_BYTES: Final = 64 << 20
_ANSWER: Final = TypeAdapter(dict[str, JsonValue])


@dataclass
class _Helper:
    process: Process
    log: IO[bytes]
    lines: BufferedByteReceiveStream
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    seq: int = 0

    async def line(self) -> dict[str, JsonValue]:
        return _ANSWER.validate_json(await self.lines.receive_until(b"\n", LINE_BYTES))

    async def close(self) -> None:
        with anyio.CancelScope(shield=True):
            if self.process.returncode is None:
                self.process.kill()
            await self.process.wait()
            self.log.close()


_helpers: dict[Path, _Helper] = {}
"""By workspace: one per worker, as a worker serves one session."""


async def _start(workspace: Path) -> _Helper:
    coding = CODING.get()
    program = shutil.which(PYTHON)
    if program is None:
        raise AidError("no_ghidra", f"{PYTHON} is not on the path: the session lacks the reverse-tools environment")
    state = workspace / STATE
    await anyio.Path(state / "home").mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "JAVA_TOOL_OPTIONS": f"-Duser.home={state / 'home'}",
        "XDG_CONFIG_HOME": str(state / "config"),
        "XDG_CACHE_HOME": str(state / "cache"),
    }
    log = await anyio.to_thread.run_sync(lambda: (state / "helper.log").open("ab"))
    # -I: neither the helper's directory nor the workspace goes on sys.path, where a file could shadow `ghidra`.
    argv = [*coding.confine(), program, "-I", "-c", HELPER, str(state / "project")]
    process = await anyio.open_process(argv, cwd=workspace, env=env, stderr=log)
    if process.stdout is None:
        raise AidError("no_ghidra", "the Ghidra helper has no output")
    helper = _Helper(process, log, BufferedByteReceiveStream(process.stdout))
    try:
        with anyio.fail_after(START_SECONDS):
            ready = await helper.line()
    except TimeoutError, anyio.IncompleteRead, anyio.EndOfStream:
        await helper.close()
        raise AidError("no_ghidra", f"Ghidra did not start; its log is {state / 'helper.log'}") from None
    if ready.get("result") != "ready":
        await helper.close()
        raise AidError("no_ghidra", f"Ghidra said {ready!r} at start")
    return helper


async def _call(op: str, args: dict[str, JsonValue], seconds: float = CALL_SECONDS) -> JsonValue:
    workspace = CODING.get().cwd.resolve()
    helper = _helpers.get(workspace)
    if helper is None or helper.process.returncode is not None:
        helper = _helpers[workspace] = await _start(workspace)
    async with helper.lock:
        helper.seq += 1
        request = json.dumps({"id": helper.seq, "op": op, "args": args}) + "\n"
        if helper.process.stdin is None:
            raise AidError("no_ghidra", "the Ghidra helper takes no input")
        try:
            with anyio.fail_after(seconds):
                await helper.process.stdin.send(request.encode())
                answer = await helper.line()
        except TimeoutError, anyio.IncompleteRead, anyio.EndOfStream, anyio.BrokenResourceError:
            # Ghidra is mid-request or gone: the next call starts a fresh one, which reopens the saved project.
            _helpers.pop(workspace, None)
            await helper.close()
            raise AidError("ghidra_failed", f"Ghidra stopped answering {op}; the next call starts it again") from None
    if not answer.get("ok"):
        raise AidError("ghidra", str(answer.get("error")))
    return answer.get("result")


def _path(path: str) -> str:
    """`path` as the helper opens it: absolute, inside the workspace."""
    workspace = CODING.get().cwd.resolve()
    target = (workspace / path).resolve()
    if not target.is_relative_to(workspace) or not target.is_file():
        raise AidError("bad_args", f"{path} is no file in the workspace")
    return str(target)


def _rows(result: JsonValue, what: str, offset: int = 0) -> str:
    if not isinstance(result, dict):
        return str(result)
    rows = result.get("rows")
    lines = [str(r) for r in rows] if isinstance(rows, list) else []
    total = result.get("total")
    head = f"{total} {what}; rows {offset}-{offset + len(lines)}" if isinstance(total, int) else f"{len(lines)} {what}"
    return "\n".join([head, *lines])


@aid.mcptool
async def ghidra_import(
    path: str, language: str | None = None, base_address: str | None = None, compiler: str | None = None
) -> str:
    """Import a file of your workspace into Ghidra and analyse it, once: the same bytes again reuse the analysis.
    ELF, PE, Mach-O and other formats Ghidra knows load as they are. A raw image (firmware, a flash dump, an
    embedded controller's code) needs `language`, a Ghidra language id such as ARM:LE:32:Cortex, ARM:LE:32:v7,
    AARCH64:LE:64:v8A, 8051:BE:16:default, x86:LE:32:default or RISCV:LE:32:default, and usually `base_address`
    (hex), where its first byte sits in memory. Analysis takes seconds to many minutes."""
    args: dict[str, JsonValue] = {
        "path": _path(path),
        "language": language,
        "base_address": base_address,
        "compiler": compiler,
    }
    return await spill(json.dumps(await _call("import", args, IMPORT_SECONDS), indent=1))


@aid.mcptool
async def ghidra_functions(path: str, query: str | None = None, offset: int = 0, limit: int = 200) -> str:
    """An imported program's functions, `address name (size)`, in address order; `query` keeps names holding it."""
    result = await _call("functions", {"path": _path(path), "query": query, "offset": offset, "limit": limit})
    return await spill(_rows(result, "functions", offset))


@aid.mcptool
async def ghidra_decompile(path: str, function: str) -> str:
    """One function of an imported program as C, by name or by an address inside it (hex)."""
    result = await _call("decompile", {"path": _path(path), "function": function})
    if not isinstance(result, dict):
        return str(result)
    return await spill(f"// {result.get('function')} at {result.get('entry')}\n{result.get('c')}")


@aid.mcptool
async def ghidra_disassemble(path: str, address: str, count: int = 60) -> str:
    """`count` instructions of an imported program from `address` (hex): address, bytes, instruction."""
    result = await _call("disassemble", {"path": _path(path), "address": address, "count": count})
    return await spill(_rows(result, "instructions"))


@aid.mcptool
async def ghidra_xrefs(path: str, address: str, limit: int = 200) -> str:
    """What refers to `address` (hex) in an imported program: each reference's source, its type (call, read,
    write, ...) and the function it is in. Follow a register, a port or a string to the code that uses it."""
    result = await _call("xrefs", {"path": _path(path), "address": address, "limit": limit})
    return await spill(_rows(result, "references"))


@aid.mcptool
async def ghidra_strings(path: str, query: str | None = None, offset: int = 0, limit: int = 200) -> str:
    """Strings Ghidra found in an imported program, with their addresses; `query` keeps those holding it."""
    result = await _call("strings", {"path": _path(path), "query": query, "offset": offset, "limit": limit})
    return await spill(_rows(result, "strings", offset))


@aid.mcptool
async def ghidra_memory(path: str) -> str:
    """An imported program's language, format, image base, function count and memory blocks with permissions."""
    return json.dumps(await _call("memory", {"path": _path(path)}), indent=1)
