"""Backend driving `claude -p` directly over NDJSON, so sessions get Remote Control.

Slice B1 streams stdout live: each prompt spawns one process, writes the user line, and handles NDJSON lines as
they arrive, until the `result` line. Measured protocol facts live in `docs/sdk-backend.md`. Permissions, Remote
Control, and CLI wiring are later slices: a `can_use_tool` prompt raises loudly instead of hanging.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import structlog

from aid.env import agent_environment
from aid.protocol import AidError, Output, Started, TextDelta

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from aid.backends.base import Emit
    from aid.spec import ClaudeSdkSpec

log = structlog.get_logger(__name__)

SESSION_ID_FILE = "claude-session-id"
"""The `system init` session id, stable across processes: `--resume` continues it."""


def sdk_argv(spec: ClaudeSdkSpec, *, resume: str | None) -> list[str]:
    """`claude -p` over stream-json NDJSON, resumed when a session id is persisted."""
    argv = [
        *spec.command,
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--model",
        spec.model,
    ]
    if resume is not None:
        argv += ["--resume", resume]
    return argv


class ClaudeSdkBackend:
    """One SDK session: each prompt runs one `claude -p` process to its `result` line."""

    def __init__(self, spec: ClaudeSdkSpec, id_file: anyio.Path, started: Started) -> None:
        self._spec = spec
        self._id_file = id_file
        self._started = started

    def started(self) -> Started:
        return self._started

    async def prompt(self, text: str, emit: Emit) -> Output:
        payload = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
        argv = sdk_argv(self._spec, resume=self._started.agent_session)
        env = agent_environment(self._spec.env, inherit=self._spec.inherit_env)
        chunks: list[str] = []
        stop_reason = "end_turn"
        stderr_parts: list[bytes] = []
        async with await anyio.open_process(argv, cwd=self._spec.cwd, env=env) as process:
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise AidError("sdk_failed", "claude process has no stdio")
            await process.stdin.send(payload.encode())
            await process.stdin.aclose()

            async def drain_stderr() -> None:
                assert process.stderr is not None
                async for piece in process.stderr:
                    stderr_parts.append(piece)

            async def read_stdout() -> None:
                nonlocal stop_reason
                assert process.stdout is not None
                buffer = b""
                async for piece in process.stdout:
                    buffer += piece
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        if (update := await self._handle_line(raw.decode(errors="replace"), emit, chunks)) is not None:
                            stop_reason = update
                if buffer.strip():
                    update = await self._handle_line(buffer.decode(errors="replace"), emit, chunks)
                    if update is not None:
                        stop_reason = update

            errors: list[BaseException] = []
            async with anyio.create_task_group() as tg:
                tg.start_soon(drain_stderr)
                try:
                    await read_stdout()
                except BaseException as error:
                    errors.append(error)
                    tg.cancel_scope.cancel()
            if errors:
                raise errors[0]
            await process.wait()
            if process.returncode != 0:
                tail = b"".join(stderr_parts).decode(errors="replace").strip()[-2000:]
                raise AidError("sdk_failed", f"claude exited with {process.returncode}: {tail}")
        return Output(output="".join(chunks), stop_reason=stop_reason)

    async def cancel(self) -> None:
        # No persistent process: a prompt's process ends with its scope. Abort mid-prompt is a later slice.
        return None

    async def _handle_line(self, line: str, emit: Emit, chunks: list[str]) -> str | None:
        """Handle one NDJSON line as it arrives; returns a new stop reason for a result line, else None."""
        if not line.strip():
            return None
        try:
            message = json.loads(line)
        except ValueError:
            log.warning("sdk_malformed_line", line=line[:200])
            return None
        if not isinstance(message, dict):
            return None
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                session_id = message.get("session_id")
                if isinstance(session_id, str) and session_id:
                    await self._id_file.write_text(session_id)
                    self._started = self._started.model_copy(update={"agent_session": session_id})
                return None
            case "assistant":
                envelope = message.get("message")
                content = envelope.get("content", []) if isinstance(envelope, dict) else []
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        chunks.append(block["text"])
                        await emit(TextDelta(text=block["text"]))
                return None
            case "control_request":
                request = message.get("request", {})
                if isinstance(request, dict) and request.get("subtype") == "can_use_tool":
                    raise AidError(
                        "not_implemented",
                        f"SDK permissions are a later slice (tool {request.get('tool_name')})",
                    )
                return None
            case "result":
                if not chunks and isinstance(message.get("result"), str):
                    chunks.append(message["result"])
                subtype = message.get("subtype")
                return subtype if isinstance(subtype, str) else "end_turn"
            case _:
                return None


@asynccontextmanager
async def open_sdk(spec: ClaudeSdkSpec, state_dir: anyio.Path) -> AsyncGenerator[ClaudeSdkBackend]:
    id_file = state_dir / SESSION_ID_FILE
    session_id = (await id_file.read_text()).strip() or None if await id_file.exists() else None
    started = Started(
        pid=os.getpid(),
        agent_session=session_id,
        resumed=session_id is not None,
        agent="claude-sdk",
        model=spec.model,
    )
    yield ClaudeSdkBackend(spec, id_file, started)
