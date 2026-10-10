"""Backend driving `claude -p` directly over NDJSON, so sessions get Remote Control.

Skeleton slice A: one process per prompt, resumed by the session id the `system init` line reports and aid
persists. Measured protocol facts live in `docs/sdk-backend.md`. Permissions, Remote Control, and CLI wiring are
later slices: a `can_use_tool` prompt raises loudly instead of hanging.
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
        completed = await anyio.run_process(
            sdk_argv(self._spec, resume=self._started.agent_session),
            input=payload.encode(),
            cwd=self._spec.cwd,
            env=agent_environment(self._spec.env, inherit=self._spec.inherit_env),
            check=False,
        )
        if completed.returncode != 0:
            tail = completed.stderr.decode(errors="replace").strip()[-2000:]
            raise AidError("sdk_failed", f"claude exited with {completed.returncode}: {tail}")
        return await self._parse(completed.stdout.decode(errors="replace"), emit)

    async def cancel(self) -> None:
        # No persistent process: a prompt's `run_process` ends with its scope. Abort mid-prompt is a later slice.
        return None

    async def _parse(self, stdout: str, emit: Emit) -> Output:
        """Assistant text out, the `system init` session id persisted; a permission prompt raises loudly."""
        chunks: list[str] = []
        stop_reason = "end_turn"
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                log.warning("sdk_malformed_line", line=line[:200])
                continue
            if not isinstance(message, dict):
                continue
            match message.get("type"):
                case "system" if message.get("subtype") == "init":
                    session_id = message.get("session_id")
                    if isinstance(session_id, str) and session_id:
                        await self._id_file.write_text(session_id)
                        self._started = self._started.model_copy(update={"agent_session": session_id})
                case "assistant":
                    envelope = message.get("message")
                    content = envelope.get("content", []) if isinstance(envelope, dict) else []
                    for block in content if isinstance(content, list) else []:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            chunks.append(block["text"])
                            await emit(TextDelta(text=block["text"]))
                case "control_request":
                    request = message.get("request", {})
                    if isinstance(request, dict) and request.get("subtype") == "can_use_tool":
                        raise AidError(
                            "not_implemented",
                            f"SDK permissions are a later slice (tool {request.get('tool_name')})",
                        )
                case "result":
                    if isinstance(message.get("subtype"), str):
                        stop_reason = message["subtype"]
                    if not chunks and isinstance(message.get("result"), str):
                        chunks.append(message["result"])
                case _:
                    continue
        return Output(output="".join(chunks), stop_reason=stop_reason)


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
