"""Backend driving `claude -p` directly over NDJSON, so sessions get Remote Control.

Slice B2 answers `can_use_tool` prompts through the session permission flow, mirroring the ACP backend's
semantics (`choose_permission`, policy at once, ask waits, cancel settles). Measured protocol facts live in
`docs/sdk-backend.md`. Remote Control and CLI wiring are later slices.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final

import anyio
import structlog

from aid.backends.acp import choose_permission
from aid.backends.permissions import PermissionWaits
from aid.env import agent_environment
from aid.protocol import (
    AidError,
    Output,
    PermissionChoice,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    Started,
    TextDelta,
    to_json,
)
from aid.spec import PermissionMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from anyio.abc import Process

    from aid.backends.base import Emit
    from aid.spec import ClaudeSdkSpec

log = structlog.get_logger(__name__)

SESSION_ID_FILE = "claude-session-id"
"""The `system init` session id, stable across processes: `--resume` continues it."""

SDK_OPTIONS: Final = [
    PermissionChoice(option_id="allow", name="Yes", kind="allow_once"),
    PermissionChoice(option_id="deny", name="No", kind="reject_once"),
]
"""Answers to a `can_use_tool` prompt. Kinds mirror ACP's, so `choose_permission` picks the policy answer."""


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
        self._waits = PermissionWaits()
        self._process: Process | None = None
        """The running prompt's process, for `cancel` to kill; one prompt runs at a time."""

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
            self._process = process
            try:
                await process.stdin.send(payload.encode())

                async def respond(request_id: str, allow: bool) -> None:
                    assert process.stdin is not None
                    behavior: dict[str, str] = (
                        {"behavior": "allow"} if allow else {"behavior": "deny", "message": "Refused in aid."}
                    )
                    response = {
                        "type": "control_response",
                        "response": {"subtype": "success", "request_id": request_id, "response": behavior},
                    }
                    await process.stdin.send((json.dumps(response) + "\n").encode())

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
                            update = await self._handle_line(raw.decode(errors="replace"), emit, chunks, respond)
                            if update is not None:
                                stop_reason = update
                    if buffer.strip():
                        update = await self._handle_line(buffer.decode(errors="replace"), emit, chunks, respond)
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
            finally:
                self._process = None
                self._waits.cancel_all()
        return Output(output="".join(chunks), stop_reason=stop_reason)

    async def cancel(self) -> None:
        # Settle pending asks as cancelled and kill the running prompt's process, if any.
        self._waits.cancel_all()
        if (process := self._process) is not None:
            with contextlib.suppress(OSError):
                process.kill()

    def answer_permission(
        self,
        request_id: str,
        option_id: str | None,
        plugin: str | None = None,
        text: str | None = None,
        supervisor: str | None = None,
    ) -> bool:
        return self._waits.answer(request_id, option_id, plugin, text, supervisor)

    async def _handle_line(
        self,
        line: str,
        emit: Emit,
        chunks: list[str],
        respond: Callable[[str, bool], Awaitable[None]],
    ) -> str | None:
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
                    await self._answer_tool(message, request, emit, respond)
                return None
            case "result":
                if not chunks and isinstance(message.get("result"), str):
                    chunks.append(message["result"])
                subtype = message.get("subtype")
                return subtype if isinstance(subtype, str) else "end_turn"
            case _:
                return None

    async def _answer_tool(
        self,
        message: dict[str, Any],
        request: dict[str, Any],
        emit: Emit,
        respond: Callable[[str, bool], Awaitable[None]],
    ) -> None:
        """Answer one `can_use_tool` prompt: policy at once, ask waits, then reply over stdin as control_response.

        Mirrors the ACP backend: ALLOW answers allow, DENY refuses, ASK emits a PermissionRequest and waits for an
        answer, refused after `permission_timeout`. A cancelled wait raises loudly; the turn ends.
        """
        tool_name = request.get("tool_name")
        title = request.get("action_description") or request.get("description") or request.get("display_name")
        tool_use_id = request.get("tool_use_id")
        permission = PermissionRequest(
            request_id=uuid.uuid4().hex,
            tool_call_id=tool_use_id if isinstance(tool_use_id, str) else "",
            tool_name=tool_name if isinstance(tool_name, str) else None,
            title=title if isinstance(title, str) else None,
            input=to_json(request.get("input")),
            options=list(SDK_OPTIONS),
        )
        await emit(permission)
        if self._spec.permission is PermissionMode.ASK:
            refused = choose_permission(PermissionMode.DENY, permission.options)
            decision = PermissionDecision(
                request_id=permission.request_id, option_id=refused, by=PermissionDecider.TIMEOUT
            )
            with anyio.move_on_after(self._spec.permission_timeout):
                decision = await self._waits.wait(permission)
        else:
            chosen = choose_permission(self._spec.permission, permission.options)
            decision = PermissionDecision(
                request_id=permission.request_id, option_id=chosen, by=PermissionDecider.POLICY
            )
        log.info(
            "permission_decided",
            tool=permission.tool_name,
            title=permission.title,
            option=decision.option_id,
            by=decision.by,
        )
        await emit(decision)
        if decision.by is PermissionDecider.CANCEL:
            raise AidError("sdk_cancelled", "turn cancelled during permission wait")
        request_id = message.get("request_id")
        if isinstance(request_id, str) and request_id:
            await respond(request_id, decision.option_id == "allow")
        else:
            log.warning("sdk_permission_without_request_id", tool=permission.tool_name)


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
