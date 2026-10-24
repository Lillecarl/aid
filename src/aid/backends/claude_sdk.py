"""Backend driving `claude -p` directly over NDJSON, so sessions get Remote Control.

Each prompt spawns one process, writes the user line plus a Remote Control
enable, and handles NDJSON lines as they arrive until the `result` line.
`can_use_tool` prompts answer through the session permission flow, and the
bridge URLs emit as a RemoteControl event. Measured protocol facts live in
`docs/sdk-backend.md`. CLI wiring is a later slice.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final, cast

import anyio
import structlog

from aid.backends.acp import choose_permission
from aid.backends.permissions import PermissionWaits
from aid.env import agent_environment
from aid.mcp import AID_TOOLS_RULE, MCP_CONFIG_FILE, claude_config, session_servers, write_private
from aid.protocol import (
    AidError,
    Output,
    PermissionChoice,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    RemoteControl,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    to_json,
)
from aid.spec import PermissionMode
from aid.transcript import items_from_entry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from anyio.abc import Process
    from pydantic import JsonValue

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


def sdk_argv(spec: ClaudeSdkSpec, *, resume: str | None, mcp_config: str | None = None) -> list[str]:
    """`claude -p` over stream-json NDJSON, resumed when a session id is persisted. With aid's tools, aid's MCP
    server comes in `mcp_config` and its tools need no permission."""
    # Both options are variadic: each goes before another option, or it would take what follows too.
    mcp = ["--mcp-config", mcp_config] if mcp_config else []
    aid = ["--allowedTools", AID_TOOLS_RULE] if spec.aid_tools else []
    argv = [
        *spec.command,
        "-p",
        *mcp,
        *aid,
        "--verbose",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
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

    def __init__(
        self, spec: ClaudeSdkSpec, id_file: anyio.Path, started: Started, mcp_config: str | None = None
    ) -> None:
        self._spec = spec
        self._id_file = id_file
        self._started = started
        self._mcp_config = mcp_config
        self._waits = PermissionWaits()
        self._process: Process | None = None
        """The running prompt's process, for `cancel` to kill; one prompt runs at a time."""
        self._rc_id: str | None = None
        """The pending Remote Control enable's request id; its control_response carries the bridge URLs."""
        self._streamed = False
        """Text deltas streamed the content block whose finished `assistant` line comes next."""

    def started(self) -> Started:
        return self._started

    async def prompt(self, text: str, emit: Emit) -> Output:
        payload = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
        argv = sdk_argv(self._spec, resume=self._started.agent_session, mcp_config=self._mcp_config)
        env = agent_environment(self._spec.env, inherit=self._spec.inherit_env)
        chunks: list[str] = []
        stop_reason = "end_turn"
        stderr_parts: list[bytes] = []
        async with await anyio.open_process(argv, cwd=self._spec.cwd, env=env) as process:
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise AidError("sdk_failed", "claude process has no stdio")
            self._process = process
            self._rc_id = uuid.uuid4().hex
            try:
                await process.stdin.send(payload.encode())
                await process.stdin.send(
                    (
                        json.dumps(
                            {
                                "type": "control_request",
                                "request": {"subtype": "remote_control", "enabled": True},
                                "request_id": self._rc_id,
                            }
                        )
                        + "\n"
                    ).encode()
                )

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
                                # The turn's `result`: terminal. Claude stays alive on open stdin, so stop
                                # reading instead of waiting for an EOF that never comes.
                                stop_reason = update
                                return
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
                    finally:
                        tg.cancel_scope.cancel()
                with contextlib.suppress(OSError):
                    await process.stdin.aclose()
                if errors:
                    raise errors[0]
                await process.wait()
                if process.returncode != 0:
                    tail = b"".join(stderr_parts).decode(errors="replace").strip()[-2000:]
                    raise AidError("sdk_failed", f"claude exited with {process.returncode}: {tail}")
            finally:
                self._process = None
                self._rc_id = None
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
        parsed: JsonValue
        try:
            parsed = json.loads(line)
        except ValueError:
            log.warning("sdk_malformed_line", line=line[:200])
            return None
        if not isinstance(parsed, dict):
            return None
        message = parsed
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                session_id = message.get("session_id")
                if isinstance(session_id, str) and session_id:
                    await self._id_file.write_text(session_id)
                    self._started = self._started.model_copy(update={"agent_session": session_id})
                return None
            case "stream_event":
                event = message.get("event")
                delta = event.get("delta") if isinstance(event, dict) else None
                if isinstance(delta, dict):
                    match delta.get("type"), delta.get("text"), delta.get("thinking"):
                        case "text_delta", str(text), _ if text:
                            self._streamed = True
                            await emit(TextDelta(text=text))
                        # Claude Haiku 5.5 sends thinking deltas with no text (display "updates"): nothing to show.
                        case "thinking_delta", _, str(thought) if thought:
                            await emit(ThoughtDelta(text=thought))
                        case _:
                            pass
                return None
            case "assistant" | "user" if message.get("parent_tool_use_id") is not None:
                return None  # A subagent's: its own conversation, as the transcript's sidechains are.
            case "user":
                # Tool results: the same entry shape as the transcript's (`tool_use_result`, snake case here).
                for item in items_from_entry(cast(dict[str, Any], message)):
                    if isinstance(item, ToolCall):
                        await emit(item)
                return None
            case "assistant":
                # One line per finished content block, after that block's stream events: its text is the turn's
                # output, and reaches the stream here only if no deltas carried it (an older Claude Code). A tool
                # call goes out as the transcript path turns it.
                for item in items_from_entry(cast(dict[str, Any], message)):
                    if isinstance(item, ToolCall):
                        await emit(item)
                envelope = message.get("message")
                content: JsonValue = envelope.get("content", []) if isinstance(envelope, dict) else []
                for block in content if isinstance(content, list) else []:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    text = block.get("text")
                    if not isinstance(text, str):
                        continue
                    chunks.append(text)
                    if not self._streamed:
                        await emit(TextDelta(text=text))
                self._streamed = False
                return None
            case "control_request":
                request = message.get("request", {})
                if isinstance(request, dict) and request.get("subtype") == "can_use_tool":
                    await self._answer_tool(message, request, emit, respond)
                return None
            case "control_response":
                await self._handle_remote_response(message, emit)
                return None
            case "result":
                if not chunks:
                    fallback = message.get("result")
                    if isinstance(fallback, str):
                        chunks.append(fallback)
                subtype = message.get("subtype")
                return subtype if isinstance(subtype, str) else "end_turn"
            case _:
                return None

    async def _answer_tool(
        self,
        message: dict[str, JsonValue],
        request: dict[str, JsonValue],
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

    async def _handle_remote_response(self, message: dict[str, JsonValue], emit: Emit) -> None:
        """The bridge's answer to this turn's Remote Control enable: emit its URLs, or log why not.

        A failed enable never fails the turn: the phone link is additive, the work stands without it.
        """
        response = message.get("response")
        if (
            self._rc_id is None
            or not isinstance(response, dict)
            or response.get("request_id") != self._rc_id
            or response.get("subtype") != "success"
        ):
            return
        self._rc_id = None
        urls = response.get("response")
        session_url = urls.get("session_url") if isinstance(urls, dict) else None
        connect_url = urls.get("connect_url") if isinstance(urls, dict) else None
        if not isinstance(session_url, str) or not session_url:
            log.warning("sdk_remote_failed", detail=str(response.get("response"))[:200])
            return
        await emit(
            RemoteControl(
                session_url=session_url,
                connect_url=connect_url if isinstance(connect_url, str) else "",
            )
        )


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
    mcp_config: str | None = None
    if servers := session_servers(spec, state_dir.name):
        mcp_file = state_dir / MCP_CONFIG_FILE
        await write_private(mcp_file, json.dumps(claude_config(servers)), 0o600)
        mcp_config = str(mcp_file)
    yield ClaudeSdkBackend(spec, id_file, started, mcp_config)
