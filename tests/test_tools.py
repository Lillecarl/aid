from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

import anyio
import pytest

import aid
from aid.agents import ENV_AGENTS_PATH, discover
from aid.paths import ENV_SESSION
from aid.protocol import Output, ToolCall, ToolInfo
from tests.conftest import py_spec
from tests.test_agents import AGENT_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.abc import ByteReceiveStream, ByteSendStream

    from aid.paths import Paths

pytestmark = pytest.mark.anyio

TIMEOUT = 30


async def test_discover_finds_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    described = discover([AGENT_DIR]).describe()["tools"]
    assert described == [
        {"name": "add", "description": "Add two numbers.", "module": "tooling"},
        {
            "name": "list_sessions",
            "description": "The aid sessions you can message: name, kind, whether its worker is running, and which "
            "one is you.",
            "module": "aid.builtin_tools",
        },
        {
            "name": "send_message",
            "description": "Send a message to another aid session, by its name from list_sessions. It wakes that "
            "session.",
            "module": "aid.builtin_tools",
        },
        {"name": "whoami", "description": "The aid session that calls this tool.", "module": "tooling"},
    ]


class Rpc:
    def __init__(self, send: ByteSendStream, receive: ByteReceiveStream) -> None:
        self._send, self._receive, self._buffer, self._id = send, receive, b"", 0

    async def notify(self, method: str) -> None:
        await self._send.send(json.dumps({"jsonrpc": "2.0", "method": method}).encode() + b"\n")

    async def _next(self) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            self._buffer += await self._receive.receive()
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    async def call(self, method: str, params: dict[str, Any]) -> Any:
        self._id += 1
        request = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        await self._send.send(json.dumps(request).encode() + b"\n")
        while (message := await self._next()).get("id") != self._id:
            pass
        return message["result"]

    async def notification(self, method: str) -> Any:
        while (message := await self._next()).get("method") != method:
            pass
        return message["params"]


async def test_mcp_server_serves_the_tools() -> None:
    env = {ENV_AGENTS_PATH: str(AGENT_DIR), ENV_SESSION: "s1"}
    command = [sys.executable, "-m", "aid.mcp_server"]
    with anyio.fail_after(TIMEOUT):
        async with await anyio.open_process(command, env={**os.environ, **env}) as process:
            assert process.stdin is not None
            assert process.stdout is not None
            rpc = Rpc(process.stdin, process.stdout)
            client = {"name": "test", "version": "0"}
            await rpc.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": client})
            await rpc.notify("notifications/initialized")
            listed = await rpc.call("tools/list", {})
            added = await rpc.call("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
            who = await rpc.call("tools/call", {"name": "whoami", "arguments": {}})
            await process.stdin.aclose()
    tools = {t["name"]: t for t in listed["tools"]}
    assert sorted(tools) == ["add", "list_sessions", "send_message", "whoami"]
    assert sorted(tools["add"]["inputSchema"]["required"]) == ["a", "b"]
    assert added["content"] == [{"type": "text", "text": "5"}]
    assert who["content"] == [{"type": "text", "text": "s1"}]


async def test_pydantic_ai_session_calls_the_tools(
    daemon: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_AGENTS_PATH, str(AGENT_DIR))
    spec = py_spec(tmp_path, "agents:tool_caller", AID_AGENTS_PATH=str(AGENT_DIR))
    with anyio.fail_after(TIMEOUT):
        async with aid.connect(daemon) as client:
            catalog = await client.agents()
            session = await client.create("caller", spec)
            events = [e async for e in session.stream("go")]
            without_tools = {"aid_tools": False, "target": "agents:every_tool_caller"}
            quiet = await client.create("quiet", spec.model_copy(update=without_tools))
            without = [e async for e in quiet.stream("go")]
    titles = {e.title for e in events if isinstance(e, ToolCall) and e.title}
    assert titles == {"add", "whoami"}
    assert isinstance(events[-1], Output)
    assert isinstance(events[-1].output, str)
    assert json.loads(events[-1].output) == {"add": 0, "whoami": "caller"}
    assert not [e for e in without if isinstance(e, ToolCall)]
    assert ToolInfo(name="add", description="Add two numbers.", module="tooling") in catalog.tools
