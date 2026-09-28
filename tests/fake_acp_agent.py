"""A deterministic ACP agent for tests. Behaviour depends on the prompt text."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, cast

import acp
import anyio
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    TextContentBlock,
    ToolCallUpdate,
)

CHUNKS = 50


class FakeAgent:
    def __init__(self) -> None:
        self.conn: Any = None
        self.cancelled = anyio.Event()
        self.turns = 0

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version, agent_capabilities=AgentCapabilities(load_session=True)
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid.uuid4().hex)

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> LoadSessionResponse:
        await self.say(session_id, "replayed history")
        return LoadSessionResponse()

    async def say(self, session_id: str, text: str) -> None:
        await self.conn.session_update(session_id=session_id, update=acp.update_agent_message_text(text))

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        text = "".join(block.text for block in prompt if isinstance(block, TextContentBlock))
        self.turns += 1
        match text:
            case "count":
                for i in range(CHUNKS):
                    await self.say(session_id, f"{i} ")
            case "permission":
                response = await self.conn.request_permission(
                    session_id=session_id,
                    tool_call=ToolCallUpdate(tool_call_id="t1", title="rm -rf /"),
                    options=[
                        PermissionOption(option_id="yes", name="Yes", kind="allow_once"),
                        PermissionOption(option_id="no", name="No", kind="reject_once"),
                    ],
                )
                outcome = response.outcome
                await self.say(session_id, outcome.option_id if isinstance(outcome, AllowedOutcome) else "cancelled")
            case "slow":
                self.cancelled = anyio.Event()
                await self.say(session_id, "waiting")
                await self.cancelled.wait()
                return PromptResponse(stop_reason="cancelled")
            case "env":
                await self.say(session_id, f"{os.environ.get('AID_TEST_VAR')} {Path.cwd()}")
            case "pid":
                await self.say(session_id, str(os.getpid()))
            case _:
                await self.say(session_id, f"echo: {text}")
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self.cancelled.set()


async def main() -> None:
    await acp.run_agent(cast("acp.Agent", FakeAgent()))


if __name__ == "__main__":
    anyio.run(main)
