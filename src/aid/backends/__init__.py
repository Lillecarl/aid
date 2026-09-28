from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from aid.backends.acp import open_acp
from aid.backends.claude_tty import open_claude_tty
from aid.backends.pydantic_ai import open_pydantic_ai
from aid.spec import AcpSpec, ClaudeTtySpec, PydanticAISpec

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import anyio

    from aid.backends.base import Backend
    from aid.spec import AgentSpec


@asynccontextmanager
async def open_backend(spec: AgentSpec, state_dir: anyio.Path) -> AsyncGenerator[Backend]:
    match spec:
        case AcpSpec():
            async with open_acp(spec, state_dir) as backend:
                yield backend
        case PydanticAISpec():
            async with open_pydantic_ai(spec, state_dir) as backend:
                yield backend
        case ClaudeTtySpec():
            async with open_claude_tty(spec, state_dir) as backend:
                yield backend
