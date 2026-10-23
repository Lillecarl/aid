from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from aid.backends.acp import open_acp
from aid.backends.claude_sdk import open_sdk
from aid.backends.claude_tty import open_claude_tty
from aid.backends.native import open_native
from aid.spec import AcpSpec, ClaudeSdkSpec, ClaudeTtySpec, NativeSpec

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
        case NativeSpec():
            async with open_native(spec, state_dir) as backend:
                yield backend
        case ClaudeTtySpec():
            async with open_claude_tty(spec, state_dir) as backend:
                yield backend
        case ClaudeSdkSpec():
            async with open_sdk(spec, state_dir) as backend:
                yield backend
