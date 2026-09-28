from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from aid.protocol import Output, SessionEvent

type Emit = Callable[[SessionEvent], Awaitable[None]]


class Backend(Protocol):
    """One agent session inside a worker. The worker runs at most one prompt at a time."""

    async def prompt(self, text: str, emit: Emit) -> Output: ...

    async def cancel(self) -> None: ...
