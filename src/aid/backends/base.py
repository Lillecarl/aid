from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aid.protocol import Output, PaneAddress, PaneView, SessionEvent, Started

type Emit = Callable[[SessionEvent], Awaitable[None]]


class Backend(Protocol):
    """One agent session inside a worker. The worker runs at most one prompt at a time."""

    def started(self) -> Started:
        """What the backend knows about its agent once open; the worker sends it in Hello."""
        ...

    async def prompt(self, text: str, emit: Emit) -> Output: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class ScreenBackend(Backend, Protocol):
    """A backend with a terminal a person can look at."""

    async def screen(self, *, stylesheet: bool, since: int | None, wait: float) -> PaneView: ...

    def pane_address(self) -> PaneAddress: ...
