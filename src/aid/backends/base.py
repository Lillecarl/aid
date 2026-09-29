from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import JsonValue

    from aid.protocol import Activity, HistoryItem, Output, PaneAddress, PaneView, SessionEvent, Started

type Emit = Callable[[SessionEvent], Awaitable[None]]
type Record = Callable[[str, HistoryItem], Awaitable[None]]
"""Records an entry of a turn nobody sent through aid, under the turn id the backend chose."""
type Report = Callable[[Activity], Awaitable[None]]
"""Tells the daemon what the agent is doing."""


class Backend(Protocol):
    """One agent session inside a worker. The worker runs at most one prompt at a time."""

    def started(self) -> Started:
        """What the backend knows about its agent once open; the worker sends it in Hello."""
        ...

    async def prompt(self, text: str, emit: Emit) -> Output: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class FollowingBackend(Backend, Protocol):
    """A backend whose agent also takes turns that do not come through aid.

    `runtime_checkable` matches by method name alone: keep `follow` unique among backends."""

    async def follow(self, record: Record, report: Report) -> None:
        """Runs for the worker's life, recording those turns. It returns when the agent has gone, which ends the
        worker."""
        ...


@runtime_checkable
class HookBackend(Backend, Protocol):
    """A backend whose agent calls aid from its hooks (`aid/hook.py`)."""

    async def hook(self, event: str, payload: JsonValue) -> JsonValue:
        """What the hook prints for the agent to act on; None for nothing."""
        ...


@runtime_checkable
class PermissionBackend(Backend, Protocol):
    """A backend whose agent can wait on a person's answer to a PermissionRequest."""

    def answer_permission(self, request_id: str, option_id: str | None) -> bool:
        """False when no such request waits, or the option is not one of its own."""
        ...


@runtime_checkable
class ScreenBackend(Backend, Protocol):
    """A backend with a terminal a person can look at."""

    async def screen(self, *, stylesheet: bool, since: int | None, wait: float) -> PaneView: ...

    def pane_address(self) -> PaneAddress: ...
