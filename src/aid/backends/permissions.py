"""Permission requests a backend's agent waits on, until a person answers one from aid (`AnswerPermission`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio

from aid.protocol import PermissionDecider

if TYPE_CHECKING:
    from aid.protocol import PermissionRequest


@dataclass
class _Wait:
    request: PermissionRequest
    answered: anyio.Event = field(default_factory=anyio.Event)
    option_id: str | None = None
    by: PermissionDecider = PermissionDecider.CANCEL


class PermissionWaits:
    def __init__(self) -> None:
        self._waits: dict[str, _Wait] = {}

    async def wait(self, request: PermissionRequest) -> tuple[str | None, PermissionDecider]:
        """The chosen option (None: cancelled) and who chose it. The caller bounds the wait."""
        wait = self._waits[request.request_id] = _Wait(request)
        try:
            await wait.answered.wait()
            return wait.option_id, wait.by
        finally:
            del self._waits[request.request_id]

    def settle(self, request_id: str, option_id: str | None, by: PermissionDecider) -> bool:
        """False when no such request waits, or the option is not one of its own."""
        wait = self._waits.get(request_id)
        if wait is None or wait.answered.is_set():
            return False
        if option_id is not None and option_id not in {o.option_id for o in wait.request.options}:
            return False
        wait.option_id, wait.by = option_id, by
        wait.answered.set()
        return True

    def answer(self, request_id: str, option_id: str | None) -> bool:
        return self.settle(request_id, option_id, PermissionDecider.PERSON)

    def cancel_all(self) -> None:
        for request_id in list(self._waits):
            self.settle(request_id, None, PermissionDecider.CANCEL)
