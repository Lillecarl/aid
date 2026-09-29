"""Permission requests a backend's agent waits on, until a person answers one from aid (`AnswerPermission`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio

from aid.protocol import PermissionDecider, PermissionDecision

if TYPE_CHECKING:
    from aid.protocol import PermissionRequest


@dataclass
class _Wait:
    request: PermissionRequest
    answered: anyio.Event = field(default_factory=anyio.Event)
    decision: PermissionDecision | None = None


class PermissionWaits:
    def __init__(self) -> None:
        self._waits: dict[str, _Wait] = {}

    async def wait(self, request: PermissionRequest) -> PermissionDecision:
        """The chosen option (None: cancelled) and who chose it. The caller bounds the wait."""
        wait = self._waits[request.request_id] = _Wait(request)
        try:
            await wait.answered.wait()
            if wait.decision is None:
                raise RuntimeError("answered with no decision")
            return wait.decision
        finally:
            del self._waits[request.request_id]

    def settle(self, request_id: str, option_id: str | None, by: PermissionDecider, plugin: str | None = None) -> bool:
        """False when no such request waits, or the option is not one of its own."""
        wait = self._waits.get(request_id)
        if wait is None or wait.answered.is_set():
            return False
        if option_id is not None and option_id not in {o.option_id for o in wait.request.options}:
            return False
        wait.decision = PermissionDecision(request_id=request_id, option_id=option_id, by=by, plugin=plugin)
        wait.answered.set()
        return True

    def answer(self, request_id: str, option_id: str | None, plugin: str | None = None) -> bool:
        """A person's answer, or with `plugin`, that plugin's."""
        return self.settle(
            request_id, option_id, PermissionDecider.PLUGIN if plugin else PermissionDecider.PERSON, plugin
        )

    def cancel_all(self) -> None:
        for request_id in list(self._waits):
            self.settle(request_id, None, PermissionDecider.CANCEL)
