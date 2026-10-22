"""A plugin whose gate raises: broken, not declined."""

from __future__ import annotations

import aid


class BrokenGate(aid.Plugin):
    @classmethod
    def available(cls) -> str | None:
        raise RuntimeError("gate exploded")

    async def run(self, client: aid.Client) -> None: ...
