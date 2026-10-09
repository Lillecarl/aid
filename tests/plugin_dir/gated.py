"""A plugin that gates itself out: declined, not broken."""

from __future__ import annotations

import aid


class Gated(aid.Plugin):
    @classmethod
    def available(cls) -> str | None:
        return "needs a token"

    async def run(self, client: aid.Client) -> None: ...
