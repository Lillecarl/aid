"""A plugin that registers under an explicit name."""

from __future__ import annotations

import aid


class Named(aid.Plugin):
    name = "custom-plugin"

    async def run(self, client: aid.Client) -> None: ...
