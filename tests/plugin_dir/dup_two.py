"""Second of two modules claiming one plugin name: this one loses."""

from __future__ import annotations

import aid


class Dup(aid.Plugin):
    name = "dup"

    async def run(self, client: aid.Client) -> None: ...
