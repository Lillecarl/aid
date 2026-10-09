"""Plugins for the catalog tests, on a test-only AID_PLUGINS_PATH."""

from __future__ import annotations

import aid


class Alpha(aid.Plugin):
    """First test plugin.

    This second paragraph is not part of the description.
    """

    version = "1.2.3"
    grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

    async def run(self, client: aid.Client) -> None: ...
