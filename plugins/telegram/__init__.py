"""The Telegram plugin: settled turn ends as Telegram messages, basic commands back.

aid plugin add telegram
AID_TELEGRAM_TOKEN=... AID_TELEGRAM_CHATS=... aid plugin run telegram
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aid
from aid.paths import default_paths

from .bot import MUTES_FILE, Mutes, read_config, run_loops

if TYPE_CHECKING:
    from aid.client import Client


class Telegram(aid.Plugin):
    """Report settled turn ends to Telegram, and take basic commands back."""

    version = "0.1.0"
    grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

    async def run(self, client: Client) -> None:
        config = read_config()
        mutes = await Mutes.load(default_paths().plugin_dir(self.plugin_name()) / MUTES_FILE)
        await run_loops(client, config, mutes)
