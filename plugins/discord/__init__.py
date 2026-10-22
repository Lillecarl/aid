"""The Discord plugin: one channel per session in a guild, settled turn ends posted there, commands back.

Runs hosted in the daemon, reading `AID_DISCORD_TOKEN` and `AID_DISCORD_GUILD` from its environment
(category `AID_DISCORD_CATEGORY` optional), with mutes and the channel map in its plugin dir:

    aid plugin add discord

Or outside it, with the same environment in the calling shell:

    AID_DISCORD_TOKEN=... AID_DISCORD_GUILD=... aid plugin run discord
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aid
from aid.mutes import Mutes
from aid.paths import default_paths

from .channels import CHANNELS_FILE, ChannelMap
from .serve import MUTES_FILE, read_config, run_loops

if TYPE_CHECKING:
    from aid.client import Client


class Discord(aid.Plugin):
    """Session channels with settled turn ends, commands back."""

    grants = frozenset({aid.Grant.READ, aid.Grant.MESSAGE})

    async def run(self, client: Client) -> None:
        config = read_config()
        plugin_dir = default_paths().plugin_dir(self.plugin_name())
        mutes = await Mutes.load(plugin_dir / MUTES_FILE)
        channel_map = await ChannelMap.load(plugin_dir / CHANNELS_FILE)
        await run_loops(client, config, mutes, channel_map)
