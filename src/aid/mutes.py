"""Runtime mutes: sessions that stay silent, chosen per chat plugin; everything else notifies.

Kept on disk beside the plugin's registration, so a restart keeps them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio

from aid.protocol import AidError

if TYPE_CHECKING:
    from pathlib import Path


class Mutes:
    """Sessions that stay silent, chosen at runtime; everything else notifies. Kept on disk so a bot
    restart keeps them."""

    def __init__(self, path: Path, muted: frozenset[str] = frozenset()) -> None:
        self._path = path
        self._muted = muted

    @classmethod
    async def load(cls, path: Path) -> Mutes:
        try:
            muted = frozenset(json.loads(await anyio.Path(path).read_text()).get("muted", []))
        except FileNotFoundError:
            return cls(path)
        except ValueError, AttributeError:
            raise AidError("bad_mutes", f"{path} holds no mute list") from None
        if not all(isinstance(name, str) for name in muted):
            raise AidError("bad_mutes", f"{path} holds no mute list")
        return cls(path, muted)

    def __contains__(self, session: object) -> bool:
        return session in self._muted

    def names(self) -> list[str]:
        return sorted(self._muted)

    async def mute(self, session: str) -> None:
        self._muted |= {session}
        await self._save()

    async def unmute(self, session: str) -> None:
        self._muted -= {session}
        await self._save()

    async def _save(self) -> None:
        path = anyio.Path(self._path)
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_text(json.dumps({"muted": sorted(self._muted)}))
