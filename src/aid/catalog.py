"""`python -m aid.catalog [--path DIR]...`: print the plugins path's catalog as JSON: plugins, agents, instruction
providers and tools. `--path` replaces the standard tiers.

The daemon runs this rather than importing plugins itself: they are user code, and a fresh process also sees a
plugin edited since the last listing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import anyio

from aid.discovery import Catalog, discover
from aid.plugins import plugins_path
from aid.protocol import AgentCatalog, AidError

if TYPE_CHECKING:
    from collections.abc import Sequence

TIMEOUT: Final = 60.0


def session_catalog() -> Catalog:
    """What a session is offered."""
    return discover(plugins_path())


async def listing(dirs: Sequence[Path] | None = None) -> AgentCatalog:
    """The catalog from a fresh process, for a caller that must not import plugins. `dirs` replaces the tiers."""
    command = [sys.executable, "-m", "aid.catalog"]
    for directory in dirs or ():
        command += ["--path", str(directory)]
    with anyio.fail_after(TIMEOUT):
        result = await anyio.run_process(command, check=False)
    if result.returncode != 0:
        raise AidError("catalog_failed", result.stderr.decode(errors="replace").strip()[-2000:])
    return AgentCatalog.model_validate_json(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aid.catalog")
    parser.add_argument("--path", action="append", type=Path, help="a plugins directory, instead of the tiers")
    args = parser.parse_args()
    print(json.dumps(discover(args.path if args.path is not None else plugins_path()).describe()))


if __name__ == "__main__":
    main()
