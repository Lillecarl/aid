"""`python -m aid.catalog`: print what sessions are offered as JSON: the agents path, plus the tools and instruction
providers of modules on the plugins path.

The daemon runs this rather than importing agent modules itself: they are user code, and a fresh process also
sees a module edited since the last listing.
"""

from __future__ import annotations

import json

from aid.agents import Catalog, agents_path, discover
from aid.plugins import discover as discover_plugins
from aid.plugins import plugins_path


def session_catalog() -> Catalog:
    """What a session is offered. On a name clash the agents path wins: it is the established source."""
    catalog = discover(agents_path())
    plugins = discover_plugins(plugins_path())
    catalog.problems.extend(plugins.problems)
    for name, tool in plugins.tools.items():
        catalog.tools.setdefault(name, tool)
    for name, provider in plugins.providers.items():
        catalog.providers.setdefault(name, provider)
    return catalog


def main() -> None:
    print(json.dumps(session_catalog().describe()))


if __name__ == "__main__":
    main()
