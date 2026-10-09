"""`python -m aid.plugin_catalog`: print the plugins on the plugins path as JSON.

The caller runs this rather than importing plugin modules itself when a fresh process should see a module
edited since the last listing. Discovery imports user code either way: never the daemon.
"""

from __future__ import annotations

import json

from aid.plugins import discover, plugins_path


def main() -> None:
    print(json.dumps(discover(plugins_path()).describe()))


if __name__ == "__main__":
    main()
