"""`python -m aid.catalog`: print the agents on the agents path as JSON.

The daemon runs this rather than importing agent modules itself: they are user code, and a fresh process also
sees a module edited since the last listing.
"""

from __future__ import annotations

import json

from aid.agents import agents_path, discover


def main() -> None:
    print(json.dumps(discover(agents_path()).describe()))


if __name__ == "__main__":
    main()
