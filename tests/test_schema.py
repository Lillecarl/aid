from __future__ import annotations

import json
from pathlib import Path

from aid.schema import render

# The checkout's, which the sandbox copies in with the tests.
SCHEMA_FILE = Path(__file__).parents[1] / "web" / "src" / "lib" / "protocol.schema.json"


def test_the_committed_schema_matches_the_models() -> None:
    # On failure: python -m aid.schema > web/src/lib/protocol.schema.json, then npm run types in web/.
    assert SCHEMA_FILE.read_text() == render()


def test_what_the_page_reads_has_every_field_required() -> None:
    definitions = json.loads(render())["$defs"]
    status = definitions["SessionStatus"]
    assert set(status["required"]) == set(status["properties"])
    spec = definitions["AcpSpec"]
    assert "permission" not in spec["required"]  # What the page sends may leave defaults out.
