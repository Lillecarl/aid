from __future__ import annotations

import json

from aid.schema import SCHEMA_FILE, render


def test_the_committed_schema_matches_the_models() -> None:
    # On failure: python -m aid.schema > web/src/lib/protocol.schema.json, then npm run types in web/.
    assert SCHEMA_FILE.read_text() == render()


def test_what_the_page_reads_has_every_field_required() -> None:
    definitions = json.loads(render())["$defs"]
    status = definitions["SessionStatus"]
    assert set(status["required"]) == set(status["properties"])
    spec = definitions["AcpSpec"]
    assert "permission" not in spec["required"]  # What the page sends may leave defaults out.
