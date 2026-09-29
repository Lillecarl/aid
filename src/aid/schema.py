"""`python -m aid.schema`: the JSON Schema of what aid web's page reads and sends, from the pydantic models.

The page's TypeScript types (`web/src/lib/protocol.ts`) are generated from it (`npm run types` in web/), and both
files are committed: `tests/test_schema.py` fails when the models move and the schema does not, and the web build
fails when the schema moves and the types do not.

What the page reads is in serialization mode, where a field with a default is still always sent, so the types
make it required. What it sends is in validation mode, where a field with a default may be left out.
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, models_json_schema

from aid.protocol import (
    AgentCatalog,
    HistoryEntry,
    HistoryItem,
    HistoryPage,
    SessionEvent,
    SessionInfo,
    SessionStatus,
    SessionSummary,
)
from aid.spec import AgentSpec
from aid.web.files import Entry, FileView
from aid.web.zws import PageReply, PageRequest


class Received(BaseModel):
    """What aid web sends the page. Only the definitions matter; this model names them."""

    session_info: SessionInfo
    session_status: SessionStatus
    history_page: HistoryPage
    history_entry: HistoryEntry
    history_item: HistoryItem
    session_event: SessionEvent
    session_summary: SessionSummary
    agent_catalog: AgentCatalog
    file_entry: Entry
    file_view: FileView
    page_reply: PageReply


class Sent(BaseModel):
    """What the page sends aid web."""

    agent_spec: AgentSpec
    page_request: PageRequest


class _Untitled(GenerateJsonSchema):
    """No field titles: json2ts makes a type alias of each (Name1, Name2, …). Models keep theirs, their type names."""

    def field_title_should_be_set(self, schema: Any) -> bool:  # pydantic's own type for it is private
        return False

    def field_is_required(self, field: Any, total: bool) -> bool:
        """A Literal with a default is a union's tag (`op`, `kind`, …): a discriminated union refuses input without
        it, whatever the default says."""
        inner = field["schema"]
        if inner["type"] == "default" and inner["schema"]["type"] == "literal":
            return True
        return super().field_is_required(field, total)


def _tuples_as_items(node: Any) -> Any:
    """JSON Schema 2020's `prefixItems` as draft 7's `items` list, which is what json2ts reads."""
    if isinstance(node, dict):
        fixed = {key: _tuples_as_items(value) for key, value in cast("dict[str, Any]", node).items()}
        if "prefixItems" in fixed:
            fixed["items"] = fixed.pop("prefixItems")
        return fixed
    if isinstance(node, list):
        return [_tuples_as_items(value) for value in cast("list[Any]", node)]
    return node


def schema() -> dict[str, Any]:
    _, document = models_json_schema(
        [(Received, "serialization"), (Sent, "validation")], title="aid protocol", schema_generator=_Untitled
    )
    return _tuples_as_items(document)


def render() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    print(render(), end="")


if __name__ == "__main__":
    main()
