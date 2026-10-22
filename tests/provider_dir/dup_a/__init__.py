"""One half of a duplicate provider name: the first module wins."""

from __future__ import annotations

import aid


class Dup(aid.InstructionsProvider):
    name = "dup"

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        return [aid.Instruction(name="dup-a", text="From dup_a.")]
