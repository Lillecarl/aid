"""The other half of a duplicate provider name: this module loses to dup_a."""

from __future__ import annotations

import aid


class Dup(aid.InstructionsProvider):
    name = "dup"

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        return [aid.Instruction(name="dup-b", text="From dup_b.")]
