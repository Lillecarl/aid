"""Providers for the catalog tests, on a test-only AID_PLUGINS_PATH."""

from __future__ import annotations

import aid


class Base(aid.InstructionsProvider):
    """Abstract: no load, so not a provider."""


class Briefing(Base):
    """A short project brief.

    This second paragraph is not part of the description.
    """

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        return [aid.Instruction(name="brief", text="Ship it.")]


class Named(aid.InstructionsProvider):
    name = "custom-provider"

    def load(self, context: aid.InstructionsContext) -> list[aid.Instruction]:
        return [aid.Instruction(name="named", text="Named text.")]
