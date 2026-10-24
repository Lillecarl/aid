"""An aid agent on DeepSeek, served by the bundled `deepseek` provider plugin, which reads DEEPSEEK_API_KEY from the
worker's environment. Not named `deepseek` itself: an earlier tier would shadow that plugin.

Put `examples` on AID_PLUGINS_PATH, or copy this directory into `~/.config/aid/plugins`, and it shows up as
`deep_seek`:

    aid agents
    aid new-py deepseek deep_seek
    aid prompt deepseek "What is aid?"
"""

from __future__ import annotations

import aid
from aid.agent import Agent


class DeepSeek(aid.AgentDefinition):
    """Short, plain answers from DeepSeek's flash model."""

    def build(self) -> Agent:
        return Agent("deepseek/deepseek-flash", instructions="Answer briefly and plainly.")
