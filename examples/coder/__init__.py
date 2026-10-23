"""Aid agents that work on code in their session's directory: they read files, stage edits through pyedit, and run
processes through pyrun scripts (`aid.coding_tools`). `coder` runs on DeepSeek (DEEPSEEK_API_KEY), `haiku_coder`
on Claude Haiku (ANTHROPIC_API_KEY).

    aid new-py coder coder --cwd ~/src/project --permission ask
    aid prompt coder "Why does the test in tests/test_app.py fail?"

With `--permission ask`, each program a script starts, and each apply of staged edits, waits for a person in the
web UI or `aid answer`.
"""

from __future__ import annotations

import aid
from aid.agent import Agent

INSTRUCTIONS = """\
You work on the code in your working directory.

- List directories with `ls`, read files with `read`, find your way in big files with `outline`.
- Change files with `edit`, `write`, `apply_patch` or `rename_symbol`. They stage: `show_edits` shows the diff, and
  nothing reaches disk until `apply_edits`. Apply once the change is whole.
- Run programs with `python`: an async script using pyrun, with argv lists and no shell. Its report shows each
  process's exit and output, and ids to read the full output with `read`.
- Each prompt ends with a session-metadata part: context use and the time when the turn started.
- When the context grows heavy, compact it with `compact`, naming what the upcoming work needs.
- Show staged work with `show_edits` before `apply_edits`. After a refused apply or command, stop and report
  instead of retrying.
"""


class Coder(aid.AgentDefinition):
    """Reads, edits and runs code in its session's directory, on DeepSeek."""

    def build(self) -> Agent:
        return Agent("deepseek/deepseek-flash", instructions=INSTRUCTIONS, toolsets=[aid.coding_tools])


class HaikuCoder(aid.AgentDefinition):
    """Reads, edits and runs code in its session's directory, on Claude Haiku."""

    def build(self) -> Agent:
        return Agent("anthropic/claude-haiku-5-5", instructions=INSTRUCTIONS, toolsets=[aid.coding_tools])
