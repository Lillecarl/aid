"""The permission classifier: the model layer between the read-only rules and the supervisor. Each request is a
fresh turn (`fresh_turns`) on the same cached instructions; it sees the request alone, never a conversation, and
answers allow, deny with a reason, or escalate to the supervisor's route."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel

import aid
from aid.agent import Agent

if TYPE_CHECKING:
    from aid.protocol import PermissionRequest

MODEL: Final = "deepseek/deepseek-flash"
INPUT_CHARS: Final = 6000
"""Characters of a request's input the classifier reads; past them it should escalate."""

INSTRUCTIONS: Final = """\
You classify one tool call an AI coding agent wants to run on a person's computer. You see only the call: no \
conversation, no task. Answer with a verdict and a one-sentence reason.

Tools, by the name the call carries:
- `shell`, `background`: a shell command line.
- `python`: a program that the agent's Python script starts; the input is its argv. Any program can appear here: \
judge the program, not the tool's name.
- `python:read`, `python:write`: a file the script opens or changes outside the working directory.
- `apply_edits`: edits the agent staged to files.

allow: the call is contained and low-risk whatever the task is.
- Reading anything that is not a secret: files, listings, logs, `--help`, version and status queries, searches.
- Evaluating or building without installing: `nix eval`, `nix build --no-link`, compilers, test runners, linters, \
formatters, type checkers on the project.
- Editing or creating files inside the session's working directory, and in /tmp.
- Version control that stays local and keeps history: status, diff, log, add, commit, branch creation, jj new, jj \
describe, jj split.
deny: the call is clearly harmful or forbidden; the reason tells the agent what to do instead.
- Reading or printing secrets: private keys, tokens, password stores, `.env` files, credential files.
- Destroying work without a plan: `rm -rf` with globs or variables over source, `git reset --hard`, `git clean`, \
`jj abandon` of others' work. Say: use the remove tool, which shows its plan.
- Disabling safety: `--no-verify`, force pushes, chmod -R 777, editing shell profiles or the agent's own config.
escalate: everything else, and anything you are unsure of.
- Reaching other machines or people: ssh, network writes, pushing, publishing, sending messages, deploying.
- Privilege: sudo, systemctl start/stop/restart, system configuration switches.
- Deletes that may be fine but need the task to judge.
- Long or obfuscated scripts, encoded payloads, inputs cut off.

Never allow what you would hesitate over: escalation costs a little time, a wrong allow can cost the person's work.
"""


class ClassifierVerdict(BaseModel):
    verdict: Literal["allow", "deny", "escalate"]
    reason: str


class Classifier(aid.AgentDefinition):
    """Classifies one permission request at a time, with no memory: allow, deny or escalate to the supervisor."""

    instruction_providers = ()
    fresh_turns = True

    def build(self) -> Agent:
        return Agent(MODEL, instructions=INSTRUCTIONS, output_type=ClassifierVerdict)


def classifier_prompt(session: str, cwd: str, request: PermissionRequest) -> str:
    """The one request, as the classifier reads it."""
    rendered = json.dumps(request.input, indent=1, default=str)
    if len(rendered) > INPUT_CHARS:
        rendered = f"{rendered[:INPUT_CHARS]}\n… {len(rendered) - INPUT_CHARS} more characters (cut off)"
    return (
        f"Session {session!r}, working directory {cwd}.\n"
        f"Tool: {request.tool_name or 'a question'}\n"
        f"Title: {request.title or '(none)'}\n"
        f"Input:\n{rendered}"
    )
