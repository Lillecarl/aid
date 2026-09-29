"""The Claude Code hook command of a claude-tty session: `python .../aid/hook.py CONTROL SESSION EVENT`.

Claude runs it for each hook event aid's settings name (`claude_tty.hook_settings`), with the event's JSON on
stdin. It passes the event to the daemon as a `hook` request and prints the answer, if any, for Claude to act on.

Run as a file, not as `-m aid.hook`: importing the aid package costs 0.2 s (pydantic), and UserPromptSubmit runs
before every prompt. This file imports only json and pyzmq (0.04 s). Keep it that way; `tests/test_hook.py`
checks its request against `aid.protocol`.

It fails open: with no daemon, no worker, or any error, it prints nothing and exits 0, and Claude carries on as it
would without aid. Never exit 2: for most events that blocks what Claude was about to do.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import zmq

# Seconds to wait for the daemon. PermissionRequest waits for a person, bounded by the hook's own timeout.
QUICK_WAIT = 2.0
WAITING_EVENTS = frozenset({"PermissionRequest"})


def request(session: str, event: str, payload: object) -> dict[str, object]:
    return {"op": "hook", "id": uuid.uuid4().hex, "session": session, "event": event, "payload": payload}


def ask(control: str, message: dict[str, object], wait: float | None) -> object:
    """The `data` of the daemon's Done, or None for anything else."""
    if control.startswith("ipc://") and not Path(control.removeprefix("ipc://")).exists():
        return None
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    try:
        sock.connect(control)
        sock.send(json.dumps(message).encode())
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        if not poller.poll(None if wait is None else int(wait * 1000)):
            return None
        reply = json.loads(sock.recv())
        return reply.get("data") if reply.get("reply") == "done" and reply.get("id") == message["id"] else None
    finally:
        sock.close(linger=0)
        ctx.term()


def main() -> None:
    try:
        control, session, event = sys.argv[1:4]
        payload = json.load(sys.stdin)
        answer = ask(control, request(session, event, payload), None if event in WAITING_EVENTS else QUICK_WAIT)
        if answer is not None:
            print(json.dumps(answer))
    except Exception as error:
        print(f"aid hook: {type(error).__name__}: {error}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
