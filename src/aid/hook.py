"""The Claude Code hook command of a claude-tty session: `python .../aid/hook.py CONTROL SESSION EVENT [HOOK]`.

Claude runs it for each hook event aid's settings name (`claude_tty.hook_settings`), with the event's JSON on
stdin. It asks the session's worker first, on its own hook socket (HOOK, `aid.worker.serve_hooks`), and falls
back to authenticated agent control when launch credentials exist. Manual hooks use the local administrator
socket. A hook from before the worker binds its socket, or after it exits, uses this fallback.

Run as a file, not as `-m aid.hook`: importing the aid package costs 0.2 s (pydantic), and UserPromptSubmit runs
before every prompt. This file imports only json and pyzmq (0.04 s). Keep it that way; `tests/test_hook.py`
checks its request against `aid.protocol`.

It fails open: with no daemon, no worker, or any error, it prints nothing and exits 0, and Claude carries on as it
would without aid. Never exit 2: for most events that blocks what Claude was about to do.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import zmq

# Seconds to wait for the daemon. PermissionRequest waits for a person, bounded by the hook's own timeout.
QUICK_WAIT = 2.0
WAITING_EVENTS = frozenset({"PermissionRequest"})


def request(session: str, event: str, payload: object) -> dict[str, object]:
    return {"op": "hook", "id": uuid.uuid4().hex, "session": session, "event": event, "payload": payload}


def ask(
    endpoint: str, message: dict[str, object], wait: float | None, credentials: dict[str, str] | None = None
) -> tuple[bool, object]:
    """Whether anyone answered, and the `data` of a Done. An explicit failure answers with nothing to say;
    only silence falls back to the next endpoint."""
    if endpoint.startswith("ipc://") and not Path(endpoint.removeprefix("ipc://")).exists():
        return False, None
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    monitor = None
    try:
        if credentials is not None:
            monitor = sock.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_FAILED_AUTH)
            sock.curve_serverkey = credentials["server_key"].encode()
            sock.curve_publickey = credentials["public_key"].encode()
            sock.curve_secretkey = credentials["secret_key"].encode()
            if urlsplit(endpoint).scheme == "wss":
                hostname = urlsplit(endpoint).hostname
                if not hostname:
                    return False, None
                sock.setsockopt_string(zmq.WSS_HOSTNAME, hostname)
                if trust := credentials.get("trust_pem"):
                    sock.setsockopt_string(zmq.WSS_TRUST_PEM, trust)
                else:
                    sock.setsockopt(zmq.WSS_TRUST_SYSTEM, 1)
        sock.connect(endpoint)
        sock.send(json.dumps(message).encode())
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        if monitor is not None:
            poller.register(monitor, zmq.POLLIN)
        ready = dict(poller.poll(None if wait is None else int(wait * 1000)))
        # A revoked key receives no daemon reply, including permission hooks that otherwise wait without a deadline.
        if monitor is not None and monitor in ready:
            return True, None
        if sock not in ready:
            return False, None
        reply = json.loads(sock.recv())
        if reply.get("id") != message["id"]:
            return False, None
        return (True, reply.get("data")) if reply.get("reply") == "done" else (True, None)
    finally:
        if monitor is not None:
            monitor.close(linger=0)
        sock.close(linger=0)
        ctx.term()


def main() -> None:
    try:
        args = sys.argv[1:]
        control, session, event = args[:3]
        hook_endpoint = args[3] if len(args) > 3 else None
        payload = json.load(sys.stdin)
        message = request(session, event, payload)
        wait = None if event in WAITING_EVENTS else QUICK_WAIT
        answered, answer = (False, None)
        if hook_endpoint is not None:
            answered, answer = ask(hook_endpoint, message, wait)
        if not answered:
            if connection := os.environ.get("AID_AGENT_CONNECTION"):
                credentials = json.loads(connection)
                answered, answer = ask(credentials["endpoint"], message, wait, credentials)
            elif not os.environ.get("AID_SESSION"):
                answered, answer = ask(control, message, wait)
        if answered and answer is not None:
            print(json.dumps(answer))
    except Exception as error:
        print(f"aid hook: {type(error).__name__}: {error}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
