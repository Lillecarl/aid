"""`usertee`: copy what flows through a pipe to the person watching an aid session, and pass it on unchanged.

    long-command 2>&1 | usertee | tail -n 5

Agents filter output (`tail`, `grep`, `head`) and so hide a command's progress from the person; piping it through
usertee first shows all of it live in the session's view. The copy is not recorded: the daemon keeps only the
newest `aid.daemon.LIVE_LIMIT` characters, for a page opened mid-command.

Run as a file (`launcher.install_usertee` puts a `usertee` shim on every worker's PATH), never imported through the
aid package: importing aid costs 0.2 s (pydantic), and this sits inside pipelines. It imports zmq only, like
`aid/hook.py`. The session is AID_SESSION; its worker's socket is `hook-<session>.sock` in AID_RUNTIME_DIR
(`aid.paths.hook_socket`), where `aid.worker` takes {"op": "usertee", "text": ...} and answers nothing.

It fails open: with no session, no worker, a full queue or any error, it is `cat`. Stdout gets every byte first,
unchanged, and never waits on the socket.
"""

from __future__ import annotations

import codecs
import json
import os
import signal
import sys
from pathlib import Path

import zmq

OP = "usertee"
READ = 65536
SEND_MAX = 16384
"""Characters per message: a big read goes out as several."""
QUEUE = 256
"""Messages zmq holds for a slow or absent worker before sends drop."""
LINGER_MS = 500
"""How long exit waits to hand queued copies over: bounded, so a dead worker delays a pipeline at most this."""


def connect() -> zmq.Socket[bytes] | None:
    session, runtime = os.environ.get("AID_SESSION"), os.environ.get("AID_RUNTIME_DIR")
    if not session or not runtime:
        return None
    path = Path(runtime) / f"hook-{session}.sock"
    if not path.exists():
        return None
    sock = zmq.Context().socket(zmq.DEALER)
    sock.setsockopt(zmq.SNDHWM, QUEUE)
    sock.setsockopt(zmq.LINGER, LINGER_MS)
    sock.connect(f"ipc://{path}")
    return sock


def copy(sock: zmq.Socket[bytes] | None, text: str) -> zmq.Socket[bytes] | None:
    """Send `text` without waiting; the socket, or None once sending failed for good."""
    if sock is None:
        return None
    try:
        for start in range(0, len(text), SEND_MAX):
            sock.send(json.dumps({"op": OP, "text": text[start : start + SEND_MAX]}).encode(), zmq.NOBLOCK)
    except zmq.Again:
        return sock
    except zmq.ZMQError:
        sock.close(linger=0)
        return None
    return sock


def main() -> None:
    # Like cat: a reader that stops (`head`) ends us, and the writer before us, with SIGPIPE.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        sock = connect()
    except Exception:
        sock = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out = sys.stdout.buffer
    while data := os.read(0, READ):
        out.write(data)
        out.flush()
        try:
            sock = copy(sock, decoder.decode(data))
        except Exception:
            sock = None
    if sock is not None:
        try:
            sock = copy(sock, decoder.decode(b"", final=True))
            if sock is not None:
                sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
