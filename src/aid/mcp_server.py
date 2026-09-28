"""`python -m aid.mcp_server`: aid's tools and the `@aid.mcptool` functions on the agents path, as a stdio MCP
server.

Sessions start it as their built-in `aid` MCP server (`aid.mcp.builtin_server`). It imports agent modules, so
it runs as a child of the agent, never in the daemon.

For interactive Claude it is also the session's channel (AID_CHANNEL=1): it declares `claude/channel`, takes the
session's messages from the daemon, and pushes each as a `notifications/claude/channel` event, which wakes Claude
even when it is idle. Channels are a Claude Code research preview; Claude takes the events only when started
with `--dangerously-load-development-channels server:aid`. Measured on Claude Code 2.1.284: the event reaches the
model as `<channel source="aid" from="...">`, and Claude frames its content as untrusted, so what to do with a
message is in the server's `instructions`.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Final

import anyio
import zmq
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from aid.agents import agents_path, discover
from aid.client import connect
from aid.paths import ENV_CHANNEL, ENV_SESSION
from aid.protocol import AidError

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectSendStream

    from aid.protocol import MessageEntry

SERVER_NAME: Final = "aid"
CHANNEL: Final = "notifications/claude/channel"
POLL: Final = 25.0
# Past this the daemon has gone, not merely found no message: reconnect.
POLL_GRACE: Final = 10.0
RETRY: Final = 2.0


def instructions(session: str) -> str:
    return (
        f"You are aid session {session!r}. Other aid sessions reach you with "
        f'<channel source="{SERVER_NAME}" from="NAME"> events; one without `from` is from a person. When a message '
        "asks for something, answer it with the send_message tool, addressed to NAME. Do not answer a message that "
        "only acknowledges or thanks. list_sessions shows who you can message."
    )


def channel_params(message: MessageEntry) -> dict[str, Any]:
    # Meta keys are identifiers; Claude Code drops any other.
    return {"content": message.text, "meta": {"from": message.sender} if message.sender else {}}


def build(channel_session: str | None) -> FastMCP:
    catalog = discover(agents_path())
    for problem in catalog.problems:
        # stdout carries the protocol.
        print(f"aid.mcp_server: {problem}", file=sys.stderr)
    server = FastMCP(
        SERVER_NAME,
        instructions=instructions(channel_session) if channel_session else None,
        log_level="WARNING",
    )
    for tool in catalog.tools.values():
        server.add_tool(tool.fn, name=tool.name, description=tool.description)
    return server


async def forward_messages(session: str, push: MemoryObjectSendStream[SessionMessage]) -> None:
    """The session's messages, from the daemon's mailbox to the channel, for as long as the server runs."""
    while True:
        try:
            async with connect() as client:
                while True:
                    with anyio.fail_after(POLL + POLL_GRACE):
                        messages = await client.receive_messages(session, wait=POLL)
                    for message in messages:
                        notification = JSONRPCNotification(
                            jsonrpc="2.0", method=CHANNEL, params=channel_params(message)
                        )
                        await push.send(SessionMessage(message=JSONRPCMessage(notification)))
        except (AidError, TimeoutError, zmq.ZMQError) as error:
            print(f"aid.mcp_server: messages for {session}: {error}; retrying", file=sys.stderr)
            await anyio.sleep(RETRY)


async def serve() -> None:
    channel_session = os.environ.get(ENV_SESSION) if os.environ.get(ENV_CHANNEL) == "1" else None
    server = build(channel_session)
    lowlevel = server._mcp_server  # pyright: ignore[reportPrivateUsage] -- FastMCP cannot declare an experimental capability
    options = lowlevel.create_initialization_options(
        experimental_capabilities={"claude/channel": {}} if channel_session else None
    )
    async with stdio_server() as (read_stream, write_stream), anyio.create_task_group() as tg:
        push = write_stream.clone()
        if channel_session:
            tg.start_soon(forward_messages, channel_session, push)
        try:
            await lowlevel.run(read_stream, write_stream, options)
        finally:
            tg.cancel_scope.cancel()
            # The stdio writer runs until every clone of its stream is closed: an open one keeps the server
            # alive after Claude has gone.
            await push.aclose()


def main() -> None:
    anyio.run(serve)


if __name__ == "__main__":
    main()
