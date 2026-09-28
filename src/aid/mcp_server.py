"""`python -m aid.mcp_server`: the `@aid.mcptool` functions on the agents path, as a stdio MCP server.

Sessions start it as their built-in `aid` MCP server (`aid.mcp.builtin_server`). It imports agent modules, so
it runs as a child of the agent, never in the daemon.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from aid.agents import agents_path, discover

SERVER_NAME = "aid"


def build() -> FastMCP:
    catalog = discover(agents_path())
    for problem in catalog.problems:
        # stdout carries the protocol.
        print(f"aid.mcp_server: {problem}", file=sys.stderr)
    server = FastMCP(SERVER_NAME)
    for tool in catalog.tools.values():
        server.add_tool(tool.fn, name=tool.name, description=tool.description)
    return server


def main() -> None:
    build().run("stdio")


if __name__ == "__main__":
    main()
