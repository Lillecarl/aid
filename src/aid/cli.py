from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from aid import daemon
from aid.client import connect
from aid.launcher import ForkserverLauncher
from aid.paths import default_paths
from aid.protocol import AidError, Output, TextDelta
from aid.spec import AcpSpec, ClaudeTtySpec, PermissionMode, PydanticAISpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aid.spec import AgentSpec


def _env(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aid", description="AI daemon: persistent ACP and pydantic-ai agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("daemon", help="run the daemon in the foreground")
    sub.add_parser("list", help="list sessions")

    def new(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name")
        p.add_argument("--cwd", default=".")
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
        return p

    acp = new("new-acp", "create a session running an ACP agent command, given after --")
    acp.add_argument("--allow", action="store_true", help="grant every permission request (default: deny)")
    acp.add_argument("--no-inherit-env", action="store_true", help="start the agent with only --env")

    claude = new("new-claude", "create a session running interactive Claude Code in pymux; claude args after --")
    claude.add_argument("--trust", action="store_true", help="answer Claude Code's trust-this-folder dialog with yes")
    claude.add_argument("--pymux-socket", help="the pymux server to run in (default: aid's own)")
    claude.add_argument("--claude", default="claude", help="the claude executable")

    py = new("new-py", "create a session running a pydantic-ai agent")
    py.add_argument("target", help="module:attribute of a pydantic_ai agent")
    py.add_argument("--python-path", action="append", default=[], help="prepend to the worker's sys.path")

    prompt = sub.add_parser("prompt", help="send a prompt and stream the answer")
    prompt.add_argument("name")
    prompt.add_argument("text", help="prompt text, or - to read stdin")

    for command in ("cancel", "stop", "delete"):
        sub.add_parser(command, help=f"{command} a session").add_argument("name")
    return parser


def _split_agent_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split `new-acp ... -- cmd args` by hand: argparse.REMAINDER after a positional swallows options too."""
    argv = list(argv)
    if argv[:1] not in (["new-acp"], ["new-claude"]) or "--" not in argv:
        return argv, []
    split = argv.index("--")
    return argv[:split], argv[split + 1 :]


def _spec(args: argparse.Namespace) -> AgentSpec:
    cwd = str(Path(args.cwd).resolve())
    env = _env(args.env)
    if args.command == "new-acp":
        command: list[str] = args.agent_command
        if not command:
            raise SystemExit("new-acp needs an agent command after --")
        return AcpSpec(
            cwd=cwd,
            env=env,
            command=command,
            inherit_env=not args.no_inherit_env,
            permission=PermissionMode.ALLOW if args.allow else PermissionMode.DENY,
        )
    if args.command == "new-claude":
        return ClaudeTtySpec(
            cwd=cwd,
            env=env,
            command=[args.claude],
            args=args.agent_command,
            trust_cwd=args.trust,
            pymux_socket=args.pymux_socket,
        )
    python_path = [str(Path(p).resolve()) for p in args.python_path]
    return PydanticAISpec(cwd=cwd, env=env, target=args.target, python_path=python_path)


async def _client_command(args: argparse.Namespace) -> None:
    async with connect() as client:
        match args.command:
            case "new-acp" | "new-py" | "new-claude":
                await client.create(args.name, _spec(args))
                print(args.name)
            case "list":
                for info in await client.sessions():
                    print(f"{info.name}\t{info.kind}\t{'running' if info.running else 'stopped'}")
            case "prompt":
                text = sys.stdin.read() if args.text == "-" else args.text
                async for event in client.session(args.name).stream(text):
                    if isinstance(event, TextDelta):
                        print(event.text, end="", flush=True)
                    elif isinstance(event, Output):
                        if not isinstance(event.output, str) and event.output is not None:
                            print(json.dumps(event.output, indent=2))
                        print(f"\n[{event.stop_reason}]", file=sys.stderr)
            case "cancel":
                await client.session(args.name).cancel()
            case "stop":
                await client.session(args.name).stop()
            case "delete":
                await client.session(args.name).delete()
            case other:
                raise SystemExit(f"unknown command {other!r}")


async def _serve() -> None:
    async with anyio.create_task_group() as tg:
        tg.start_soon(daemon.run, default_paths(), ForkserverLauncher())
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
            async for signum in signals:
                logging.getLogger("aid").info("%s: stopping workers", signal.Signals(signum).name)
                tg.cancel_scope.cancel()
                return


def main(argv: Sequence[str] | None = None) -> None:
    head, agent_command = _split_agent_command(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(head)
    args.agent_command = agent_command
    if args.command == "daemon":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s")
        anyio.run(_serve)
        return
    try:
        anyio.run(_client_command, args)
    except AidError as error:
        raise SystemExit(f"aid: {error}") from None
