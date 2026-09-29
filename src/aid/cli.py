from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread

from aid import daemon, speech
from aid.client import connect
from aid.launcher import CommandLauncher, ForkserverLauncher, WorkerArgs, process_main
from aid.mcp import from_claude_config
from aid.paths import default_paths
from aid.protocol import (
    AidError,
    MessageEntry,
    Output,
    PromptEntry,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    TurnError,
)
from aid.spec import AcpSpec, ClaudeTtySpec, PermissionMode, PydanticAISpec
from aid.web import OidcConfig, create_app
from aid.web import serve as serve_web
from aid.web.app import ENV_ASSETS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aid.launcher import Launcher
    from aid.protocol import HistoryEntry
    from aid.spec import AgentSpec, McpServer

ENV_CLIENT_SECRET = "AID_OIDC_CLIENT_SECRET"
ENV_SESSION_SECRET = "AID_WEB_SESSION_SECRET"


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

    daemon_cmd = sub.add_parser("daemon", help="run the daemon in the foreground")
    daemon_cmd.add_argument(
        "--workers-listen",
        action="append",
        default=[],
        metavar="ENDPOINT",
        help="also take worker connections here, such as ws://0.0.0.0:7070/aid behind a reverse proxy (repeatable)",
    )
    daemon_cmd.add_argument(
        "--worker-command",
        metavar="COMMAND",
        help="start workers by running COMMAND (split like a shell), which must run `aid worker`; "
        "by default the daemon forks them",
    )
    daemon_cmd.add_argument(
        "--worker-endpoint",
        metavar="ENDPOINT",
        help="with --worker-command: the workers socket as the worker reaches it, such as ws://aid.example:7070/aid",
    )
    sub.add_parser("worker", help="run one worker; `aid daemon --worker-command` starts it and sends its arguments")
    sub.add_parser("list", help="list sessions")

    web = sub.add_parser(
        "web",
        help="serve the web UI, with OIDC login",
        description=f"Secrets come from the environment: {ENV_CLIENT_SECRET} and {ENV_SESSION_SECRET}.",
    )
    web.add_argument("--bind", default="127.0.0.1:8080", help="host:port to listen on")
    web.add_argument("--base-url", help="where browsers reach aid (default: http://BIND)")
    web.add_argument("--issuer", required=True, help="the OIDC issuer URL")
    web.add_argument("--client-id", required=True)
    web.add_argument("--assets", default=os.environ.get(ENV_ASSETS), help=f"the built UI (default: ${ENV_ASSETS})")
    web.add_argument(
        "--allow-email", action="append", required=True, help="a verified email that may log in; repeat for more"
    )
    web.add_argument(
        "--speech-model",
        help="a sherpa-onnx streaming transducer directory, for speech to text (default: none, no microphone)",
    )

    def new(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name")
        p.add_argument("--cwd", default=".")
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
        return p

    def with_mcp(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--mcp-config",
            action="append",
            default=[],
            metavar="FILE",
            help='MCP servers for the agent, in Claude Code\'s {"mcpServers": {...}} format; repeat for more',
        )
        return p

    acp = with_mcp(new("new-acp", "create a session running an ACP agent command, given after --"))
    acp.add_argument("--allow", action="store_true", help="grant every permission request (default: deny)")
    acp.add_argument("--no-inherit-env", action="store_true", help="start the agent with only --env")

    claude = with_mcp(
        new("new-claude", "create a session running interactive Claude Code in pymux; claude args after --")
    )
    claude.add_argument("--trust", action="store_true", help="answer Claude Code's trust-this-folder dialog with yes")
    claude.add_argument("--pymux-socket", help="the pymux server to run in (default: aid's own)")
    claude.add_argument("--claude", default="claude", help="the claude executable")

    py = new("new-py", "create a session running a pydantic-ai agent")
    py.add_argument("agent", help="an agent `aid agents` lists, or module:attribute of a pydantic_ai agent")
    py.add_argument("--python-path", action="append", default=[], help="prepend to the worker's sys.path")

    sub.add_parser(
        "agents", help="list the aid.PydanticAgent classes and @aid.mcptool functions on the daemon's agents path"
    )

    hist = sub.add_parser("history", help="print a session's history, newest last")
    hist.add_argument("name")
    hist.add_argument("--limit", type=int, default=50)
    hist.add_argument("--before", type=int, help="only entries older than this seq")
    hist.add_argument("--json", action="store_true", help="one JSON entry per line")

    prompt = sub.add_parser("prompt", help="send a prompt and stream the answer")
    prompt.add_argument("name")
    prompt.add_argument("text", help="prompt text, or - to read stdin")

    message = sub.add_parser("message", help="leave a message for a session, which wakes it")
    message.add_argument("name")
    message.add_argument("text", help="message text, or - to read stdin")
    message.add_argument("--from", dest="sender", help="the session it is from (default: a person)")

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


def _mcp_servers(files: Sequence[str]) -> list[McpServer]:
    servers: list[McpServer] = []
    for file in files:
        try:
            servers += from_claude_config(json.loads(Path(file).read_text()))
        except (OSError, ValueError) as error:
            raise SystemExit(f"--mcp-config {file}: {error}") from None
    return servers


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
            mcp_servers=_mcp_servers(args.mcp_config),
        )
    if args.command == "new-claude":
        return ClaudeTtySpec(
            cwd=cwd,
            env=env,
            command=[args.claude],
            args=args.agent_command,
            trust_cwd=args.trust,
            pymux_socket=args.pymux_socket,
            mcp_servers=_mcp_servers(args.mcp_config),
        )
    python_path = [str(Path(p).resolve()) for p in args.python_path]
    source = {"target": args.agent} if ":" in args.agent else {"agent": args.agent}
    return PydanticAISpec(cwd=cwd, env=env, python_path=python_path, **source)


def _history_line(entry: HistoryEntry) -> str:
    item = entry.item
    match item:
        case PromptEntry():
            body = f"> {item.text}"
        case TextDelta():
            body = item.text
        case ThoughtDelta():
            body = f"(thinking) {item.text}"
        case ToolCall():
            body = f"[tool] {item.title or item.tool_call_id} {item.status or ''}".rstrip()
        case Output():
            body = (
                f"[{item.stop_reason}]"
                if isinstance(item.output, str)
                else f"{json.dumps(item.output)} [{item.stop_reason}]"
            )
        case MessageEntry():
            body = f"[message from {item.sender or 'a person'}] {item.text}"
        case TurnError():
            body = f"[error {item.code}] {item.message}"
    return f"{entry.seq:>6}  {body}"


async def _client_command(args: argparse.Namespace) -> None:
    async with connect() as client:
        match args.command:
            case "new-acp" | "new-py" | "new-claude":
                await client.create(args.name, _spec(args))
                print(args.name)
            case "list":
                for info in await client.sessions():
                    print(f"{info.name}\t{info.kind}\t{'running' if info.running else 'stopped'}")
            case "history":
                page = await client.session(args.name).history(before=args.before, limit=args.limit)
                if page.has_older and not args.json:
                    print(f"… {page.entries[0].seq if page.entries else page.total} older entries", file=sys.stderr)
                for entry in page.entries:
                    print(entry.model_dump_json() if args.json else _history_line(entry))
            case "agents":
                catalog = await client.agents()
                for agent in catalog.agents:
                    print(f"{agent.name}\t{agent.module}\t{agent.description}")
                for tool in catalog.tools:
                    print(f"tool {tool.name}\t{tool.module}\t{tool.description}")
                for problem in catalog.problems:
                    print(f"problem: {problem}", file=sys.stderr)
            case "prompt":
                text = sys.stdin.read() if args.text == "-" else args.text
                async for event in client.session(args.name).stream(text):
                    if isinstance(event, TextDelta):
                        print(event.text, end="", flush=True)
                    elif isinstance(event, Output):
                        if not isinstance(event.output, str) and event.output is not None:
                            print(json.dumps(event.output, indent=2))
                        print(f"\n[{event.stop_reason}]", file=sys.stderr)
            case "message":
                text = sys.stdin.read() if args.text == "-" else args.text
                await client.send_message(args.name, text, sender=args.sender)
            case "cancel":
                await client.session(args.name).cancel()
            case "stop":
                await client.session(args.name).stop()
            case "delete":
                await client.session(args.name).delete()
            case other:
                raise SystemExit(f"unknown command {other!r}")


def _launcher(args: argparse.Namespace) -> Launcher:
    if args.worker_command is None:
        if args.worker_endpoint is not None:
            raise SystemExit("aid daemon: --worker-endpoint needs --worker-command")
        return ForkserverLauncher()
    if args.worker_endpoint is None:
        raise SystemExit("aid daemon: --worker-command needs --worker-endpoint")
    return CommandLauncher(shlex.split(args.worker_command), args.worker_endpoint)


async def _serve(args: argparse.Namespace) -> None:
    launcher = _launcher(args)
    async with anyio.create_task_group() as tg:
        tg.start_soon(partial(daemon.run, default_paths(), launcher, workers_listen=args.workers_listen))
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
            async for signum in signals:
                logging.getLogger("aid").info("%s: stopping workers", signal.Signals(signum).name)
                tg.cancel_scope.cancel()
                return


def _secret(name: str) -> str:
    if not (value := os.environ.get(name)):
        raise SystemExit(f"aid web: set {name}")
    return value


async def _web(args: argparse.Namespace) -> None:
    oidc = OidcConfig(
        issuer=args.issuer,
        client_id=args.client_id,
        client_secret=_secret(ENV_CLIENT_SECRET),
        base_url=args.base_url or f"http://{args.bind}",
        allowed_emails=frozenset(email.lower() for email in args.allow_email),
    )
    assets = Path(args.assets) if args.assets else None
    # About a second, once, before aid web listens.
    recognizer = await anyio.to_thread.run_sync(speech.load, Path(args.speech_model)) if args.speech_model else None
    app = create_app(oidc, _secret(ENV_SESSION_SECRET), assets=assets, recognizer=recognizer)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: serve_web(app, args.bind, shutdown=shutdown))
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
            async for _signum in signals:
                shutdown.set()
                return


def main(argv: Sequence[str] | None = None) -> None:
    head, agent_command = _split_agent_command(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(head)
    args.agent_command = agent_command
    if args.command in ("daemon", "web", "worker"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s")
        if args.command == "worker":
            process_main(WorkerArgs.from_json(sys.stdin.read()))
        else:
            anyio.run(lambda: _serve(args) if args.command == "daemon" else _web(args))
        return
    try:
        anyio.run(_client_command, args)
    except AidError as error:
        raise SystemExit(f"aid: {error}") from None
