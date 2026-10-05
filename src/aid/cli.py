from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import structlog

from aid import daemon, guard, speech
from aid.client import connect, register_plugin
from aid.launcher import PRELOAD, CommandLauncher, ForkserverLauncher, WorkerArgs, process_main
from aid.log import configure_logging
from aid.mcp import from_claude_config
from aid.paths import default_paths
from aid.plugins import Grant
from aid.protocol import (
    AidError,
    Lifecycle,
    MessageEntry,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    TurnError,
    Usage,
)
from aid.spec import AcpSpec, ClaudeTtySpec, PermissionMode, PydanticAISpec
from aid.web import OidcConfig, create_app
from aid.web import serve as serve_web
from aid.web.app import ENV_ASSETS
from aid.web.highlight import ENV_GRAMMARS, Grammars
from aid.web.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aid.client import Client
    from aid.launcher import Launcher
    from aid.protocol import HistoryEntry
    from aid.spec import AgentSpec, McpServer

ENV_CLIENT_SECRET = "AID_OIDC_CLIENT_SECRET"
ENV_SESSION_SECRET = "AID_WEB_SESSION_SECRET"

log = structlog.get_logger(__name__)


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
        help="with --worker-command: the workers socket as the worker reaches it, such as wss://aid.example/aid",
    )
    daemon_cmd.add_argument(
        "--worker-ca",
        metavar="FILE",
        help="with a wss:// --worker-endpoint: PEM CAs its certificate must chain to (default: the system's)",
    )
    daemon_cmd.add_argument(
        "--no-preload",
        action="store_true",
        help="fork workers from a bare interpreter: every worker imports everything fresh",
    )
    daemon_cmd.add_argument(
        "--preload-module",
        action="append",
        default=None,
        metavar="MODULE",
        help="also import MODULE in the forkserver, so workers share its pages copy-on-write (repeatable)",
    )
    daemon_cmd.add_argument(
        "--allow-worker-command",
        action="store_true",
        help="let sessions name their own worker command (runs as the daemon, with the session's worker "
        "credentials: only where session creators are trusted)",
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
    web.add_argument(
        "--grammars",
        default=os.environ.get(ENV_GRAMMARS),
        help=f"tree-sitter grammars for the file viewer (default: ${ENV_GRAMMARS})",
    )
    web.add_argument(
        "--theme",
        type=_theme,
        help="colours, as pymux names a theme: pygments:<name> or base16:<name> (default: the browser's)",
    )

    def new(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name")
        p.add_argument("--cwd", default=".")
        p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument(
            "--worker-command",
            metavar="COMMAND",
            help="start this session's worker by running COMMAND (split like a shell), which must run `aid worker`; "
            "by default the daemon forks it",
        )
        p.add_argument(
            "--worker-endpoint",
            metavar="ENDPOINT",
            help="with --worker-command: the workers socket as the worker reaches it (default: the daemon's own)",
        )
        p.add_argument(
            "--worker-ca",
            metavar="FILE",
            help="with a wss:// --worker-endpoint: PEM CAs its certificate must chain to (default: the system's)",
        )
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

    def with_permission(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--permission",
            type=PermissionMode,
            choices=list(PermissionMode),
            default=PermissionMode.DENY,
            help="answer what the agent asks permission for: allow, deny, or ask a person (`aid answer`, the web UI)",
        )
        p.add_argument(
            "--allow", dest="permission", action="store_const", const=PermissionMode.ALLOW, help="--permission allow"
        )
        return p

    acp = with_permission(with_mcp(new("new-acp", "create a session running an ACP agent command, given after --")))
    acp.add_argument("--no-inherit-env", action="store_true", help="start the agent with only --env")

    claude = with_mcp(
        new("new-claude", "create a session running interactive Claude Code in pymux; claude args after --")
    )
    claude.add_argument("--trust", action="store_true", help="answer Claude Code's trust-this-folder dialog with yes")
    claude.add_argument("--pymux-socket", help="the pymux server to run in (default: aid's own)")
    claude.add_argument("--claude", default="claude", help="the claude executable")

    py = with_permission(new("new-py", "create a session running a pydantic-ai agent"))
    py.add_argument("agent", help="an agent `aid agents` lists, or module:attribute of a pydantic_ai agent")
    py.add_argument("--python-path", action="append", default=[], help="prepend to the worker's sys.path")
    py.add_argument(
        "--max-context",
        type=int,
        help="tokens the context gauge calls full, as a rule of thumb half the model's window",
    )

    sub.add_parser(
        "agents", help="list the aid.PydanticAgent classes and @aid.mcptool functions on the daemon's agents path"
    )

    sub.add_parser(
        "summary", help="a session's totals as JSON: tokens, models, agent sessions, cost, files touched"
    ).add_argument("name")

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

    answer = sub.add_parser("answer", help="answer a permission request the session waits on")
    answer.add_argument("name")
    answer.add_argument("request", help="the request id, as `aid prompt` and `aid history` print it")
    answer.add_argument("option", nargs="?", help="the option id to choose; none cancels the request")

    for command in ("start", "cancel", "stop", "delete"):
        sub.add_parser(command, help=f"{command} a session").add_argument("name")

    compact = sub.add_parser("compact", help="summarize a session's history into a digest, outside a turn")
    compact.add_argument("name")
    compact.add_argument("focus", nargs="?", help="what the digest keeps for the upcoming work")

    plugin = sub.add_parser("plugin", help="register the processes that reach the daemon as plugins, with grants")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    add = plugin_sub.add_parser("add", help="register a plugin, or replace its key and grants; prints its public key")
    add.add_argument("name")
    add.add_argument(
        "--grant",
        action="append",
        default=[],
        choices=[g.value for g in Grant],
        help="what it may do; repeat for more",
    )
    add.add_argument(
        "--public-key",
        help="a key made elsewhere, for a plugin on another host; by default aid makes a keypair and keeps the "
        "secret where `aid.connect(plugin=NAME)` reads it",
    )
    plugin_sub.add_parser("list", help="list plugins and their grants")
    guard_cmd = sub.add_parser(
        "guard", help="run aid.guard: answer permission requests that only read, leave the rest for a person"
    )
    guard_cmd.add_argument("--plugin", default="guard", help="the plugin it registered as, with read and permissions")
    plugin_sub.add_parser("remove", help="remove a plugin; it is refused from its next request").add_argument("name")
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
    if args.worker_command:
        worker_command: list[str] | None = [part for part in shlex.split(args.worker_command) if part]
        if not worker_command:
            raise SystemExit("new: --worker-command names no program")
    else:
        worker_command = None
    worker_endpoint: str | None = args.worker_endpoint
    worker_ca: str | None = args.worker_ca
    if worker_command is None and (worker_endpoint is not None or worker_ca is not None):
        raise SystemExit("new: --worker-endpoint and --worker-ca need --worker-command")
    if args.command == "new-acp":
        command: list[str] = args.agent_command
        if not command:
            raise SystemExit("new-acp needs an agent command after --")
        return AcpSpec(
            cwd=cwd,
            env=env,
            command=command,
            inherit_env=not args.no_inherit_env,
            permission=args.permission,
            mcp_servers=_mcp_servers(args.mcp_config),
            worker_command=worker_command,
            worker_endpoint=worker_endpoint,
            worker_ca=worker_ca,
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
            worker_command=worker_command,
            worker_endpoint=worker_endpoint,
            worker_ca=worker_ca,
        )
    python_path = [str(Path(p).resolve()) for p in args.python_path]
    source = {"target": args.agent} if ":" in args.agent else {"agent": args.agent}
    return PydanticAISpec(
        cwd=cwd,
        env=env,
        python_path=python_path,
        permission=args.permission,
        worker_command=worker_command,
        worker_endpoint=worker_endpoint,
        worker_ca=worker_ca,
        max_context=args.max_context,
        **source,
    )


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
        case Started():
            how = "[resumed]" if item.resumed else "[started]"
            body = " ".join(p for p in (how, f"pid {item.pid}", item.agent, item.model, item.agent_session) if p)
        case Usage():
            body = " ".join([f"[usage] {item.input_tokens} in, {item.output_tokens} out", *item.models])
        case PermissionRequest() | PermissionDecision():
            body = _permission_line(item)
        case Lifecycle():
            body = " ".join(p for p in (f"[{item.event}]", item.detail) if p)
    return f"{entry.seq:>6}  {body}"


def _permission_line(item: PermissionRequest | PermissionDecision) -> str:
    match item:
        case PermissionRequest():
            options = ", ".join(f"{o.option_id} ({o.name})" for o in item.options)
            return f"[permission {item.request_id}] {item.tool_name or ''} {item.title or ''}: {options}"
        case PermissionDecision(by=PermissionDecider.TERMINAL):
            return f"[permission {item.request_id}] answered in the terminal"
        case PermissionDecision():
            by = f"plugin {item.plugin}" if item.plugin else item.by
            return f"[permission {item.request_id}] {item.option_id or 'cancelled'} by {by}"


async def _client_command(args: argparse.Namespace) -> None:
    async with connect() as client:
        match args.command:
            case "new-acp" | "new-py" | "new-claude":
                await client.create(args.name, _spec(args))
                print(args.name)
            case "list":
                for info in await client.sessions():
                    print(f"{info.name}\t{info.kind}\t{'running' if info.running else 'stopped'}")
            case "summary":
                print((await client.session(args.name).summary()).model_dump_json(indent=2))
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
                    elif isinstance(event, PermissionRequest | PermissionDecision):
                        print(f"\n{_permission_line(event)}", file=sys.stderr, flush=True)
                    elif isinstance(event, Output):
                        if not isinstance(event.output, str) and event.output is not None:
                            print(json.dumps(event.output, indent=2))
                        print(f"\n[{event.stop_reason}]", file=sys.stderr)
            case "message":
                text = sys.stdin.read() if args.text == "-" else args.text
                await client.send_message(args.name, text, sender=args.sender)
            case "start":
                await client.session(args.name).start()
            case "compact":
                print(await client.session(args.name).compact(args.focus))
            case "cancel":
                await client.session(args.name).cancel()
            case "answer":
                await client.session(args.name).answer(args.request, args.option)
            case "stop":
                await client.session(args.name).stop()
            case "delete":
                await client.session(args.name).delete()
            case "plugin":
                await _plugin_command(client, args)
            case other:
                raise SystemExit(f"unknown command {other!r}")


async def _plugin_command(client: Client, args: argparse.Namespace) -> None:
    match args.plugin_command:
        case "add":
            grants = frozenset(Grant(g) for g in args.grant)
            print(await register_plugin(client, default_paths(), args.name, grants, public_key=args.public_key))
        case "list":
            for spec in await client.plugins():
                print(f"{spec.name}\t{','.join(sorted(spec.grants)) or '-'}\t{spec.public_key}")
        case "remove":
            await client.remove_plugin(args.name)
        case other:
            raise SystemExit(f"unknown plugin command {other!r}")


def _preload(args: argparse.Namespace) -> list[str]:
    """The forkserver's preload: the default plus `--preload-module`, or only `--preload-module` after `--no-preload`."""
    modules: list[str] = args.preload_module or []
    return list(modules) if args.no_preload else [*PRELOAD, *modules]


def _launcher(args: argparse.Namespace) -> Launcher:
    if args.worker_command is None:
        if args.worker_endpoint is not None:
            raise SystemExit("aid daemon: --worker-endpoint needs --worker-command")
        return ForkserverLauncher(_preload(args))
    if args.worker_endpoint is None:
        raise SystemExit("aid daemon: --worker-command needs --worker-endpoint")
    trust_pem = Path(args.worker_ca).read_text() if args.worker_ca else ""
    return CommandLauncher(shlex.split(args.worker_command), args.worker_endpoint, trust_pem=trust_pem)


async def _serve(args: argparse.Namespace) -> None:
    launcher = _launcher(args)
    async with anyio.create_task_group() as tg:
        tg.start_soon(
            partial(
                daemon.run,
                default_paths(),
                launcher,
                workers_listen=args.workers_listen,
                allow_worker_command=args.allow_worker_command,
            )
        )
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
            async for signum in signals:
                log.info("stopping_workers", signal=signal.Signals(signum).name)
                tg.cancel_scope.cancel()
                return


def _theme(spec: str) -> Theme:
    try:
        return Theme.parse(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


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
    grammars = Grammars(Path(args.grammars)) if args.grammars else None
    app = create_app(
        oidc,
        _secret(ENV_SESSION_SECRET),
        assets=assets,
        recognizer=recognizer,
        colors=args.theme,
        grammars=grammars,
    )
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
    if args.command == "guard":
        configure_logging()
        try:
            anyio.run(guard.main, args.plugin)
        except AidError as error:
            raise SystemExit(f"aid guard: {error}") from None
        return
    if args.command in ("daemon", "web", "worker"):
        configure_logging()
        if args.command == "worker":
            process_main(WorkerArgs.from_json(sys.stdin.read()))
        else:
            anyio.run(lambda: _serve(args) if args.command == "daemon" else _web(args))
        return
    try:
        anyio.run(_client_command, args)
    except AidError as error:
        raise SystemExit(f"aid: {error}") from None
