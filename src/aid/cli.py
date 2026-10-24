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

from aid import daemon, guard, plugin_host, speech
from aid.client import connect, register_plugin
from aid.discovery import discover
from aid.launcher import PRELOAD, CommandLauncher, ForkserverLauncher, WorkerArgs, process_main
from aid.log import configure_logging
from aid.mcp import from_claude_config
from aid.paths import default_paths
from aid.plugins import ENV_PLUGINS_PATH, Grant, plugins_path
from aid.protocol import (
    AidError,
    Delivery,
    Lifecycle,
    MessageEntry,
    Output,
    PermissionDecider,
    PermissionDecision,
    PermissionRequest,
    PromptEntry,
    RemoteControl,
    SessionInfo,
    Started,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    TurnError,
    Usage,
)
from aid.spec import AcpSpec, ClaudeSdkSpec, ClaudeTtySpec, NativeSpec, PermissionMode, Sandbox
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
    parser = argparse.ArgumentParser(prog="aid", description="AI daemon: persistent ACP and native agents.")
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
    sessions = sub.add_parser("list", help="list sessions")
    sessions.add_argument("--json", action="store_true", help="one session JSON object per line")

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
        "--cwd-prefix",
        help="what a new session's working directory starts as in the page (default: the home directory)",
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
        p.add_argument(
            "--supervisor",
            metavar="SESSION",
            help="the session watching this one: it reviews every turn end, and answers requests of tools in "
            "supervisor or ask_supervisor mode (`aid permissions`)",
        )
        p.add_argument(
            "--supervisor-permission-timeout",
            type=float,
            default=300,
            metavar="SECONDS",
            help="seconds a person holds a request in a supervisor mode before it moves on (default: 300)",
        )
        p.add_argument(
            "--supervisor-permission-absent-timeout",
            type=float,
            default=60,
            metavar="SECONDS",
            help="that wait, shortened when no person-authored entry is recent (default: 60)",
        )
        p.add_argument(
            "--supervisor-nudge-budget",
            type=int,
            default=3,
            metavar="COUNT",
            help="nudges the supervisor may send per stuck episode before the daemon refuses and escalates (default: 3)",
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

    py = with_permission(new("new-py", "create a session running a native agent"))
    py.add_argument("agent", help="an agent `aid agents` lists, or module:attribute of an aid.Agent")
    py.add_argument("--model", help="provider/model to run on instead of the agent's own (`aid models` lists them)")
    py.add_argument("--python-path", action="append", default=[], help="prepend to the worker's sys.path")
    py.add_argument(
        "--max-context",
        type=int,
        help="tokens the context gauge calls full, as a rule of thumb half the model's window",
    )
    py.add_argument(
        "--max-requests",
        type=int,
        help="model requests one turn may spend before it ends capped",
    )

    sdk = with_permission(new("new-sdk", "create a session driving Claude Code over NDJSON, with Remote Control"))
    sdk.add_argument("--model", required=True, help="the model the session runs on")
    sdk.add_argument("--claude", default="claude", help="the claude executable")

    sub.add_parser(
        "agents", help="list the aid.AgentDefinition classes and @aid.mcptool functions on the daemon's plugins path"
    )
    sub.add_parser("models", help="model providers as JSON; with a provider, the models it serves").add_argument(
        "provider", nargs="?"
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
    prompt.add_argument("-v", "--verbose", action="store_true", help="tool calls on stderr as they start and end")

    message = sub.add_parser("message", help="leave a message for a session, which wakes it")
    message.add_argument("name")
    message.add_argument("text", help="message text, or - to read stdin")
    message.add_argument("--from", dest="sender", help="the session it is from (default: a person)")
    message.add_argument(
        "--wait-for-turn",
        dest="delivery",
        action="store_const",
        const=Delivery.WAIT_FOR_TURN,
        default=Delivery.INTERRUPT,
        help="hold it until the session's running turn ends (default: between the turn's steps, where it can)",
    )

    answer = sub.add_parser("answer", help="answer a permission request the session waits on")
    answer.add_argument("name")
    answer.add_argument("request", help="the request id, as `aid prompt` and `aid history` print it")
    answer.add_argument("option", nargs="?", help="the option id to choose; none cancels the request")
    answer.add_argument("--text", help="your own words: with an option, alongside the pick; alone, the whole answer")

    cd = sub.add_parser("cd", help="move a session to another working directory, or answer its agent's request to")
    cd.add_argument("name")
    cd.add_argument("path", nargs="?", help="the new directory, relative to the current one")
    answer_cd = cd.add_mutually_exclusive_group()
    answer_cd.add_argument("--approve", action="store_true", help="approve the move the agent asked for")
    answer_cd.add_argument("--decline", action="store_true", help="decline the move the agent asked for")

    permissions = sub.add_parser("permissions", help="list a native session's tools and modes, or set them")
    permissions.add_argument("name")
    permissions.add_argument(
        "set",
        nargs="*",
        metavar="TOOL=MODE",
        help=f"{', '.join(PermissionMode)}; empty returns a tool to its default",
    )
    permissions.add_argument(
        "--default", choices=list(PermissionMode), help="the mode of tools that ask for themselves"
    )

    box = sub.add_parser("sandbox", help="run a native session's tools under bwrap, or stop that")
    box.add_argument("name")
    box.add_argument("--off", action="store_true", help="run the tools unconfined")
    box.add_argument("--no-network", action="store_true", help="cut the sandbox off the network")
    box.add_argument("--writable", action="append", default=[], metavar="PATH", help="absolute; repeatable")

    supervise = sub.add_parser("supervise", help="set or end the session supervising another")
    supervise.add_argument("name")
    supervise.add_argument("supervisor", nargs="?", help="the supervising session; omitted ends supervision")

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
    run = plugin_sub.add_parser("run", help="run a plugin from the plugins path as its registered name")
    run.add_argument("name")
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
            supervisor=args.supervisor,
            supervisor_permission_timeout=args.supervisor_permission_timeout,
            supervisor_permission_absent_timeout=args.supervisor_permission_absent_timeout,
            supervisor_nudge_budget=args.supervisor_nudge_budget,
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
            supervisor=args.supervisor,
            supervisor_permission_timeout=args.supervisor_permission_timeout,
            supervisor_permission_absent_timeout=args.supervisor_permission_absent_timeout,
            supervisor_nudge_budget=args.supervisor_nudge_budget,
        )
    if args.command == "new-sdk":
        return ClaudeSdkSpec(
            cwd=cwd,
            env=env,
            command=[args.claude],
            model=args.model,
            permission=args.permission,
            worker_command=worker_command,
            worker_endpoint=worker_endpoint,
            worker_ca=worker_ca,
            supervisor=args.supervisor,
            supervisor_permission_timeout=args.supervisor_permission_timeout,
            supervisor_permission_absent_timeout=args.supervisor_permission_absent_timeout,
            supervisor_nudge_budget=args.supervisor_nudge_budget,
        )
    python_path = [str(Path(p).resolve()) for p in args.python_path]
    source = {"target": args.agent} if ":" in args.agent else {"agent": args.agent}
    return NativeSpec(
        cwd=cwd,
        env=env,
        python_path=python_path,
        permission=args.permission,
        worker_command=worker_command,
        worker_endpoint=worker_endpoint,
        worker_ca=worker_ca,
        max_context=args.max_context,
        max_requests=args.max_requests,
        model=args.model,
        supervisor=args.supervisor,
        supervisor_permission_timeout=args.supervisor_permission_timeout,
        supervisor_permission_absent_timeout=args.supervisor_permission_absent_timeout,
        supervisor_nudge_budget=args.supervisor_nudge_budget,
        **source,
    )


BRIEF = 120


def _tool_line(call: ToolCall, titles: dict[str, str]) -> str:
    """Updates carry only the call id, so `titles` keeps each call's name from its first event."""
    if call.title:
        titles[call.tool_call_id] = call.title
    parts = ["[tool]", titles.get(call.tool_call_id, call.tool_call_id)]
    if call.status:
        parts.append(call.status)
    detail = call.output if call.status == "failed" else call.input
    if detail is not None:
        text = " ".join((detail if isinstance(detail, str) else json.dumps(detail)).split())
        parts.append(text if len(text) <= BRIEF else text[: BRIEF - 1] + "…")
    return " ".join(parts)


def _history_line(entry: HistoryEntry, titles: dict[str, str]) -> str:
    item = entry.item
    match item:
        case PromptEntry():
            body = f"> {item.text}"
        case TextDelta():
            body = item.text
        case ThoughtDelta():
            body = f"(thinking) {item.text}"
        case ToolCall():
            body = _tool_line(item, titles)
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
            # Cents, or two significant digits below a cent, as the web UI shows it.
            amount = item.cost.amount if item.cost is not None else 0.0
            shown = f"{amount:.2f}" if amount >= 0.01 or amount == 0 else f"{amount:.2g}"
            cost = [f"{shown} {item.cost.currency}"] if item.cost is not None else []
            body = " ".join([f"[usage] {item.input_tokens} in, {item.output_tokens} out", *item.models, *cost])
        case PermissionRequest() | PermissionDecision():
            body = _permission_line(item)
        case Lifecycle():
            body = " ".join(p for p in (f"[{item.event}]", item.detail) if p)
        case RemoteControl():
            body = f"[remote] {item.session_url}"
    return f"{entry.seq:>6}  {body}"


def _session_line(info: SessionInfo) -> str:
    """Attention goes last: it is free text, and may hold tabs."""
    parts = [info.name, info.kind, "working" if info.working else "running" if info.running else "stopped"]
    if info.permissions:
        parts.append(f"{info.permissions} permission{'s' if info.permissions != 1 else ''}")
    if info.attention:
        parts.append(info.attention)
    return "\t".join(parts)


def _permission_line(item: PermissionRequest | PermissionDecision) -> str:
    match item:
        case PermissionRequest():
            options = ", ".join(f"{o.option_id} ({o.name})" for o in item.options)
            return f"[permission {item.request_id}] {item.tool_name or ''} {item.title or ''}: {options}"
        case PermissionDecision(by=PermissionDecider.TERMINAL):
            return f"[permission {item.request_id}] answered in the terminal"
        case PermissionDecision():
            by = f"plugin {item.plugin}" if item.plugin else item.by
            picked = item.option_id or "cancelled"
            text = f": {item.text}" if item.text else ""
            return f"[permission {item.request_id}] {picked} by {by}{text}"


async def _client_command(args: argparse.Namespace) -> None:
    async with connect() as client:
        match args.command:
            case "new-acp" | "new-py" | "new-claude" | "new-sdk":
                await client.create(args.name, _spec(args))
                print(args.name)
            case "list":
                for info in await client.sessions():
                    print(info.model_dump_json() if args.json else _session_line(info))
            case "summary":
                print((await client.session(args.name).summary()).model_dump_json(indent=2))
            case "history":
                page = await client.session(args.name).history(before=args.before, limit=args.limit)
                if page.has_older and not args.json:
                    print(f"… {page.entries[0].seq if page.entries else page.total} older entries", file=sys.stderr)
                titles: dict[str, str] = {}
                for entry in page.entries:
                    print(entry.model_dump_json() if args.json else _history_line(entry, titles))
            case "agents":
                catalog = await client.agents()
                for agent in catalog.agents:
                    print(f"{agent.name}\t{agent.module}\t{agent.description}")
                for tool in catalog.tools:
                    print(f"tool {tool.name}\t{tool.module}\t{tool.description}")
                for problem in catalog.problems:
                    print(f"problem: {problem}", file=sys.stderr)
            case "models":
                listing = await client.models(args.provider)
                if args.provider and not listing.providers:
                    raise AidError("unknown_provider", f"no model provider {args.provider!r}")
                print(listing.model_dump_json(exclude_none=True))
                for row in listing.providers:
                    if row.problem:
                        print(f"problem: {row.name}: {row.problem}", file=sys.stderr)
            case "prompt":
                text = sys.stdin.read() if args.text == "-" else args.text
                titles: dict[str, str] = {}
                written = after_tool = False
                async for event in client.session(args.name).stream(text):
                    if isinstance(event, TextDelta):
                        # Each model response streams its own text: without a break they run together.
                        if written and after_tool:
                            print("\n", flush=True)
                        print(event.text, end="", flush=True)
                        written, after_tool = True, False
                    elif isinstance(event, ToolCall):
                        if args.verbose:
                            lead = "\n" if written and not after_tool else ""
                            print(lead + _tool_line(event, titles), file=sys.stderr, flush=True)
                        after_tool = True
                    elif isinstance(event, PermissionRequest | PermissionDecision):
                        print(f"\n{_permission_line(event)}", file=sys.stderr, flush=True)
                    elif isinstance(event, Output):
                        if not isinstance(event.output, str) and event.output is not None:
                            print(json.dumps(event.output, indent=2))
                        print(f"\n[{event.stop_reason}]", file=sys.stderr)
            case "message":
                text = sys.stdin.read() if args.text == "-" else args.text
                await client.send_message(args.name, text, sender=args.sender, delivery=args.delivery)
            case "start":
                await client.session(args.name).start()
            case "compact":
                print(await client.session(args.name).compact(args.focus))
            case "cancel":
                await client.session(args.name).cancel()
            case "answer":
                await client.session(args.name).answer(args.request, args.option, args.text)
            case "cd":
                session = client.session(args.name)
                if args.approve or args.decline:
                    await session.answer_directory(args.approve)
                elif args.path:
                    print(await session.change_directory(args.path))
                else:
                    raise SystemExit("aid cd: give a path, --approve or --decline")
            case "permissions":
                session = client.session(args.name)
                if args.set or args.default:
                    await session.set_permissions(_tool_modes(args.set), args.default)
                else:
                    for tool in (await session.status()).tools:
                        print(f"{tool.group}\t{tool.name}\t{tool.mode}\t{'set' if tool.set else 'default'}")
            case "sandbox":
                chosen = None if args.off else Sandbox(network=not args.no_network, writable=args.writable)
                await client.session(args.name).set_sandbox(chosen)
            case "supervise":
                await client.session(args.name).set_supervisor(args.supervisor)
            case "stop":
                await client.session(args.name).stop()
            case "delete":
                await client.session(args.name).delete()
            case "plugin":
                await _plugin_command(client, args)
            case other:
                raise SystemExit(f"unknown command {other!r}")


def _tool_modes(settings: list[str]) -> dict[str, PermissionMode | None]:
    modes: dict[str, PermissionMode | None] = {}
    for setting in settings:
        name, sep, mode = setting.partition("=")
        if not sep or not name:
            raise SystemExit(f"aid permissions: {setting!r} is not TOOL=MODE")
        try:
            modes[name] = PermissionMode(mode) if mode else None
        except ValueError:
            raise SystemExit(f"aid permissions: {mode!r} is not allow, ask or deny") from None
    return modes


async def _plugin_command(client: Client, args: argparse.Namespace) -> None:
    match args.plugin_command:
        case "add":
            grants = frozenset(Grant(g) for g in args.grant) if args.grant else _declared_grants(args.name)
            print(await register_plugin(client, default_paths(), args.name, grants, public_key=args.public_key))
        case "list":
            for spec in await client.plugins():
                print(f"{spec.name}\t{','.join(sorted(spec.grants)) or '-'}\t{spec.public_key}")
        case "remove":
            await client.remove_plugin(args.name)
        case "run":
            raise SystemExit("aid plugin run connects as the plugin itself: it runs outside this command")
        case other:
            raise SystemExit(f"unknown plugin command {other!r}")


def _declared_grants(name: str) -> frozenset[Grant]:
    """The grants the plugins path's `name` declares. Explicit --grant stays for plugins registered from
    elsewhere, whose code lives on another host."""
    found = discover(plugins_path(), only=[name]).plugins.get(name)
    if found is None:
        raise SystemExit(
            f"aid plugin add: no plugin {name!r} on the plugins path: pass --grant, or set {ENV_PLUGINS_PATH}"
        )
    return found.grants


def _grant_flags(grants: frozenset[Grant]) -> str:
    return " ".join(f"--grant {grant.value}" for grant in sorted(grants, key=lambda grant: grant.value))


async def _run_plugin(name: str) -> None:
    """Run the plugin the plugins path names, connected as its registered name."""
    found = discover(plugins_path(), only=[name]).plugins.get(name)
    if found is None:
        raise SystemExit(f"aid plugin run: no plugin {name!r} on the plugins path")
    plugin = found()
    async with connect() as control:
        registered = {spec.name: spec.grants for spec in await control.plugins()}
    if name not in registered:
        raise SystemExit(
            f"aid plugin run: no plugin {name!r} registered: `aid plugin add {name} {_grant_flags(plugin.grants)}` first"
        )
    if missing := set(plugin.grants) - set(registered[name]):
        raise SystemExit(
            f"aid plugin run: {name!r} is registered without {_grant_flags(frozenset(missing))}: "
            f"`aid plugin add {name} {_grant_flags(plugin.grants)}` again to replace it"
        )
    try:
        async with connect(plugin=name) as client:
            await plugin_host.run(plugin, client)
    except AidError as error:
        if error.code != "forbidden":
            raise
        raise SystemExit(f"aid plugin run: the daemon refused {name!r}: it was removed mid-run") from error


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
        cwd_prefix=args.cwd_prefix,
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
    if args.command == "plugin" and args.plugin_command == "run":
        configure_logging()
        try:
            anyio.run(_run_plugin, args.name)
        except AidError as error:
            raise SystemExit(f"aid plugin run: {error}") from None
        return
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
