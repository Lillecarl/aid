// Generated from protocol.schema.json (python -m aid.schema) by npm run types. Do not edit.

/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "HookMatch".
 */
export type HookMatch = string;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "McpServer".
 */
export type McpServer = McpStdio | McpHttp | McpSse;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "McpServers".
 */
export type McpServers = McpServer[];
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 */
export type PermissionMode = "allow" | "deny" | "ask";
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionMode".
 */
export type PermissionMode1 = "allow" | "deny" | "ask";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Grant".
 */
export type Grant = "read" | "prompt" | "message" | "permissions" | "manage" | "supervise";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentKind".
 */
export type AgentKind = "acp" | "native" | "claude-tty" | "claude-sdk";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentSpec".
 */
export type AgentSpec = AcpSpec | NativeSpec | ClaudeTtySpec | ClaudeSdkSpec;
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 */
export type PermissionMode2 = "allow" | "deny" | "ask";
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 */
export type PermissionMode3 = "allow" | "deny" | "ask";
/**
 * When a message reaches a session that is in a turn.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Delivery".
 */
export type Delivery = "interrupt" | "wait_for_turn";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SessionEvent".
 */
export type SessionEvent =
  TextDelta | ThoughtDelta | ToolCall | PermissionRequest | PermissionDecision | Usage | Output | RemoteControl;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionDecider".
 */
export type PermissionDecider =
  "person" | "policy" | "timeout" | "auto" | "cancel" | "terminal" | "plugin" | "supervisor";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "JsonValue".
 */
export type JsonValue = unknown;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "HistoryItem".
 */
export type HistoryItem =
  | PromptEntry
  | MessageEntry
  | Started
  | Lifecycle
  | TextDelta
  | ThoughtDelta
  | ToolCall
  | PermissionRequest
  | PermissionDecision
  | Usage
  | Output
  | RemoteControl
  | TurnError;
/**
 * When a message reaches a session that is in a turn.
 */
export type Delivery1 = "interrupt" | "wait_for_turn";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PageReply".
 */
export type PageReply = Event | Done | Failure;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PageRequest".
 */
export type PageRequest =
  | ListSessions
  | ListAgents
  | ListModels
  | CreateSession
  | GetStatus
  | GetHistory
  | GetSummary
  | GetLive
  | Prompt
  | Cancel
  | CompactSession
  | AnswerPermission
  | SetPermissions
  | SetSandbox
  | StartSession
  | StopSession
  | DeleteSession
  | ChangeDirectory
  | AnswerDirectory;

export interface AidProtocol {
  [k: string]: unknown;
}
/**
 * An external ACP agent, such as `claude-agent-acp` or `opencode acp`.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AcpSpec".
 */
export interface AcpSpec {
  aid_tools?: boolean;
  /**
   * @minItems 1
   */
  command: [string, ...string[]];
  cwd: string;
  env?: {
    [k: string]: string;
  };
  hooks?: Hook[];
  inherit_env?: boolean;
  kind: "acp";
  mcp_servers?: McpServers;
  parent?: string | null;
  permission?: PermissionMode;
  permission_timeout?: number;
  supervisor?: string | null;
  supervisor_nudge_budget?: number;
  supervisor_permission_absent_timeout?: number;
  supervisor_permission_timeout?: number;
  supervisor_permissions?: string[];
  tool_permissions?: {
    [k: string]: PermissionMode1;
  };
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
}
/**
 * A pre-tool gate: when `tool` runs with rendered text matching `match`, refuse it with `message`.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Hook".
 */
export interface Hook {
  match: HookMatch;
  message: string;
  tool: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "McpStdio".
 */
export interface McpStdio {
  /**
   * @minItems 1
   */
  command: [string, ...string[]];
  env?: {
    [k: string]: string;
  };
  name: string;
  type: "stdio";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "McpHttp".
 */
export interface McpHttp {
  headers?: {
    [k: string]: string;
  };
  name: string;
  type: "http";
  url: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "McpSse".
 */
export interface McpSse {
  headers?: {
    [k: string]: string;
  };
  name: string;
  type: "sse";
  url: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentCatalog".
 */
export interface AgentCatalog {
  agents: AgentInfo[];
  model_providers: ProviderInfo[];
  plugins: PluginInfo[];
  problems: string[];
  providers: ProviderInfo[];
  tools: ToolInfo[];
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentInfo".
 */
export interface AgentInfo {
  description: string;
  module: string;
  name: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ProviderInfo".
 */
export interface ProviderInfo {
  description: string;
  module: string;
  name: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PluginInfo".
 */
export interface PluginInfo {
  autostart: boolean;
  declined: string | null;
  description: string;
  grants: Grant[] | null;
  name: string;
  path: string;
  requires: string[];
  version: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ToolInfo".
 */
export interface ToolInfo {
  description: string;
  module: string;
  name: string;
}
/**
 * A native agent: `agent`, the name of an `aid.AgentDefinition` on the plugins path, or `target`, a
 * `package.module:attribute` that is an agent itself. Exactly one of them.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "NativeSpec".
 */
export interface NativeSpec {
  agent?: string | null;
  aid_tools?: boolean;
  ask_autoselect_after?: number;
  cwd: string;
  env?: {
    [k: string]: string;
  };
  hooks?: Hook[];
  instruction_providers?: string[] | null;
  kind: "native";
  max_context?: number | null;
  max_requests?: number | null;
  parent?: string | null;
  permission?: PermissionMode2;
  permission_timeout?: number;
  python_path?: string[];
  sandbox?: Sandbox | null;
  seed_history?: string | null;
  skip_instructions?: string[];
  supervisor?: string | null;
  supervisor_nudge_budget?: number;
  supervisor_permission_absent_timeout?: number;
  supervisor_permission_timeout?: number;
  supervisor_permissions?: string[];
  target?: string | null;
  tool_permissions?: {
    [k: string]: PermissionMode1;
  };
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
}
/**
 * bubblewrap around what a native session's tools start (`aid.sandbox`): the filesystem read-only but the
 * session's directory and `writable`, a private /tmp. A session that asks for one never runs without it: no bwrap,
 * or no user namespaces, and the worker refuses to start; `python`'s `in-loop` mode, which no sandbox can hold, is
 * refused.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Sandbox".
 */
export interface Sandbox {
  network?: boolean;
  writable?: string[];
}
/**
 * Interactive Claude Code in a pymux window: prompts are pasted in, events come from its transcript.
 *
 * A person can attach to the window, and Remote Control works, which the Agent SDK behind ACP does not offer.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ClaudeTtySpec".
 */
export interface ClaudeTtySpec {
  aid_tools?: boolean;
  args?: string[];
  /**
   * @minItems 1
   */
  command?: [string, ...string[]];
  cwd: string;
  env?: {
    [k: string]: string;
  };
  hooks?: Hook[];
  inherit_env?: boolean;
  kind: "claude-tty";
  mcp_servers?: McpServers;
  parent?: string | null;
  /**
   * @minItems 1
   */
  pymux_command?: [string, ...string[]];
  pymux_socket?: string | null;
  supervisor?: string | null;
  supervisor_nudge_budget?: number;
  supervisor_permission_absent_timeout?: number;
  supervisor_permission_timeout?: number;
  supervisor_permissions?: string[];
  tool_permissions?: {
    [k: string]: PermissionMode1;
  };
  trust_cwd?: boolean;
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
}
/**
 * Claude Code driven directly over NDJSON (`claude -p`), so sessions get Remote Control.
 *
 * Skeleton slice A: one process per prompt, resumed by the persisted session id. Permissions, Remote Control,
 * and CLI wiring arrive in later slices.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ClaudeSdkSpec".
 */
export interface ClaudeSdkSpec {
  aid_tools?: boolean;
  /**
   * @minItems 1
   */
  command?: [string, ...string[]];
  cwd: string;
  env?: {
    [k: string]: string;
  };
  hooks?: Hook[];
  inherit_env?: boolean;
  kind: "claude-sdk";
  mcp_servers?: McpServers;
  model: string;
  parent?: string | null;
  permission?: PermissionMode3;
  permission_timeout?: number;
  supervisor?: string | null;
  supervisor_nudge_budget?: number;
  supervisor_permission_absent_timeout?: number;
  supervisor_permission_timeout?: number;
  supervisor_permissions?: string[];
  tool_permissions?: {
    [k: string]: PermissionMode1;
  };
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
}
/**
 * A person's answer to a session's `RequestDirectory`.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AnswerDirectory".
 */
export interface AnswerDirectory {
  approve: boolean;
  id?: string;
  op: "cd_answer";
  session: string;
}
/**
 * A person's answer to a pending PermissionRequest: one of its options, or None to cancel it. `text`
 * is the person's own words: with an option, alongside the pick; alone, the whole answer, which only a
 * session whose agent reads answer text (the `ask_user` tool) takes.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AnswerPermission".
 */
export interface AnswerPermission {
  id?: string;
  op: "answer_permission";
  option_id: string | null;
  plugin?: string | null;
  request_id: string;
  session: string;
  supervisor?: string | null;
  text?: string | null;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Cancel".
 */
export interface Cancel {
  id?: string;
  op: "cancel";
  session: string;
  turn?: string | null;
}
/**
 * Move a session to another working directory: saved in its spec, and its worker restarted there if it runs.
 * Refused while a turn runs. A relative `cwd` is taken from the session's current one.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ChangeDirectory".
 */
export interface ChangeDirectory {
  cwd: string;
  id?: string;
  op: "cd";
  session: string;
}
/**
 * Summarize the session's history into a digest, outside a turn; the worker records a `compacted` lifecycle.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "CompactSession".
 */
export interface CompactSession {
  id?: string;
  instructions?: string;
  op: "compact";
  session: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Cost".
 */
export interface Cost {
  amount: number;
  currency: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "CreateSession".
 */
export interface CreateSession {
  id?: string;
  name: string;
  op: "create";
  spec: AgentSpec;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "DeleteSession".
 */
export interface DeleteSession {
  id?: string;
  op: "delete";
  session: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "DirCompletions".
 */
export interface DirCompletions {
  dirs: string[];
  prefix: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "DirectoryRequest".
 */
export interface DirectoryRequest {
  approved: boolean;
  cwd: string;
  reason: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Done".
 */
export interface Done {
  data: {
    [k: string]: unknown;
  };
  id: string;
  reply: "done";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Entry".
 */
export interface Entry {
  dir: boolean;
  name: string;
  size: number | null;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Event".
 */
export interface Event {
  event: SessionEvent;
  id: string;
  reply: "event";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "TextDelta".
 */
export interface TextDelta {
  text: string;
  type: "text";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ThoughtDelta".
 */
export interface ThoughtDelta {
  text: string;
  type: "thought";
}
/**
 * A tool call starting, or an update to it: fields left empty keep what an earlier event of the call said.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ToolCall".
 */
export interface ToolCall {
  diffs: ToolDiff[];
  input: {
    [k: string]: unknown;
  };
  kind: string | null;
  output: string | null;
  paths: string[];
  status: string | null;
  title: string | null;
  tool: string | null;
  tool_call_id: string;
  type: "tool_call";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ToolDiff".
 */
export interface ToolDiff {
  new: string;
  old: string | null;
  path: string;
}
/**
 * The agent asks before a tool call, and waits for a PermissionDecision. A session whose permission mode is
 * `ask` waits for a person to answer it (`AnswerPermission`). A question from the `ask_user` tool is the
 * same shape: `tool_name` is the tool, `title` the question, `options` its alternatives (possibly none).
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionRequest".
 */
export interface PermissionRequest {
  input: {
    [k: string]: unknown;
  };
  kind: string | null;
  options: PermissionChoice[];
  request_id: string;
  title: string | null;
  tool_call_id: string;
  tool_name: string | null;
  type: "permission_request";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionChoice".
 */
export interface PermissionChoice {
  kind: string;
  name: string;
  option_id: string;
  recommended: boolean;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionDecision".
 */
export interface PermissionDecision {
  by: PermissionDecider;
  option_id: string | null;
  plugin: string | null;
  request_id: string;
  supervisor: string | null;
  text: string | null;
  type: "permission_decision";
}
/**
 * Tokens a turn used, as its agent reports them; sent just before the turn's Output. None where the agent
 * says nothing.
 *
 * What "the turn" covers differs by agent (measured 2026-09):
 * - claude-agent-acp 0.75.1: every model call of the turn, subagents included.
 * - opencode 1.x: the turn's last model response only.
 * - Native: the run, every request. `input_tokens` follows the provider: Chat Completions counts cache reads
 *   in it, Anthropic's Messages API does not.
 * - claude-tty: the main conversation's messages, each counted once; subagents write transcripts of their own,
 *   which aid does not read.
 *
 * `context_used`/`context_size` come from the agent's own report for ACP and claude-tty. The native backend sets them
 * itself: `context_size` is the session's context gauge (`max_context`, half the model window by default) and
 * `context_used` is aid's rough estimate of the send view, since the engine reports no window of its own.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Usage".
 */
export interface Usage {
  agent: string | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  context_size: number | null;
  context_used: number | null;
  cost: Cost | null;
  input_tokens: number | null;
  models: string[];
  output_tokens: number | null;
  requests: number | null;
  session_cost: Cost | null;
  thought_tokens: number | null;
  type: "usage";
}
/**
 * Final event of a prompt. `output` is text for ACP and the agent's typed output for native.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Output".
 */
export interface Output {
  output: JsonValue;
  stop_reason: string;
  type: "output";
}
/**
 * Claude Code's Remote Control bridge for the turn: join it in a browser or over the connect URL.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "RemoteControl".
 */
export interface RemoteControl {
  connect_url: string;
  session_url: string;
  type: "remote_control";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Failure".
 */
export interface Failure {
  code: string;
  id: string;
  message: string;
  reply: "failure";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "FileActivity".
 */
export interface FileActivity {
  last_seq: number;
  path: string;
  reads: number;
  writes: number;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "FileView".
 */
export interface FileView {
  highlights: [number, number, string][] | null;
  path: string;
  size: number;
  text: string | null;
  truncated: boolean;
}
/**
 * A page of a session's history: the newest entries before `before`, the oldest after `after`, or the
 * newest of all when neither is given.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "GetHistory".
 */
export interface GetHistory {
  after?: number | null;
  before?: number | null;
  id?: string;
  limit?: number;
  op: "history";
  session: string;
  wait?: number;
}
/**
 * The `usertee` output the daemon still holds for a session, as a `LiveChunk`: the newest characters of the
 * turn it last came in. Never recorded, so gone with the daemon.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "GetLive".
 */
export interface GetLive {
  id?: string;
  op: "live";
  session: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "GetStatus".
 */
export interface GetStatus {
  id?: string;
  op: "status";
  session: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "GetSummary".
 */
export interface GetSummary {
  id?: string;
  op: "summary";
  session: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "HistoryEntry".
 */
export interface HistoryEntry {
  at: number;
  item: HistoryItem;
  seq: number;
  turn: string;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PromptEntry".
 */
export interface PromptEntry {
  text: string;
  type: "prompt";
}
/**
 * A message another session, or a person, sent to this one.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "MessageEntry".
 */
export interface MessageEntry {
  delivery: Delivery1;
  sender: string | null;
  text: string;
  type: "message";
}
/**
 * A worker started, and what its backend said about itself. The daemon records one per start.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Started".
 */
export interface Started {
  agent: string | null;
  agent_session: string | null;
  model: string | null;
  pid: number;
  resumed: boolean;
  type: "started";
}
/**
 * Something happened to the agent's session outside a turn: interactive Claude compacted its context,
 * cleared it (a new agent session follows, with a Started), or ended; or the session moved directory, or changed
 * its sandbox.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Lifecycle".
 */
export interface Lifecycle {
  detail: string | null;
  event: "compacted" | "cleared" | "ended" | "moved" | "sandboxed";
  summary: string | null;
  type: "lifecycle";
}
/**
 * A prompt that ended in a Failure rather than an Output.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "TurnError".
 */
export interface TurnError {
  code: string;
  message: string;
  type: "error";
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "HistoryPage".
 */
export interface HistoryPage {
  entries: HistoryEntry[];
  has_newer: boolean;
  has_older: boolean;
  total: number;
}
/**
 * The `aid.AgentDefinition`s on the daemon's plugins path.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ListAgents".
 */
export interface ListAgents {
  id?: string;
  op: "agents";
}
/**
 * Model providers (`aid.ModelProvider` plugins, aid's built-in ones, and models.dev's) and, for `provider`,
 * the models it serves. Answered from a subprocess (`python -m aid.model_lists`): the daemon imports no plugin.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ListModels".
 */
export interface ListModels {
  id?: string;
  op: "models";
  provider?: string | null;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ListSessions".
 */
export interface ListSessions {
  id?: string;
  op: "list";
}
/**
 * Output an agent piped through `usertee` (`aid/usertee.py`), for a person to watch: published on its
 * session's live topic, and the answer to `GetLive`. `turn` is the turn it came in, None outside one. Advisory
 * and never recorded: a dropped chunk is gone.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "LiveChunk".
 */
export interface LiveChunk {
  text: string;
  turn: string | null;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Prompt".
 */
export interface Prompt {
  id?: string;
  op: "prompt";
  session: string;
  text: string;
}
/**
 * Change a session's permission modes: `default`, the spec's `permission`, and `tools` by name (None returns a
 * tool to its default); every kind but claude-tty. Saved in the spec; a running worker applies them to its next
 * request (claude-sdk: its next prompt), while a request already waiting keeps its own.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SetPermissions".
 */
export interface SetPermissions {
  default?: PermissionMode1 | null;
  id?: string;
  op: "set_permissions";
  session: string;
  tools?: {
    [k: string]: PermissionMode1 | null;
  };
}
/**
 * Change a native session's sandbox (`NativeSpec.sandbox`; None turns it off): saved in its spec, and a
 * running worker restarts under it, at once when idle, else when its turn ends. Refused when bwrap cannot run
 * the new sandbox here.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SetSandbox".
 */
export interface SetSandbox {
  id?: string;
  op: "sandbox";
  sandbox: Sandbox | null;
  session: string;
}
/**
 * Start a stopped session's worker without a turn; for interactive Claude, its pane. A running one is left be.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "StartSession".
 */
export interface StartSession {
  id?: string;
  op: "start";
  session: string;
}
/**
 * Stop the worker and keep the session's state, so the next prompt starts it again.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "StopSession".
 */
export interface StopSession {
  id?: string;
  op: "stop";
  session: string;
}
/**
 * What aid web sends the page. Only the definitions matter; this model names them.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Received".
 */
export interface Received {
  agent_catalog: AgentCatalog;
  dir_completions: DirCompletions;
  file_entry: Entry;
  file_view: FileView;
  history_entry: HistoryEntry;
  history_item: HistoryItem;
  history_page: HistoryPage;
  live_chunk: LiveChunk;
  page_reply: PageReply;
  session_event: SessionEvent;
  session_info: SessionInfo;
  session_status: SessionStatus;
  session_summary: SessionSummary;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SessionInfo".
 */
export interface SessionInfo {
  attention: string | null;
  kind: AgentKind;
  name: string;
  parent: string | null;
  permissions: number;
  running: boolean;
  working: boolean;
}
/**
 * What a session is doing now, and what it runs. Never env values or MCP headers: those hold credentials.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SessionStatus".
 */
export interface SessionStatus {
  agent: string | null;
  agent_session: string | null;
  aid_tools: boolean;
  attention: string | null;
  busy: boolean;
  cost: {
    [k: string]: number;
  };
  cwd: string;
  directory_request: DirectoryRequest | null;
  kind: AgentKind;
  mcp_servers: string[];
  model: string | null;
  name: string;
  parent: string | null;
  pending: number;
  permission: PermissionMode1 | null;
  permissions: PermissionRequest[];
  pid: number | null;
  running: boolean;
  runs: string;
  sandbox: Sandbox | null;
  tools: ToolMode[];
  working: boolean;
}
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ToolMode".
 */
export interface ToolMode {
  asks: boolean;
  group: string;
  mode: PermissionMode1 | null;
  name: string;
  set: boolean;
}
/**
 * A session's whole history, totalled. Token counts sum each turn's Usage, so they cover what the agents
 * reported and nothing more (see Usage).
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SessionSummary".
 */
export interface SessionSummary {
  agent_sessions: string[];
  cache_read_tokens: number;
  cache_write_tokens: number;
  cost: {
    [k: string]: number;
  };
  files: FileActivity[];
  input_tokens: number;
  models: {
    [k: string]: number;
  };
  output_tokens: number;
  requests: number;
  starts: number;
  thought_tokens: number;
  turns: number;
}
/**
 * What the page sends aid web.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Sent".
 */
export interface Sent {
  agent_spec: AgentSpec;
  page_request: PageRequest;
}
