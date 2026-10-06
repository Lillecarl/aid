// Generated from protocol.schema.json (python -m aid.schema) by npm run types. Do not edit.

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
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentKind".
 */
export type AgentKind = "acp" | "pydantic-ai" | "claude-tty";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "AgentSpec".
 */
export type AgentSpec = AcpSpec | PydanticAISpec | ClaudeTtySpec;
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 */
export type PermissionMode1 = "allow" | "deny" | "ask";
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "SessionEvent".
 */
export type SessionEvent =
  TextDelta | ThoughtDelta | ToolCall | PermissionRequest | PermissionDecision | Usage | Output;
/**
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionDecider".
 */
export type PermissionDecider = "person" | "policy" | "timeout" | "auto" | "cancel" | "terminal" | "plugin";
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
  | TurnError;
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
  | CreateSession
  | GetStatus
  | GetHistory
  | GetSummary
  | Prompt
  | Cancel
  | CompactSession
  | AnswerPermission
  | StartSession
  | StopSession
  | DeleteSession;
/**
 * How aid answers what an agent asks permission for: an ACP agent's requests, `aid.coding`'s commands.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PermissionMode".
 */
export type PermissionMode2 = "allow" | "deny" | "ask";

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
  inherit_env?: boolean;
  kind: "acp";
  mcp_servers?: McpServers;
  permission?: PermissionMode;
  permission_timeout?: number;
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
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
  problems: string[];
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
 * via the `definition` "ToolInfo".
 */
export interface ToolInfo {
  description: string;
  module: string;
  name: string;
}
/**
 * A pydantic-ai agent: `agent`, the name of an `aid.PydanticAgent` on the agents path, or `target`, a
 * `package.module:attribute` that is an agent itself. Exactly one of them.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "PydanticAISpec".
 */
export interface PydanticAISpec {
  agent?: string | null;
  aid_tools?: boolean;
  ask_autoselect_after?: number;
  cwd: string;
  env?: {
    [k: string]: string;
  };
  kind: "pydantic-ai";
  max_context?: number | null;
  permission?: PermissionMode1;
  permission_timeout?: number;
  python_path?: string[];
  target?: string | null;
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
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
  inherit_env?: boolean;
  kind: "claude-tty";
  mcp_servers?: McpServers;
  /**
   * @minItems 1
   */
  pymux_command?: [string, ...string[]];
  pymux_socket?: string | null;
  trust_cwd?: boolean;
  worker_ca?: string | null;
  worker_command?: [string, ...string[]] | null;
  worker_endpoint?: string | null;
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
 * - pydantic-ai: the run, every request.
 * - claude-tty: the main conversation's messages, each counted once; subagents write transcripts of their own,
 *   which aid does not read.
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
  input_tokens: number | null;
  models: string[];
  output_tokens: number | null;
  requests: number | null;
  session_cost: Cost | null;
  thought_tokens: number | null;
  type: "usage";
}
/**
 * Final event of a prompt. `output` is text for ACP and the agent's typed output for pydantic-ai.
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
 * cleared it (a new agent session follows, with a Started), or ended.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "Lifecycle".
 */
export interface Lifecycle {
  detail: string | null;
  event: "compacted" | "cleared" | "ended";
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
 * The `aid.PydanticAgent`s on the daemon's agents path.
 *
 * This interface was referenced by `AidProtocol`'s JSON-Schema
 * via the `definition` "ListAgents".
 */
export interface ListAgents {
  id?: string;
  op: "agents";
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
  file_entry: Entry;
  file_view: FileView;
  history_entry: HistoryEntry;
  history_item: HistoryItem;
  history_page: HistoryPage;
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
  kind: AgentKind;
  mcp_servers: string[];
  model: string | null;
  name: string;
  pending: number;
  permissions: PermissionRequest[];
  pid: number | null;
  running: boolean;
  runs: string;
  working: boolean;
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
