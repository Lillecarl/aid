// The aid web API. Types mirror src/aid/protocol.py and src/aid/spec.py.

export type AgentKind = "acp" | "pydantic-ai" | "claude-tty";

export interface SessionInfo {
  name: string;
  kind: AgentKind;
  running: boolean;
  /** Permission requests the agent waits on an answer to. */
  permissions: number;
  /** In a turn: aid's, or one typed into the pane. */
  working: boolean;
  /** What the agent asks a person to look at, in its own words. */
  attention: string | null;
}

export interface SessionStatus {
  name: string;
  kind: AgentKind;
  running: boolean;
  busy: boolean;
  pending: number;
  pid: number | null;
  cwd: string;
  runs: string;
  mcp_servers: string[];
  aid_tools: boolean;
  /** From the last worker start; null before the first. */
  agent_session: string | null;
  agent: string | null;
  model: string | null;
  /** Requests the agent waits on an answer to. */
  permissions: PermissionRequestEvent[];
  working: boolean;
  attention: string | null;
}

export const statusEventsUrl = (name: string): string => `/api/sessions/${encodeURIComponent(name)}/status/events`;

/** Every history entry after `after`, as the daemon records it: turns from any client, and turns typed into the
 * pane. */
export const historyEventsUrl = (name: string, after: number): string =>
  `/api/sessions/${encodeURIComponent(name)}/history/events?after=${after}`;

/**
 * Follow a stream of Server-Sent Events, one per change, while the browser tab is visible; a hidden tab closes
 * it, so nothing streams that nobody sees. Returns the function that stops following.
 */
export function watch<T>(
  url: string | (() => string),
  ondata: (data: T) => void,
  onproblem: (problem: string) => void,
): () => void {
  let source: EventSource | null = null;
  let ended = false;

  function open(): void {
    if (source !== null || ended) return;
    // A function is asked again each time the tab comes back, for a stream that resumes where the page is.
    source = new EventSource(typeof url === "string" ? url : url());
    source.onmessage = (event: MessageEvent<string>) => {
      ondata(JSON.parse(event.data) as T);
      onproblem("");
    };
    source.addEventListener("error", (event) => {
      // A server-sent `error` event carries data and ends the stream for good; a dropped connection has none,
      // and EventSource retries it.
      const data = (event as MessageEvent<string>).data;
      if (data) {
        onproblem((JSON.parse(data) as { error: string }).error);
        ended = true;
        close();
      } else {
        onproblem("reconnecting…");
      }
    });
  }

  function close(): void {
    source?.close();
    source = null;
  }

  function onvisibility(): void {
    if (document.visibilityState === "visible") open();
    else close();
  }

  onvisibility();
  document.addEventListener("visibilitychange", onvisibility);
  return () => {
    document.removeEventListener("visibilitychange", onvisibility);
    close();
  };
}

export interface ToolDiff {
  path: string;
  /** null for a new file. */
  old: string | null;
  new: string;
}

/** A tool call starting, or an update to it: empty fields keep what an earlier event of the call said. */
export interface ToolCallEvent {
  type: "tool_call";
  tool_call_id: string;
  title: string | null;
  kind: string | null;
  status: string | null;
  // Absent in history written before aid recorded them.
  input?: unknown;
  output?: string | null;
  diffs?: ToolDiff[];
  /** `path` or `path:line`. */
  paths?: string[];
}

/** A turn's tokens as its agent reports them; null where it says nothing. See `aid.protocol.Usage`. */
export interface UsageEvent {
  type: "usage";
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  thought_tokens: number | null;
  requests: number | null;
  models: string[];
  context_used: number | null;
  context_size: number | null;
  /** The agent's running total for its session, not this turn's share. */
  session_cost: { amount: number; currency: string } | null;
}

/** A worker start. */
export interface StartedEntry {
  type: "started";
  pid: number;
  agent_session: string | null;
  resumed: boolean;
  agent: string | null;
  model: string | null;
}

export interface PermissionChoice {
  option_id: string;
  name: string;
  /** allow_once, allow_always, reject_once or reject_always. */
  kind: string;
}

/** The agent waits on an answer before a tool call; see `aid.protocol.PermissionRequest`. */
export interface PermissionRequestEvent {
  type: "permission_request";
  request_id: string;
  tool_call_id: string;
  tool_name: string | null;
  title: string | null;
  kind: string | null;
  input: unknown;
  options: PermissionChoice[];
}

export interface PermissionDecisionEvent {
  type: "permission_decision";
  request_id: string;
  /** null: cancelled. */
  option_id: string | null;
  by: "person" | "policy" | "timeout" | "cancel" | "terminal";
}

export type SessionEvent =
  | { type: "text"; text: string }
  | { type: "thought"; text: string }
  | ToolCallEvent
  | PermissionRequestEvent
  | PermissionDecisionEvent
  | UsageEvent
  | { type: "output"; output: unknown; stop_reason: string };

export type HistoryItem =
  | SessionEvent
  | StartedEntry
  | { type: "prompt"; text: string }
  | { type: "message"; sender: string | null; text: string }
  | { type: "error"; code: string; message: string };

export interface FileActivity {
  path: string;
  reads: number;
  writes: number;
  last_seq: number;
}

/** A session's whole history totalled; see `aid.protocol.SessionSummary`. */
export interface SessionSummary {
  turns: number;
  starts: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  thought_tokens: number;
  requests: number;
  /** Turns per model. */
  models: Record<string, number>;
  /** Per currency, summed over agent sessions. */
  cost: Record<string, number>;
  agent_sessions: string[];
  /** Most recently touched first. */
  files: FileActivity[];
}

export interface HistoryEntry {
  seq: number;
  at: number;
  turn: string;
  item: HistoryItem;
}

export interface HistoryPage {
  entries: HistoryEntry[];
  has_older: boolean;
  has_newer: boolean;
  total: number;
}

export type PermissionMode = "ask" | "allow" | "deny";

export type AgentSpec =
  | { kind: "acp"; cwd: string; command: string[]; permission: PermissionMode }
  | { kind: "pydantic-ai"; cwd: string; agent: string }
  | { kind: "claude-tty"; cwd: string; args: string[]; trust_cwd: boolean };

export interface AgentInfo {
  name: string;
  description: string;
  module: string;
}

export interface AgentCatalog {
  agents: AgentInfo[];
  problems: string[];
}

export class ApiError extends Error {}

let csrf = "";

async function request(method: string, path: string, body?: unknown): Promise<Response> {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) {
    location.href = "/login";
    throw new ApiError("login required");
  }
  if (!response.ok) {
    const data: { error?: unknown } = await response.json().catch(() => ({}));
    throw new ApiError(typeof data.error === "string" ? data.error : JSON.stringify(data.error ?? response.statusText));
  }
  return response;
}

export async function me(): Promise<{ email: string }> {
  const data: { email: string; csrf: string } = await (await request("GET", "/api/me")).json();
  csrf = data.csrf;
  return { email: data.email };
}

export async function logout(): Promise<void> {
  await request("POST", "/logout");
}

/** The agents on the daemon's agents path. Each call imports them afresh, so edits show up. */
export async function agents(): Promise<AgentCatalog> {
  return (await request("GET", "/api/agents")).json();
}

/** Whether aid web has a speech model, so the page can offer the microphone. */
export async function speechEnabled(): Promise<boolean> {
  const data: { enabled: boolean } = await (await request("GET", "/api/speech")).json();
  return data.enabled;
}

export async function sessions(): Promise<SessionInfo[]> {
  return (await request("GET", "/api/sessions")).json();
}

export async function create(name: string, spec: AgentSpec): Promise<void> {
  await request("POST", "/api/sessions", { name, spec });
}

/** A page of a session's history: the newest before `before`, the oldest after `after`, or the newest. */
export async function history(
  name: string,
  page: { before?: number; after?: number; limit: number },
): Promise<HistoryPage> {
  const query = new URLSearchParams({ limit: String(page.limit) });
  if (page.before !== undefined) query.set("before", String(page.before));
  if (page.after !== undefined) query.set("after", String(page.after));
  return (await request("GET", `/api/sessions/${encodeURIComponent(name)}/history?${query}`)).json();
}

export interface FileEntry {
  name: string;
  dir: boolean;
  size: number | null;
}

export interface FileView {
  path: string;
  size: number;
  /** null for a binary file. */
  text: string | null;
  truncated: boolean;
  /** tree-sitter spans: UTF-16 start and end, and a Pygments class that /theme.css colours. null when aid web
   * has no grammar for the file. */
  highlights?: [number, number, string][] | null;
}

const fileUrl = (name: string, what: "files" | "file", path: string): string =>
  `/api/sessions/${encodeURIComponent(name)}/${what}?${new URLSearchParams({ path })}`;

/** A directory of the session's working directory; `path` is relative to it. */
export async function summary(name: string): Promise<SessionSummary> {
  return (await request("GET", `/api/sessions/${encodeURIComponent(name)}/summary`)).json();
}

export async function listFiles(name: string, path: string): Promise<FileEntry[]> {
  return (await request("GET", fileUrl(name, "files", path))).json();
}

export async function readFile(name: string, path: string): Promise<FileView> {
  return (await request("GET", fileUrl(name, "file", path))).json();
}

export async function control(name: string, verb: "start" | "cancel" | "stop" | "delete"): Promise<void> {
  const path = `/api/sessions/${encodeURIComponent(name)}`;
  await (verb === "delete" ? request("DELETE", path) : request("POST", `${path}/${verb}`));
}

/** Answer a permission request the session waits on: one of its options, or null to cancel it. */
export async function answerPermission(name: string, requestId: string, optionId: string | null): Promise<void> {
  const path = `/api/sessions/${encodeURIComponent(name)}/permissions/${encodeURIComponent(requestId)}`;
  await request("POST", path, { option_id: optionId });
}

/** Send a prompt and call `onEvent` for each event of the answer, which arrives as Server-Sent Events. */
export async function prompt(name: string, text: string, onEvent: (event: SessionEvent) => void): Promise<void> {
  const response = await request("POST", `/api/sessions/${encodeURIComponent(name)}/prompt`, { text });
  if (response.body === null) throw new ApiError("the answer has no body");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const kind = /^event: (.*)$/m.exec(block)?.[1] ?? "message";
      const data: unknown = JSON.parse(/^data: (.*)$/m.exec(block)?.[1] ?? "null");
      if (kind === "error") throw new ApiError((data as { error: string }).error);
      onEvent(data as SessionEvent);
    }
  }
}
