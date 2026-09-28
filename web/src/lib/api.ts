// The aid web API. Types mirror src/aid/protocol.py and src/aid/spec.py.

export type AgentKind = "acp" | "pydantic-ai" | "claude-tty";

export interface SessionInfo {
  name: string;
  kind: AgentKind;
  running: boolean;
}

export type SessionEvent =
  | { type: "text"; text: string }
  | { type: "thought"; text: string }
  | { type: "tool_call"; tool_call_id: string; title: string | null; kind: string | null; status: string | null }
  | { type: "output"; output: unknown; stop_reason: string };

export type AgentSpec =
  | { kind: "acp"; cwd: string; command: string[] }
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

export async function sessions(): Promise<SessionInfo[]> {
  return (await request("GET", "/api/sessions")).json();
}

export async function create(name: string, spec: AgentSpec): Promise<void> {
  await request("POST", "/api/sessions", { name, spec });
}

export async function control(name: string, verb: "cancel" | "stop" | "delete"): Promise<void> {
  const path = `/api/sessions/${encodeURIComponent(name)}`;
  await (verb === "delete" ? request("DELETE", path) : request("POST", `${path}/${verb}`));
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
