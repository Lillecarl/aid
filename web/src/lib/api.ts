// The aid web API. Its types are generated from the Python models: protocol.ts (npm run types).
//
// What the daemon answers goes over aid web's ZWS relays (zws.ts): requests on /api/zws/control, published changes
// on /api/zws/events. What aid web answers itself (the login, files, speech) stays plain HTTP.

import type {
  AgentCatalog,
  AgentSpec,
  Entry,
  FileView,
  HistoryEntry,
  HistoryPage,
  PageReply,
  PageRequest,
  SessionEvent,
  SessionInfo,
  SessionStatus,
  SessionSummary,
} from "./protocol";
import { ZwsSocket } from "./zws";

export type * from "./protocol";

export class ApiError extends Error {}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// The daemon's requests. -----------------------------------------------------------------------------------------

type Exchange = {
  onevent?: (event: SessionEvent) => void;
  resolve: (data: unknown) => void;
  reject: (error: Error) => void;
};

const exchanges = new Map<string, Exchange>();
let waiting: { resolve: () => void; reject: (error: Error) => void }[] = [];
let lastId = 0;

const requests = new ZwsSocket("/api/zws/control", {
  message([payload]) {
    const reply = JSON.parse(decoder.decode(payload)) as PageReply;
    const exchange = exchanges.get(reply.id);
    if (exchange === undefined) return;
    if (reply.reply === "event") {
      exchange.onevent?.(reply.event);
      return;
    }
    exchanges.delete(reply.id);
    if (reply.reply === "done") exchange.resolve(reply.data);
    else exchange.reject(new ApiError(reply.message));
  },
  open() {
    for (const w of waiting) w.resolve();
    waiting = [];
  },
  close(opened) {
    // A request in flight is lost with its WebSocket; never sent again, because a prompt must not run twice.
    for (const exchange of exchanges.values())
      exchange.reject(new ApiError("the connection to aid web dropped"));
    exchanges.clear();
    if (opened) return;
    for (const w of waiting) w.reject(new ApiError("cannot reach aid web"));
    waiting = [];
    void me().catch(() => undefined); // Sends the page to /login when the login is what is gone.
  },
});

/** Send a request to the daemon; resolve with its Done's data. `onevent` takes a prompt's events as they come. */
async function call(
  request: PageRequest,
  onevent?: (event: SessionEvent) => void,
): Promise<unknown> {
  requests.start();
  if (!requests.isOpen)
    await new Promise<void>((resolve, reject) =>
      waiting.push({ resolve, reject }),
    );
  // Unique on this WebSocket is enough: the relay gives the daemon an id of its own.
  const id = String(++lastId);
  return new Promise((resolve, reject) => {
    exchanges.set(id, { onevent, resolve, reject });
    requests.send([encoder.encode(JSON.stringify({ ...request, id }))]);
  });
}

// The daemon's published changes. ---------------------------------------------------------------------------------

type Subscriber = {
  /** A message published under the topic. */
  message: (payload: Uint8Array) => void;
  /** The subscription is (re)made: fetch what may have changed while there was none. */
  open: () => void;
};

const subscribers = new Map<string, Set<Subscriber>>();
const subscribe = (topic: string): Uint8Array => encoder.encode(`\x01${topic}`);

const published = new ZwsSocket("/api/zws/events", {
  message([topic, payload]) {
    if (topic === undefined || payload === undefined) return;
    const name = decoder.decode(topic);
    // A subscription is a prefix; the page subscribes only to whole topics.
    for (const s of subscribers.get(name) ?? []) s.message(payload);
  },
  open() {
    if (subscribers.size)
      published.send([...subscribers.keys()].map(subscribe));
    for (const set of subscribers.values()) for (const s of set) s.open();
  },
  close() {},
});

// Streams only while the browser tab is visible: a hidden tab closes the WebSocket, and a visible one opens it again
// and fetches what changed in between.
function followVisibility(): void {
  if (document.visibilityState === "visible" && subscribers.size)
    published.start();
  else published.stop();
}
document.addEventListener("visibilitychange", followVisibility);

/** Take each message published under `topic` (aid.events) while the page shows. Returns the function that stops. */
function listen(topic: string, subscriber: Subscriber): () => void {
  let set = subscribers.get(topic);
  if (set === undefined) {
    set = new Set();
    subscribers.set(topic, set);
    if (published.isOpen) published.send([subscribe(topic)]);
  }
  set.add(subscriber);
  if (published.isOpen) subscriber.open();
  else followVisibility();
  return () => {
    set.delete(subscriber);
    if (set.size) return;
    subscribers.delete(topic);
    if (published.isOpen) published.send([encoder.encode(`\x00${topic}`)]);
    followVisibility();
  };
}

/** Follow a value the daemon publishes whole on each change: `fetch` it on each subscription, then take each change.
 * Changes that arrive before the fetched value answers are applied after it: they are newer. */
function follow<T>(
  topic: string,
  fetch: () => Promise<T>,
  ondata: (data: T) => void,
  onproblem: (problem: string) => void,
): () => void {
  let held: T[] | null = null;
  return listen(topic, {
    message(payload) {
      const data = JSON.parse(decoder.decode(payload)) as T;
      if (held !== null) held.push(data);
      else ondata(data);
    },
    open() {
      const mine: T[] = [];
      held = mine;
      fetch().then(
        (data) => {
          if (held !== mine) return; // A newer subscription fetches again.
          held = null;
          for (const d of [data, ...mine]) ondata(d);
          onproblem("");
        },
        (error: unknown) => {
          if (held === mine) held = null;
          onproblem(error instanceof Error ? error.message : String(error));
        },
      );
    },
  });
}

/** The session list now and on each change. */
export const followSessions = (
  ondata: (list: SessionInfo[]) => void,
  onproblem: (problem: string) => void,
) => follow("sessions/", sessions, ondata, onproblem);

/** A session's status now and on each change. */
export const followStatus = (
  name: string,
  ondata: (status: SessionStatus) => void,
  onproblem: (problem: string) => void,
) => follow(`session/${name}/status/`, () => status(name), ondata, onproblem);

/** Each history entry as the daemon records it, whoever started its turn. `onopen` runs on each subscription: fetch
 * what was recorded while there was none. An entry the relay dropped shows as a jump in seq. */
export const followHistory = (
  name: string,
  onentry: (entry: HistoryEntry) => void,
  onopen: () => void,
) =>
  listen(`session/${name}/history/`, {
    message: (payload) =>
      onentry(JSON.parse(decoder.decode(payload)) as HistoryEntry),
    open: onopen,
  });

// Plain HTTP: what aid web answers itself. ------------------------------------------------------------------------

let csrf = "";

async function request(
  method: string,
  path: string,
  body?: unknown,
): Promise<Response> {
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
    throw new ApiError(
      typeof data.error === "string"
        ? data.error
        : JSON.stringify(data.error ?? response.statusText),
    );
  }
  return response;
}

export async function me(): Promise<{ email: string }> {
  const data: { email: string; csrf: string } = await (
    await request("GET", "/api/me")
  ).json();
  csrf = data.csrf;
  return { email: data.email };
}

export async function logout(): Promise<void> {
  await request("POST", "/logout");
}

/** Whether aid web has a speech model, so the page can offer the microphone. */
export async function speechEnabled(): Promise<boolean> {
  const data: { enabled: boolean } = await (
    await request("GET", "/api/speech")
  ).json();
  return data.enabled;
}

const fileUrl = (name: string, what: "files" | "file", path: string): string =>
  `/api/sessions/${encodeURIComponent(name)}/${what}?${new URLSearchParams({ path })}`;

/** A directory of the session's working directory; `path` is relative to it. */
export async function listFiles(name: string, path: string): Promise<Entry[]> {
  return (await request("GET", fileUrl(name, "files", path))).json();
}

export async function readFile(name: string, path: string): Promise<FileView> {
  return (await request("GET", fileUrl(name, "file", path))).json();
}

// The daemon's answers, typed. -------------------------------------------------------------------------------------

/** The agents on the daemon's agents path. Each call imports them afresh, so edits show up. */
export const agents = async (): Promise<AgentCatalog> =>
  (await call({ op: "agents" })) as AgentCatalog;

export const sessions = async (): Promise<SessionInfo[]> =>
  (await call({ op: "list" })) as SessionInfo[];

export const status = async (name: string): Promise<SessionStatus> =>
  (await call({ op: "status", session: name })) as SessionStatus;

export async function create(name: string, spec: AgentSpec): Promise<void> {
  await call({ op: "create", name, spec });
}

/** A page of a session's history: the newest before `before`, the oldest after `after`, or the newest. */
export const history = async (
  name: string,
  page: { before?: number; after?: number; limit: number },
): Promise<HistoryPage> =>
  (await call({ op: "history", session: name, ...page })) as HistoryPage;

export const summary = async (name: string): Promise<SessionSummary> =>
  (await call({ op: "summary", session: name })) as SessionSummary;

export async function control(
  name: string,
  verb: "start" | "cancel" | "stop" | "delete",
): Promise<void> {
  await call({ op: verb, session: name });
}

/** Answer a permission request the session waits on: one of its options, or null to cancel it. */
export async function answerPermission(
  name: string,
  requestId: string,
  optionId: string | null,
): Promise<void> {
  await call({
    op: "answer_permission",
    session: name,
    request_id: requestId,
    option_id: optionId,
  });
}

/** Send a prompt and call `onEvent` for each event of the answer. */
export async function prompt(
  name: string,
  text: string,
  onEvent: (event: SessionEvent) => void,
): Promise<void> {
  await call({ op: "prompt", session: name, text }, onEvent);
}
