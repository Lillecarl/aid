import type { ToolCallEvent, ToolDiff } from "./api";

/** A tool call as the log shows it: its first event, with the later ones merged in. */
export interface Tool {
  id: string;
  title: string | null;
  kind: string | null;
  status: string | null;
  input: unknown;
  output: string | null;
  diffs: ToolDiff[];
  paths: string[];
}

export const toolOf = (e: ToolCallEvent): Tool => ({
  id: e.tool_call_id,
  title: e.title,
  kind: e.kind,
  status: e.status,
  input: e.input ?? null,
  output: e.output ?? null,
  diffs: e.diffs ?? [],
  paths: e.paths ?? [],
});

/** Apply an update of the call to what the log holds. */
export function merge(tool: Tool, e: ToolCallEvent): void {
  tool.title = e.title ?? tool.title;
  tool.kind = e.kind ?? tool.kind;
  tool.status = e.status ?? tool.status;
  if (e.input !== undefined && e.input !== null) tool.input = e.input;
  tool.output = e.output ?? tool.output;
  if (e.diffs?.length) tool.diffs = e.diffs;
  if (e.paths?.length) tool.paths = e.paths;
}

// Argument names that say what a call is about, in the order Claude Code's and common MCP tools use them.
const GIST_KEYS = ["command", "file_path", "path", "pattern", "url", "query", "description", "to", "prompt"];

/** A one-line hint from the arguments, for a title that is only the tool's name. */
export function gist(input: unknown): string | null {
  if (input === null || typeof input !== "object" || Array.isArray(input)) return null;
  const args = input as Record<string, unknown>;
  for (const key of GIST_KEYS) {
    const value = args[key];
    if (typeof value === "string" && value.trim() !== "") return value.trim().split("\n", 1)[0] ?? null;
  }
  return null;
}

/** Arguments worth showing: not an empty object or list. */
export function hasInput(input: unknown): boolean {
  if (input === null || input === undefined) return false;
  if (typeof input === "object") return Object.keys(input).length > 0;
  return true;
}

/** Output as the card shows it: JSON indented, anything else as it came. */
export function readable(output: string): string {
  const start = output.trimStart()[0];
  if (start !== "{" && start !== "[") return output;
  try {
    return JSON.stringify(JSON.parse(output), null, 2);
  } catch {
    return output;
  }
}

/** The shell command of a call, when its only interesting argument is one. */
export function command(input: unknown): string | null {
  if (input === null || typeof input !== "object") return null;
  const value = (input as Record<string, unknown>).command;
  return typeof value === "string" ? value : null;
}

/** `…` stands for unchanged lines left out. */
export type DiffLine = { op: " " | "-" | "+" | "…"; text: string };

// Past this many line pairs the LCS table costs too much memory; show both sides whole instead.
const DIFF_CELLS = 4_000_000;

/** A line diff by longest common subsequence, with `context` unchanged lines kept around each change. */
export function diffLines(old: string | null, next: string, context = 3): DiffLine[] {
  const a = old === null || old === "" ? [] : old.replace(/\n$/, "").split("\n");
  const b = next === "" ? [] : next.replace(/\n$/, "").split("\n");
  if (a.length * b.length > DIFF_CELLS) {
    return [...a.map((text) => ({ op: "-" as const, text })), ...b.map((text) => ({ op: "+" as const, text }))];
  }
  const cols = b.length + 1;
  const lcs = new Uint32Array((a.length + 1) * cols);
  const at = (i: number, j: number): number => lcs[i * cols + j] ?? 0;
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i * cols + j] = a[i] === b[j] ? at(i + 1, j + 1) + 1 : Math.max(at(i + 1, j), at(i, j + 1));
    }
  }
  const all: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    const left = a[i];
    const right = b[j];
    if (left !== undefined && left === right) {
      all.push({ op: " ", text: left });
      i++;
      j++;
    } else if (left !== undefined && (right === undefined || at(i + 1, j) >= at(i, j + 1))) {
      all.push({ op: "-", text: left });
      i++;
    } else {
      all.push({ op: "+", text: right ?? "" });
      j++;
    }
  }
  const near = all.map((_, k) => all.slice(Math.max(0, k - context), k + context + 1).some((l) => l.op !== " "));
  const out: DiffLine[] = [];
  all.forEach((line, k) => {
    if (near[k]) out.push(line);
    else if (out.at(-1)?.op !== "…") out.push({ op: "…", text: "" });
  });
  return out;
}
