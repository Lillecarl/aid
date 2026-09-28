<script lang="ts">
  import { onMount, tick } from "svelte";
  import * as api from "./api";
  import type { HistoryEntry, HistoryItem, SessionEvent } from "./api";

  interface Props {
    name: string;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Kind = "user" | "message" | "assistant" | "thought" | "tool" | "meta" | "error";
  /** One line of the log. `seq` is set for rows read from history; rows of a turn still streaming lack it. */
  type Row = { key: string; seq?: number; kind: Kind; text: string };

  const PAGE = 100;
  const EDGE_PX = 200;
  const WINDOW_KEY = "aid.historyWindow";

  let { name, onchange, ondeleted }: Props = $props();

  // How many rows the page keeps; the rest stay on the server and come back on scroll.
  let windowSize = $state(Number(localStorage.getItem(WINDOW_KEY)) || 300);
  $effect(() => localStorage.setItem(WINDOW_KEY, String(windowSize)));

  let rows: Row[] = $state([]);
  let hasOlder = $state(false);
  let hasNewer = $state(false);
  let loading = $state(false);
  let text = $state("");
  let busy = $state(false);
  let log: HTMLDivElement | undefined = $state();
  let liveCount = 0;

  function rowsOf(item: HistoryItem, key: string, seq?: number): Row[] {
    const row = (kind: Kind, text: string): Row => ({ key, seq, kind, text });
    switch (item.type) {
      case "prompt":
        return [row("user", item.text)];
      case "message":
        return [row("message", `From ${item.sender ?? "a person"}: ${item.text}`)];
      case "text":
        return [row("assistant", item.text)];
      case "thought":
        return [row("thought", item.text)];
      case "tool_call":
        return [row("tool", `${item.title ?? item.tool_call_id} ${item.status ?? ""}`.trim())];
      case "output": {
        const typed = item.output !== null && typeof item.output !== "string";
        const meta = row("meta", `[${item.stop_reason}]`);
        return typed ? [row("assistant", JSON.stringify(item.output, null, 2)), { ...meta, key: `${key}m` }] : [meta];
      }
      case "error":
        return [row("error", `${item.code}: ${item.message}`)];
    }
  }

  const fromHistory = (entries: HistoryEntry[]): Row[] =>
    entries.flatMap((e) => rowsOf(e.item, `h${e.seq}`, e.seq));

  const firstSeq = (): number | undefined => rows.find((r) => r.seq !== undefined)?.seq;
  const lastSeq = (): number | undefined => rows.findLast((r) => r.seq !== undefined)?.seq;
  const nearBottom = (): boolean => !log || log.scrollHeight - log.scrollTop - log.clientHeight < EDGE_PX;

  /** Keep the view where it is while rows are added or removed above it. */
  async function keepingPosition(change: () => void): Promise<void> {
    const before = log ? log.scrollHeight - log.scrollTop : 0;
    change();
    await tick();
    if (log) log.scrollTop = log.scrollHeight - before;
  }

  async function toBottom(): Promise<void> {
    await tick();
    log?.scrollTo({ top: log.scrollHeight });
  }

  function trimTop(): void {
    if (rows.length > windowSize) {
      rows = rows.slice(rows.length - windowSize);
      hasOlder = true;
    }
  }

  async function guard(load: () => Promise<void>): Promise<void> {
    if (loading) return;
    loading = true;
    try {
      await load();
    } catch (e) {
      rows.push({ key: `e${Date.now()}`, kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      loading = false;
    }
  }

  const loadLatest = () =>
    guard(async () => {
      const page = await api.history(name, { limit: Math.min(PAGE, windowSize) });
      rows = fromHistory(page.entries);
      hasOlder = page.has_older;
      hasNewer = false;
      await toBottom();
    });

  const loadOlder = () =>
    guard(async () => {
      const before = firstSeq();
      if (before === undefined) return;
      const page = await api.history(name, { before, limit: PAGE });
      await keepingPosition(() => {
        rows = [...fromHistory(page.entries), ...rows];
        hasOlder = page.has_older;
      });
      if (rows.length > windowSize) {
        rows = rows.slice(0, windowSize);
        hasNewer = true;
      }
    });

  const loadNewer = () =>
    guard(async () => {
      const after = lastSeq();
      if (after === undefined) return;
      const page = await api.history(name, { after, limit: PAGE });
      rows = [...rows, ...fromHistory(page.entries)];
      hasNewer = page.has_newer;
      await keepingPosition(trimTop);
    });

  function onscroll(): void {
    if (!log || loading) return;
    if (log.scrollTop < EDGE_PX && hasOlder) void loadOlder();
    else if (nearBottom() && hasNewer) void loadNewer();
  }

  function apply(event: SessionEvent): void {
    const follow = nearBottom();
    const last = rows.at(-1);
    if (event.type === "text" && last?.kind === "assistant" && last.seq === undefined) {
      last.text += event.text;
    } else {
      rows.push(...rowsOf(event, `l${liveCount++}`));
      trimTop();
    }
    if (follow) void toBottom();
  }

  /** Swap the streamed rows for the entries history recorded, which carry their seq. */
  async function reconcile(): Promise<void> {
    const after = lastSeq() ?? -1;
    const page = await api.history(name, { after, limit: 1000 });
    rows = [...rows.filter((r) => r.seq !== undefined), ...fromHistory(page.entries)];
    await keepingPosition(trimTop);
    await toBottom();
  }

  async function send(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();
    const prompt = text.trim();
    if (prompt === "" || busy) return;
    text = "";
    busy = true;
    if (hasNewer) await loadLatest();
    rows.push({ key: `l${liveCount++}`, kind: "user", text: prompt });
    await toBottom();
    try {
      await api.prompt(name, prompt, apply);
    } catch (e) {
      rows.push({ key: `l${liveCount++}`, kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      busy = false;
      await reconcile().catch(() => undefined);
      await onchange();
    }
  }

  async function control(verb: "cancel" | "stop" | "delete"): Promise<void> {
    try {
      await api.control(name, verb);
      if (verb === "delete") await ondeleted();
      else await onchange();
    } catch (e) {
      rows.push({ key: `l${liveCount++}`, kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) void send();
  }

  onMount(loadLatest);
</script>

<div class="head">
  <label title="How many entries this page keeps; older ones load again when you scroll up.">
    Keep <input type="number" min="50" max="5000" step="50" bind:value={windowSize} /> entries
  </label>
</div>
<div class="log" bind:this={log} {onscroll}>
  {#if hasOlder}<div class="more">{loading ? "Loading…" : "Scroll up for older entries"}</div>{/if}
  {#each rows as row (row.key)}
    <div class={row.kind}>{row.text}</div>
  {:else}
    {#if !loading}<div class="more">No history yet.</div>{/if}
  {/each}
  {#if hasNewer}<div class="more">{loading ? "Loading…" : "Scroll down for newer entries"}</div>{/if}
</div>
<form onsubmit={send}>
  <textarea bind:value={text} onkeydown={keydown} rows="4" placeholder="Prompt (Ctrl+Enter sends)"></textarea>
  <div class="buttons">
    <button type="submit" disabled={busy}>{busy ? "Working…" : "Send"}</button>
    <button type="button" onclick={() => control("cancel")} disabled={!busy}>Cancel</button>
    <button type="button" onclick={() => control("stop")}>Stop</button>
    <button type="button" onclick={() => control("delete")}>Delete</button>
  </div>
</form>

<style>
  .head {
    display: flex;
    align-items: baseline;
    justify-content: flex-end;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }
  .head label {
    color: var(--muted);
    font-size: 0.85em;
    white-space: nowrap;
  }
  .head input {
    width: 5rem;
    display: inline;
  }
  .log {
    flex: 1;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
  }
  .log div {
    white-space: pre-wrap;
    margin: 0.25rem 0;
  }
  .more {
    text-align: center;
    color: var(--muted);
    font-size: 0.85em;
  }
  .user {
    font-weight: 600;
  }
  .message {
    font-weight: 600;
    border-left: 3px solid var(--line);
    padding-left: 0.5rem;
  }
  .thought,
  .meta,
  .tool {
    color: var(--muted);
    font-size: 0.9em;
  }
  .tool {
    font-family: ui-monospace, monospace;
  }
  .error {
    color: #d33;
  }
  .buttons {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }
  form {
    display: flex;
    flex-direction: column;
  }
</style>
