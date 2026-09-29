<script lang="ts">
  import { onMount, tick } from "svelte";
  import * as api from "./api";
  import type { HistoryEntry, HistoryItem, SessionEvent } from "./api";
  import { Dictation } from "./dictation";
  import Markdown from "./Markdown.svelte";
  import ToolCard from "./ToolCard.svelte";
  import { merge, type Tool, toolOf } from "./tools";
  import { startedLine, usageLine } from "./usage";

  interface Props {
    name: string;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Kind = "user" | "message" | "assistant" | "thought" | "tool" | "meta" | "error";
  /** One entry of the log. `seq` is set for rows read from history; rows of a turn still streaming lack it. A
   * tool call is one row, which later updates of the same call change. */
  type Row = { key: string; seq?: number; kind: Kind; text: string; tool?: Tool };

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
        return [{ ...row("tool", ""), tool: toolOf(item) }];
      case "output": {
        const typed = item.output !== null && typeof item.output !== "string";
        const meta = row("meta", `[${item.stop_reason}]`);
        return typed ? [row("assistant", JSON.stringify(item.output, null, 2)), { ...meta, key: `${key}m` }] : [meta];
      }
      case "error":
        return [row("error", `${item.code}: ${item.message}`)];
      case "usage":
        return [row("meta", usageLine(item))];
      case "started":
        return [row("meta", startedLine(item))];
    }
  }

  /** Add an item's rows to `into`: text and thought deltas join the row before them, and a tool call's update
   * changes the row of its call. */
  function append(into: Row[], item: HistoryItem, key: string, seq?: number): void {
    const last = into.at(-1);
    if ((item.type === "text" || item.type === "thought") && seq === undefined && last?.seq === undefined) {
      const kind = item.type === "text" ? "assistant" : "thought";
      if (last?.kind === kind) {
        last.text += item.text;
        return;
      }
    }
    if (item.type === "tool_call") {
      const call = into.findLast((r) => r.tool?.id === item.tool_call_id)?.tool;
      if (call) {
        merge(call, item);
        return;
      }
    }
    into.push(...rowsOf(item, key, seq));
  }

  function fromHistory(entries: HistoryEntry[]): Row[] {
    const out: Row[] = [];
    for (const e of entries) append(out, e.item, `h${e.seq}`, e.seq);
    return out;
  }

  const firstSeq = (): number | undefined => rows.find((r) => r.seq !== undefined)?.seq;
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
    if (behind && !busy && !hasNewer) await reconcile().catch(() => undefined);
  }

  /** The newest history entry the rows hold. Not the last row's seq: a tool call's updates join its first row. */
  let newest = -1;
  /** An entry arrived while the rows could not take it: fetch it once they can. */
  let behind = false;

  const loadLatest = () =>
    guard(async () => {
      const page = await api.history(name, { limit: Math.min(PAGE, windowSize) });
      rows = fromHistory(page.entries);
      newest = page.entries.at(-1)?.seq ?? -1;
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
      const page = await api.history(name, { after: newest, limit: PAGE });
      rows = [...rows, ...fromHistory(page.entries)];
      newest = page.entries.at(-1)?.seq ?? newest;
      hasNewer = page.has_newer;
      await keepingPosition(trimTop);
    });

  /** An entry the history stream sent: a turn this page did not send, such as one typed into the pane. */
  function arrived(entry: HistoryEntry): void {
    if (entry.seq <= newest || hasNewer) return; // Seen already, or the rows end before it: scrolling fetches it.
    if (busy || loading) {
      // This page's own turn is streaming, or a page is loading; either one fetches what it missed once done.
      behind = true;
      return;
    }
    newest = entry.seq;
    const follow = nearBottom();
    append(rows, entry.item, `h${entry.seq}`, entry.seq);
    trimTop();
    if (follow) void toBottom();
  }

  function onscroll(): void {
    if (!log || loading) return;
    if (log.scrollTop < EDGE_PX && hasOlder) void loadOlder();
    else if (nearBottom() && hasNewer) void loadNewer();
  }

  function apply(event: SessionEvent): void {
    const follow = nearBottom();
    append(rows, event, `l${liveCount++}`);
    trimTop();
    if (follow) void toBottom();
  }

  /** Swap the streamed rows for the entries history recorded, which carry their seq. */
  async function reconcile(): Promise<void> {
    behind = false;
    const page = await api.history(name, { after: newest, limit: 1000 });
    rows = [...rows.filter((r) => r.seq !== undefined), ...fromHistory(page.entries)];
    newest = page.entries.at(-1)?.seq ?? newest;
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

  // Speech to text: what is heard goes into the prompt box, to edit before sending.
  let canDictate = $state(false);
  let dictation: Dictation | null = $state(null);
  let stopping = $state(false);

  async function toggleDictation(): Promise<void> {
    if (dictation !== null) {
      stopping = true;
      try {
        await dictation.stop();
      } finally {
        dictation = null;
        stopping = false;
      }
      return;
    }
    // Finished utterances join what was typed; the latest guess follows them until it is final.
    const before = text.trimEnd();
    let heard = "";
    const show = (guess: string): void => {
      text = [before, heard, guess].filter((part) => part !== "").join(" ");
    };
    const started = new Dictation((h) => {
      if (h.final) {
        heard = [heard, h.text].filter((part) => part !== "").join(" ");
        show("");
      } else {
        show(h.text);
      }
    });
    try {
      await started.start();
      dictation = started;
    } catch (e) {
      rows.push({ key: `l${liveCount++}`, kind: "error", text: e instanceof Error ? e.message : String(e) });
      await started.stop().catch(() => undefined);
    }
  }

  onMount(() => {
    let unwatch: (() => void) | null = null;
    let mounted = true;
    // Streams while this tab is mounted and the browser tab visible, from wherever the rows end then.
    void loadLatest().then(() => {
      if (mounted) unwatch = api.watch(() => api.historyEventsUrl(name, newest), arrived, () => undefined);
    });
    api.speechEnabled().then(
      (enabled) => (canDictate = enabled),
      () => (canDictate = false),
    );
    return () => {
      mounted = false;
      unwatch?.();
      void dictation?.stop();
    };
  });
</script>

<div class="head">
  <label title="How many entries this page keeps; older ones load again when you scroll up.">
    Keep <input type="number" min="50" max="5000" step="50" bind:value={windowSize} /> entries
  </label>
</div>
<div class="log" bind:this={log} {onscroll}>
  {#if hasOlder}<div class="more">{loading ? "Loading…" : "Scroll up for older entries"}</div>{/if}
  {#each rows as row (row.key)}
    {#if row.kind === "assistant"}
      <div class="assistant"><Markdown text={row.text} /></div>
    {:else if row.kind === "thought"}
      <details class="thought">
        <summary>Thinking <span class="gist">{row.text.trim().split("\n", 1)[0]}</span></summary>
        <Markdown text={row.text} />
      </details>
    {:else if row.tool}
      <ToolCard tool={row.tool} />
    {:else}
      <div class={row.kind}>{row.text}</div>
    {/if}
  {:else}
    {#if !loading}<div class="more">No history yet.</div>{/if}
  {/each}
  {#if hasNewer}<div class="more">{loading ? "Loading…" : "Scroll down for newer entries"}</div>{/if}
</div>
<form onsubmit={send}>
  <textarea bind:value={text} onkeydown={keydown} rows="4" placeholder="Prompt (Ctrl+Enter sends)"></textarea>
  <div class="buttons">
    <button type="submit" disabled={busy}>{busy ? "Working…" : "Send"}</button>
    {#if canDictate}
      <button
        type="button"
        class:listening={dictation !== null}
        onclick={toggleDictation}
        disabled={stopping}
        title="Speech to text: what you say goes into the prompt, to edit before sending"
        >{stopping ? "Finishing…" : dictation !== null ? "Stop listening" : "Dictate"}</button
      >
    {/if}
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
  .log > * {
    margin: 0.4rem 0;
  }
  .more {
    text-align: center;
    color: var(--muted);
    font-size: 0.85em;
  }
  .user,
  .message {
    white-space: pre-wrap;
    background: var(--code-bg);
    border-left: 3px solid var(--accent);
    padding: 0.4rem 0.6rem;
    border-radius: 0 0.3rem 0.3rem 0;
    margin-top: 1rem;
  }
  .message {
    border-left-color: var(--muted);
  }
  .thought {
    color: var(--muted);
    font-size: 0.9em;
  }
  .thought summary {
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .thought[open] .gist {
    display: none;
  }
  .gist {
    font-style: italic;
    margin-left: 0.4em;
  }
  .thought[open] {
    border-left: 2px solid var(--line);
    padding-left: 0.6rem;
  }
  .meta {
    color: var(--muted);
    font-size: 0.8em;
    text-align: right;
  }
  .error {
    white-space: pre-wrap;
    color: var(--bad);
  }
  .listening {
    color: var(--bad);
    border-color: var(--bad);
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
