<script lang="ts">
  import { onMount, tick } from "svelte";
  import * as api from "./api";
  import type {
    HistoryEntry,
    HistoryItem,
    Lifecycle,
    PermissionDecision,
    PermissionRequest,
    SessionEvent,
  } from "./api";
  import { Dictation } from "./dictation";
  import { editLine } from "./lineedit";
  import Markdown from "./Markdown.svelte";
  import PermissionCard from "./PermissionCard.svelte";
  import ToolCard from "./ToolCard.svelte";
  import { merge, type Tool, toolOf } from "./tools";
  import { startedLine, usageLine } from "./usage";

  interface Props {
    name: string;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Kind = "user" | "message" | "assistant" | "thought" | "summary" | "tool" | "permission" | "meta" | "error";
  const LIFECYCLE: Record<Lifecycle["event"], string> = {
    compacted: "Context compacted",
    cleared: "Context cleared: a new session follows",
    ended: "Claude exited",
  };
  type Permission = { request: PermissionRequest; decision: PermissionDecision | null };
  /** One entry of the log. `seq` is set for rows read from history; rows of a turn still streaming lack it. A
   * tool call is one row, which later updates of the same call change; so is a permission request and its
   * decision. `local` marks rows the page synthesized that history will never contain (slash output, errors):
   * reconcile keeps those and swaps the rest for what history recorded. */
  type Row = {
    key: string;
    seq?: number;
    kind: Kind;
    text: string;
    tool?: Tool;
    permission?: Permission;
    local?: boolean;
  };

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
      case "permission_request":
        return [{ ...row("permission", ""), permission: { request: item, decision: null } }];
      case "permission_decision":
        // Its request is outside the rows loaded.
        return [row("meta", `permission ${item.option_id ?? "cancelled"} (${item.plugin ?? item.by})`)];
      case "lifecycle": {
        const line = row("meta", LIFECYCLE[item.event] + (item.detail ? ` (${item.detail})` : ""));
        return item.summary ? [line, { ...row("summary", item.summary), key: `${key}s` }] : [line];
      }
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
    if (item.type === "permission_decision") {
      const asked = into.findLast((r) => r.permission?.request.request_id === item.request_id)?.permission;
      if (asked) {
        asked.decision = item;
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
      rows.push({ key: `e${Date.now()}`, kind: "error", local: true, text: e instanceof Error ? e.message : String(e) });
    } finally {
      loading = false;
    }
    if (behind && !busy && !hasNewer) await reconcile().catch(() => undefined);
  }

  /** The newest history entry the rows hold. Not the last row's seq: a tool call's updates join its first row. */
  let newest = -1;
  /** An entry arrived while the rows could not take it: fetch it once they can. */
  let behind = false;
  let catchingUp = false;

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
    if (busy || loading || catchingUp || entry.seq !== newest + 1) {
      // This page's own turn is streaming, a page is loading, or entries before this one are not here yet: fetch
      // what is missing in order, never skip ahead of it.
      behind = true;
      if (!busy && !loading) void reconcile().catch(() => undefined);
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

  /** Swap the streamed rows for the entries history recorded, which carry their seq, and fetch any the history
   * stream sent while the rows could not take them. Rows the page synthesized stay: history will never contain
   * them. One runs at a time; `arrived` waits for it. */
  async function reconcile(): Promise<void> {
    if (catchingUp) {
      behind = true;
      return;
    }
    catchingUp = true;
    try {
      do {
        behind = false;
        const page = await api.history(name, { after: newest, limit: 1000 });
        rows = [...rows.filter((r) => r.seq !== undefined || r.local), ...fromHistory(page.entries)];
        newest = page.entries.at(-1)?.seq ?? newest;
        behind ||= page.has_newer;
      } while (behind);
    } finally {
      catchingUp = false;
    }
    await keepingPosition(trimTop);
    await toBottom();
  }

  async function send(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();
    const prompt = text.trim();
    if (prompt === "" || busy) return;
    text = "";
    if (prompt.startsWith("/")) {
      await slash(prompt);
      return;
    }
    busy = true;
    if (hasNewer) await loadLatest();
    rows.push({ key: `l${liveCount++}`, kind: "user", text: prompt });
    await toBottom();
    try {
      await api.prompt(name, prompt, apply);
    } catch (e) {
      rows.push({ key: `l${liveCount++}`, kind: "error", local: true, text: e instanceof Error ? e.message : String(e) });
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
      rows.push({ key: `l${liveCount++}`, kind: "error", local: true, text: e instanceof Error ? e.message : String(e) });
    }
  }

  /** Slash commands: parsed here, run against the daemon; the digest a command returns arrives through history. */
  type Slash = {
    name: string;
    usage: string;
    description: string;
    run: (args: string) => Promise<void>;
  };
  const slashes: Slash[] = [
    {
      name: "help",
      usage: "/help",
      description: "List these commands",
      run: async () => {
        for (const s of slashes)
          rows.push({ key: `l${liveCount++}`, kind: "meta", local: true, text: `${s.usage} — ${s.description}` });
      },
    },
    {
      name: "compact",
      usage: "/compact [focus]",
      description: "Summarize history into a digest outside a turn",
      run: async (args: string) => {
        await api.compact(name, args === "" ? undefined : args);
      },
    },
  ];

  async function slash(line: string): Promise<void> {
    const [head = "", ...rest] = line.split(/\s+/);
    const command = slashes.find((s) => s.name === head.slice(1));
    rows.push({ key: `l${liveCount++}`, kind: "user", text: line });
    await toBottom();
    if (command === undefined) {
      rows.push({ key: `l${liveCount++}`, kind: "error", local: true, text: `unknown slash command ${head} (try /help)` });
      return;
    }
    busy = true;
    try {
      await command.run(rest.join(" "));
    } catch (e) {
      rows.push({ key: `l${liveCount++}`, kind: "error", local: true, text: e instanceof Error ? e.message : String(e) });
    } finally {
      busy = false;
      await reconcile().catch(() => undefined);
      await onchange();
    }
  }

  // Slash completion: a prompt box holding only "/prefix" offers the commands it starts, from the same array
  // /help lists. Enter or Tab completes the selected one; Escape closes until the text changes again.
  let slashSel = $state(0);
  let slashOff = $state(false);
  const slashQuery = $derived(/^\s*\/([A-Za-z]*)$/.exec(text)?.[1] ?? null);
  const slashMatches = $derived(slashes.filter((s) => slashQuery !== null && s.name.startsWith(slashQuery)));
  const slashOpen = $derived(slashQuery !== null && slashMatches.length > 0 && !slashOff);
  $effect(() => {
    slashQuery;
    slashSel = 0;
    slashOff = false;
  });

  function slashAccept(command: Slash): void {
    text = `/${command.name} `;
    slashOff = true;
  }

  // A touch keyboard has no Shift+Enter, so there Enter is a newline and the button sends.
  const touchOnly = matchMedia("(pointer: coarse) and (not (any-pointer: fine))").matches;

  function newline(area: HTMLTextAreaElement): void {
    area.setRangeText("\n", area.selectionStart, area.selectionEnd, "end");
    area.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function keydown(event: KeyboardEvent): void {
    // keyCode 229: Safari ends an IME composition with an Enter whose isComposing is already false.
    if (event.isComposing || event.keyCode === 229) return;
    const area = event.currentTarget as HTMLTextAreaElement;
    if (editLine(event, area)) return;
    if (slashOpen) {
      const selected = slashMatches[Math.min(slashSel, slashMatches.length - 1)];
      if (selected === undefined) return;
      if (event.key === "Escape") {
        event.preventDefault();
        slashOff = true;
        return;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const n = slashMatches.length;
        slashSel = (slashSel + (event.key === "ArrowDown" ? 1 : n - 1)) % n;
        return;
      }
      if (event.key === "Tab") {
        event.preventDefault();
        slashAccept(selected);
        return;
      }
      if (event.key === "Enter" && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        slashAccept(selected);
        return;
      }
    }
    if (event.ctrlKey && !event.altKey && !event.metaKey && event.key.toLowerCase() === "j") {
      event.preventDefault(); // Chromium opens its downloads page otherwise.
      newline(area);
      return;
    }
    if (event.key !== "Enter") return;
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      void send();
    } else if (event.altKey) {
      event.preventDefault();
      newline(area);
    } else if (!event.shiftKey && !touchOnly) {
      event.preventDefault();
      void send();
    }
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
      rows.push({ key: `l${liveCount++}`, kind: "error", local: true, text: e instanceof Error ? e.message : String(e) });
      await started.stop().catch(() => undefined);
    }
  }

  onMount(() => {
    let unwatch: (() => void) | null = null;
    let mounted = true;
    // Streams while this tab is mounted and the browser tab visible. Each subscription fetches what was recorded
    // since the rows end, as a jump in seq does.
    void loadLatest().then(() => {
      if (!mounted) return;
      unwatch = api.followHistory(name, arrived, () => {
        if (hasNewer) return; // The rows end before the newest entries: scrolling fetches them.
        behind = true;
        if (!busy && !loading) void reconcile().catch(() => undefined);
      });
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
<div class="log" data-testid="chat-log" bind:this={log} {onscroll}>
  {#if hasOlder}<div class="more">{loading ? "Loading…" : "Scroll up for older entries"}</div>{/if}
  {#each rows as row (row.key)}
    {#if row.kind === "assistant"}
      <div class="assistant" data-testid="chat-row"><Markdown text={row.text} /></div>
    {:else if row.kind === "thought"}
      <details class="thought" data-testid="chat-row">
        <summary>Thinking <span class="gist">{row.text.trim().split("\n", 1)[0]}</span></summary>
        <Markdown text={row.text} />
      </details>
    {:else if row.kind === "summary"}
      <details class="thought" data-testid="chat-row">
        <summary>What Claude kept of the conversation</summary>
        <Markdown text={row.text} />
      </details>
    {:else if row.tool}
      <ToolCard tool={row.tool} />
    {:else if row.permission}
      <PermissionCard session={name} request={row.permission.request} decision={row.permission.decision} />
    {:else}
      <div class={row.kind} data-testid="chat-row">{row.text}</div>
    {/if}
  {:else}
    {#if !loading}<div class="more">No history yet.</div>{/if}
  {/each}
  {#if hasNewer}<div class="more">{loading ? "Loading…" : "Scroll down for newer entries"}</div>{/if}
</div>
<form onsubmit={send}>
  {#if slashOpen}
    <div class="slashmenu" role="listbox" aria-label="Slash commands" data-testid="slash-menu">
      {#each slashMatches as s, i}
        <button
          type="button"
          role="option"
          aria-selected={i === slashSel}
          class:selected={i === slashSel}
          data-testid="slash-item"
          onmousedown={(e) => e.preventDefault()}
          onmouseover={() => (slashSel = i)}
          onfocus={() => (slashSel = i)}
          onclick={() => slashAccept(s)}
          ><b>{s.usage}</b><span>{s.description}</span></button
        >
      {/each}
    </div>
  {/if}
  <textarea data-testid="chat-input" bind:value={text} onkeydown={keydown} rows="4" placeholder={touchOnly ? "Prompt" : "Prompt (Enter sends, Shift+Enter or Ctrl+J for a new line, /help for commands)"}></textarea>
  <div class="buttons">
    <button type="submit" data-testid="chat-send" disabled={busy}>{busy ? "Working…" : "Send"}</button>
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
    position: relative;
  }
  .slashmenu {
    position: absolute;
    bottom: 100%;
    left: 0;
    margin-bottom: 0.25rem;
    min-width: 22rem;
    background: var(--code-bg);
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.25rem;
  }
  .slashmenu button {
    display: flex;
    gap: 0.75rem;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    border-radius: 0.2rem;
    padding: 0.3rem 0.5rem;
    cursor: pointer;
  }
  .slashmenu button.selected {
    background: var(--accent);
  }
  .slashmenu button span {
    color: var(--muted);
  }
</style>
