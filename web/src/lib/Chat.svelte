<script lang="ts">
  import { tick } from "svelte";
  import * as api from "./api";
  import type { SessionEvent } from "./api";

  interface Props {
    name: string;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Entry = { kind: "user" | "assistant" | "thought" | "tool" | "meta" | "error"; text: string };

  let { name, onchange, ondeleted }: Props = $props();

  let entries: Entry[] = $state([]);
  let text = $state("");
  let busy = $state(false);
  let log: HTMLDivElement | undefined = $state();

  function add(entry: Entry): void {
    entries.push(entry);
  }

  function apply(event: SessionEvent): void {
    switch (event.type) {
      case "text": {
        const last = entries.at(-1);
        if (last?.kind === "assistant") last.text += event.text;
        else add({ kind: "assistant", text: event.text });
        break;
      }
      case "thought":
        add({ kind: "thought", text: event.text });
        break;
      case "tool_call":
        add({ kind: "tool", text: `${event.title ?? event.tool_call_id} ${event.status ?? ""}`.trim() });
        break;
      case "output":
        if (event.output !== null && typeof event.output !== "string") {
          add({ kind: "assistant", text: JSON.stringify(event.output, null, 2) });
        }
        add({ kind: "meta", text: `[${event.stop_reason}]` });
        break;
    }
  }

  $effect(() => {
    void entries.length;
    tick().then(() => log?.scrollTo({ top: log.scrollHeight }));
  });

  async function send(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();
    const prompt = text.trim();
    if (prompt === "" || busy) return;
    text = "";
    busy = true;
    add({ kind: "user", text: prompt });
    try {
      await api.prompt(name, prompt, apply);
    } catch (e) {
      add({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      busy = false;
      await onchange();
    }
  }

  async function control(verb: "cancel" | "stop" | "delete"): Promise<void> {
    try {
      await api.control(name, verb);
      if (verb === "delete") await ondeleted();
      else await onchange();
    } catch (e) {
      add({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) void send();
  }
</script>

<h2>{name}</h2>
<div class="log" bind:this={log}>
  {#each entries as entry, i (i)}
    <div class={entry.kind}>{entry.text}</div>
  {/each}
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
  .user {
    font-weight: 600;
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
