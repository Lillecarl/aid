<script lang="ts">
  import type { AgentKind } from "./api";
  import Chat from "./Chat.svelte";
  import Status from "./Status.svelte";
  import Terminal from "./Terminal.svelte";

  interface Props {
    name: string;
    kind: AgentKind | undefined;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Tab = "chat" | "terminal" | "status";
  const TAB_KEY = "aid.sessionTab";

  let { name, kind, onchange, ondeleted }: Props = $props();

  const tabs = $derived<{ id: Tab; label: string }[]>([
    { id: "chat", label: "Chat" },
    ...(kind === "claude-tty" ? [{ id: "terminal" as const, label: "Terminal" }] : []),
    { id: "status", label: "Status" },
  ]);

  function stored(): Tab {
    try {
      const value = localStorage.getItem(TAB_KEY);
      return value === "terminal" || value === "status" ? value : "chat";
    } catch {
      return "chat";
    }
  }

  let chosen: Tab = $state(stored());
  // The remembered tab may not exist for this session: a Terminal tab only for interactive Claude.
  const tab = $derived(tabs.some((t) => t.id === chosen) ? chosen : "chat");
  $effect(() => {
    try {
      localStorage.setItem(TAB_KEY, chosen);
    } catch {
      // Storage off: the tab is simply not remembered.
    }
  });
</script>

<div class="head">
  <h2>{name}</h2>
  <div class="tabs" role="tablist">
    {#each tabs as t (t.id)}
      <button
        type="button"
        role="tab"
        aria-selected={tab === t.id}
        class:active={tab === t.id}
        onclick={() => (chosen = t.id)}>{t.label}</button
      >
    {/each}
  </div>
</div>
<!-- Chat stays mounted while hidden: a prompt it is streaming keeps going. Status mounts only when shown, and
     streams only while mounted. -->
<div class="panel" hidden={tab !== "chat"}>
  <Chat {name} {onchange} {ondeleted} />
</div>
{#if tab === "terminal"}
  <div class="panel">
    <Terminal {name} />
  </div>
{:else if tab === "status"}
  <div class="panel">
    <Status {name} />
  </div>
{/if}

<style>
  .head {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }
  h2 {
    margin: 0;
  }
  .tabs {
    display: flex;
    gap: 0.25rem;
  }
  .tabs button {
    background: none;
    border: 1px solid transparent;
    border-bottom-color: var(--line);
    border-radius: 0.3rem 0.3rem 0 0;
    padding: 0.3rem 0.8rem;
    color: var(--muted);
    cursor: pointer;
  }
  .tabs button.active {
    border-color: var(--line);
    border-bottom-color: transparent;
    color: inherit;
  }
  .panel {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
  }
  .panel[hidden] {
    display: none;
  }
</style>
