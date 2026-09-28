<script lang="ts">
  import Chat from "./Chat.svelte";
  import Status from "./Status.svelte";

  interface Props {
    name: string;
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Tab = "chat" | "status";
  const TABS: { id: Tab; label: string }[] = [
    { id: "chat", label: "Chat" },
    { id: "status", label: "Status" },
  ];
  const TAB_KEY = "aid.sessionTab";

  let { name, onchange, ondeleted }: Props = $props();

  function stored(): Tab {
    try {
      const value = localStorage.getItem(TAB_KEY);
      return TABS.some((t) => t.id === value) ? (value as Tab) : "chat";
    } catch {
      return "chat";
    }
  }

  let tab: Tab = $state(stored());
  $effect(() => {
    try {
      localStorage.setItem(TAB_KEY, tab);
    } catch {
      // Storage off: the tab is simply not remembered.
    }
  });
</script>

<div class="head">
  <h2>{name}</h2>
  <div class="tabs" role="tablist">
    {#each TABS as t (t.id)}
      <button
        type="button"
        role="tab"
        aria-selected={tab === t.id}
        class:active={tab === t.id}
        onclick={() => (tab = t.id)}>{t.label}</button
      >
    {/each}
  </div>
</div>
<!-- Chat stays mounted while hidden: a prompt it is streaming keeps going. Status mounts only when shown, and
     streams only while mounted. -->
<div class="panel" hidden={tab !== "chat"}>
  <Chat {name} {onchange} {ondeleted} />
</div>
{#if tab === "status"}
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
