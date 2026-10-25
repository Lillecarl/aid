<script lang="ts">
  import { SvelteSet } from "svelte/reactivity";
  import type { SessionInfo } from "./api";
  import Chat from "./Chat.svelte";
  import Files from "./Files.svelte";
  import LinkedTab from "./LinkedTab.svelte";
  import { family, tabLabel } from "./links";
  import Permissions from "./Permissions.svelte";
  import Status from "./Status.svelte";
  import Terminal from "./Terminal.svelte";

  interface Props {
    name: string;
    sessions: SessionInfo[];
    onchange: () => void | Promise<void>;
    ondeleted: () => void | Promise<void>;
  }

  type Tab = "chat" | "files" | "terminal" | "permissions" | "status";
  const TAB_KEY = "aid.sessionTab";

  let { name, sessions, onchange, ondeleted }: Props = $props();

  // The linked tab picked under Chat; every tab shows that session. A tab whose session went away falls back.
  const linked = $derived(family(sessions, name));
  let picked: string | null = $state(null);
  const shown = $derived(picked !== null && linked.includes(picked) ? picked : name);
  // A linked session's chat mounts when first picked and then stays, like Files: a prompt it streams goes on.
  const chats = new SvelteSet<string>();
  $effect(() => {
    chats.add(shown);
  });
  const kind = $derived(sessions.find((s) => s.name === shown)?.kind);

  const tabs = $derived<{ id: Tab; label: string }[]>([
    { id: "chat", label: "Chat" },
    { id: "files", label: "Files" },
    ...(kind === "claude-tty" ? [{ id: "terminal" as const, label: "Terminal" }] : []),
    { id: "permissions" as const, label: "Permissions" },
    { id: "status", label: "Status" },
  ]);

  function stored(): Tab {
    try {
      const value = localStorage.getItem(TAB_KEY);
      return value === "files" || value === "terminal" || value === "permissions" || value === "status"
        ? value
        : "chat";
    } catch {
      return "chat";
    }
  }

  let chosen: Tab = $state(stored());
  // The remembered tab may not exist for this session: a Terminal tab only for interactive Claude.
  const tab = $derived(tabs.some((t) => t.id === chosen) ? chosen : "chat");
  // Files mounts when first shown and then stays, keeping the open folders and file across tab switches.
  let filesShown = $state(false);
  $effect(() => {
    if (tab === "files") filesShown = true;
  });
  $effect(() => {
    try {
      localStorage.setItem(TAB_KEY, chosen);
    } catch {
      // Storage off: the tab is simply not remembered.
    }
  });
</script>

{#snippet linkedTabs()}
  {#each linked as other (other)}
    <LinkedTab
      name={other}
      label={tabLabel(sessions, other)}
      active={other === shown}
      onselect={() => (picked = other)}
    />
  {/each}
{/snippet}

<div class="head">
  <h2>{name}{#if shown !== name}<span class="viewed">{` › ${tabLabel(sessions, shown)}`}</span>{/if}</h2>
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
<!-- Chat stays mounted while hidden: a prompt it is streaming keeps going. Status and Permissions mount only when
     shown, and stream only while mounted. -->
{#each [...chats].filter((c) => linked.includes(c)) as chat (chat)}
  <div class="panel" hidden={tab !== "chat" || chat !== shown}>
    <Chat
      name={chat}
      {onchange}
      ondeleted={chat === name
        ? ondeleted
        : async () => {
            chats.delete(chat);
            picked = null;
            await onchange();
          }}
      linked={linked.length > 1 ? linkedTabs : undefined}
    />
  </div>
{/each}
{#key shown}
  {#if filesShown}
    <div class="panel" hidden={tab !== "files"}>
      <Files name={shown} />
    </div>
  {/if}
  {#if tab === "terminal"}
    <div class="panel">
      <Terminal name={shown} />
    </div>
  {:else if tab === "permissions"}
    <div class="panel scroll">
      <Permissions name={shown} />
    </div>
  {:else if tab === "status"}
    <div class="panel">
      <Status name={shown} />
    </div>
  {/if}
{/key}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.25rem 1rem;
    margin-bottom: 0.5rem;
  }
  h2 {
    margin: 0;
  }
  .viewed {
    color: var(--muted);
    font-weight: normal;
  }
  .tabs {
    display: flex;
    flex-wrap: wrap;
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
  .scroll {
    overflow-y: auto;
  }
  .panel[hidden] {
    display: none;
  }
</style>
