<script lang="ts">
  import { onMount } from "svelte";
  import * as api from "./lib/api";
  import type { SessionInfo } from "./lib/api";
  import Chat from "./lib/Chat.svelte";
  import CreateSession from "./lib/CreateSession.svelte";
  import SessionList from "./lib/SessionList.svelte";

  let email = $state("");
  let list: SessionInfo[] = $state([]);
  let current: string | null = $state(null);
  let error = $state("");

  async function refresh(): Promise<void> {
    try {
      list = await api.sessions();
      error = "";
    } catch (e) {
      error = String(e);
    }
  }

  async function logout(): Promise<void> {
    await api.logout();
    location.href = "/";
  }

  onMount(async () => {
    email = (await api.me()).email;
    await refresh();
  });
</script>

<header>
  <strong>aid</strong>
  <span class="who">{email}</span>
  <button type="button" onclick={logout}>Log out</button>
</header>
<main>
  <aside>
    <h2>Sessions</h2>
    <SessionList sessions={list} {current} onselect={(name) => (current = name)} />
    {#if error}<p class="error">{error}</p>{/if}
    <CreateSession
      oncreated={async (name) => {
        await refresh();
        current = name;
      }}
    />
  </aside>
  <section>
    {#if current === null}
      <p class="empty">Select or create a session.</p>
    {:else}
      {#key current}
        <Chat
          name={current}
          onchange={refresh}
          ondeleted={async () => {
            current = null;
            await refresh();
          }}
        />
      {/key}
    {/if}
  </section>
</main>

<style>
  header {
    display: flex;
    gap: 1rem;
    align-items: center;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--line);
  }
  .who {
    margin-left: auto;
    color: var(--muted);
  }
  main {
    display: grid;
    grid-template-columns: minmax(16rem, 22rem) 1fr;
    height: calc(100vh - 3rem);
  }
  aside {
    border-right: 1px solid var(--line);
    padding: 1rem;
    overflow: auto;
  }
  section {
    display: flex;
    flex-direction: column;
    padding: 1rem;
    min-width: 0;
    min-height: 0;
  }
  .empty {
    color: var(--muted);
  }
  @media (max-width: 700px) {
    main {
      grid-template-columns: 1fr;
      height: auto;
    }
    aside {
      border-right: 0;
    }
  }
</style>
