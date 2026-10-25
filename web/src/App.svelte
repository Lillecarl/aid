<script lang="ts">
  import { onMount } from "svelte";
  import * as api from "./lib/api";
  import type { SessionInfo } from "./lib/api";
  import CreateSession from "./lib/CreateSession.svelte";
  import { linkedOnly } from "./lib/links";
  import Session from "./lib/Session.svelte";
  import SessionList from "./lib/SessionList.svelte";
  import { parseFilter, tagged } from "./lib/tags";

  let email = $state("");
  let list: SessionInfo[] = $state([]);
  let current: string | null = $state(null);
  let error = $state("");
  // Supervisors and their classifiers live as tabs under the sessions they watch; the list can still show them.
  let showLinked = $state(false);
  const hidden = $derived(linkedOnly(list));
  let filter = $state("");
  const terms = $derived(parseFilter(filter));
  const anyTags = $derived(list.some((s) => Object.keys(s.tags).length > 0));
  const listed = $derived(list.filter((s) => (showLinked || !hidden.has(s.name)) && tagged(s, terms)));

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

  onMount(() => {
    void api.me().then((me) => (email = me.email));
    // Sessions start, stop and appear from other pages and the CLI too, so the list follows the daemon.
    return api.followSessions(
      (sessions) => (list = sessions),
      (problem) => (error = problem),
    );
  });
</script>

<header>
  <strong>aid</strong>
  <span class="who" data-testid="user-email">{email}</span>
  <button type="button" onclick={logout}>Log out</button>
</header>
<main>
  <aside>
    <h2>Sessions</h2>
    {#if anyTags || filter}
      <input
        class="filter"
        type="search"
        placeholder="key or key=value …"
        aria-label="Filter sessions by tag"
        data-testid="tag-filter"
        bind:value={filter}
      />
    {/if}
    <SessionList
      sessions={listed}
      {current}
      onselect={(name) => (current = name)}
      empty={terms.length > 0 ? "No session has these tags." : undefined}
    />
    {#if hidden.size > 0}
      <button type="button" class="linked" data-testid="show-linked" onclick={() => (showLinked = !showLinked)}
        >{showLinked ? "Hide" : "Show"} {hidden.size} linked session{hidden.size === 1 ? "" : "s"}</button
      >
    {/if}
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
        <Session
          name={current}
          sessions={list}
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
  .filter {
    width: 100%;
    box-sizing: border-box;
    margin-bottom: 0.5rem;
  }
  .linked {
    width: auto;
    margin: -1rem 0 1.5rem;
    font-size: 0.85em;
    color: var(--muted);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-decoration: underline;
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
