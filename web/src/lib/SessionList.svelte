<script lang="ts">
  import type { SessionInfo } from "./api";
  import { tagLabels } from "./tags";

  interface Props {
    sessions: SessionInfo[];
    current: string | null;
    onselect: (name: string) => void;
    /** What an empty list says. */
    empty?: string;
  }

  let { sessions, current, onselect, empty = "No sessions yet." }: Props = $props();
</script>

<ul data-testid="session-list">
  {#each sessions as session (session.name)}
    <li>
      <button type="button" class:current={session.name === current} onclick={() => onselect(session.name)}>
        <span class="name">{session.name}</span>
        <span class="meta"
          >{session.kind} · {session.working ? "working" : session.running ? "running" : "stopped"}</span
        >
        {#if Object.keys(session.tags).length > 0}
          <span class="tags">
            {#each tagLabels(session.tags) as tag (tag)}<span class="tag" data-testid="tag">{tag}</span>{/each}
          </span>
        {/if}
        {#if session.attention}<span class="asks">{session.attention}</span>{/if}
        {#if session.permissions > 0}
          <span class="asks">waits on {session.permissions === 1 ? "an approval" : `${session.permissions} approvals`}</span>
        {/if}
      </button>
    </li>
  {:else}
    <li class="none">{empty}</li>
  {/each}
</ul>

<style>
  ul {
    list-style: none;
    padding: 0;
    margin: 0 0 1.5rem;
  }
  button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    border: 0;
    background: none;
    text-align: left;
  }
  button:hover {
    background: var(--code-bg);
  }
  button.current {
    background: color-mix(in srgb, var(--accent) 20%, transparent);
  }
  .asks {
    color: var(--accent);
    font-size: 0.9em;
    font-weight: 600;
  }
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    min-width: 0;
  }
  .tag {
    font-size: 0.8em;
    padding: 0 0.35rem;
    border: 1px solid var(--line);
    border-radius: 0.6rem;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .meta,
  .none {
    color: var(--muted);
    font-size: 0.9em;
  }
</style>
