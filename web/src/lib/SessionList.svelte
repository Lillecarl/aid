<script lang="ts">
  import type { SessionInfo } from "./api";

  interface Props {
    sessions: SessionInfo[];
    current: string | null;
    onselect: (name: string) => void;
  }

  let { sessions, current, onselect }: Props = $props();
</script>

<ul>
  {#each sessions as session (session.name)}
    <li>
      <button type="button" class:current={session.name === current} onclick={() => onselect(session.name)}>
        <span class="name">{session.name}</span>
        <span class="meta">{session.kind} · {session.running ? "running" : "stopped"}</span>
      </button>
    </li>
  {:else}
    <li class="none">No sessions yet.</li>
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
  button:hover,
  button.current {
    background: var(--line);
  }
  .meta,
  .none {
    color: var(--muted);
    font-size: 0.9em;
  }
</style>
