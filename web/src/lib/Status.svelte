<script lang="ts">
  import { onMount } from "svelte";
  import { statusEventsUrl, watch } from "./api";
  import type { SessionStatus } from "./api";

  let { name }: { name: string } = $props();

  let status: SessionStatus | null = $state(null);
  let problem = $state("");

  // Mounted only while its tab is shown.
  onMount(() =>
    watch<SessionStatus>(
      statusEventsUrl(name),
      (data) => (status = data),
      (text) => (problem = text),
    ),
  );
</script>

{#if problem}<p class="problem">{problem}</p>{/if}
{#if status === null}
  <p class="muted">Loading…</p>
{:else}
  <dl>
    <dt>State</dt>
    <dd>
      {status.running ? (status.busy ? "running a turn" : "idle") : "stopped"}
      {#if status.pid !== null}<span class="muted">(pid {status.pid})</span>{/if}
    </dd>
    <dt>Waiting messages</dt>
    <dd>{status.pending}</dd>
    <dt>Kind</dt>
    <dd>{status.kind}</dd>
    <dt>Runs</dt>
    <dd><code>{status.runs}</code></dd>
    <dt>Directory</dt>
    <dd><code>{status.cwd}</code></dd>
    <dt>MCP servers</dt>
    <dd>{status.mcp_servers.length ? status.mcp_servers.join(", ") : "none"}</dd>
    <dt>aid tools</dt>
    <dd>{status.aid_tools ? "on" : "off"}</dd>
  </dl>
{/if}

<style>
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.4rem 1rem;
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .muted {
    color: var(--muted);
  }
  .problem {
    color: var(--bad);
  }
</style>
