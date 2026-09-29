<script lang="ts">
  import { onMount } from "svelte";
  import * as api from "./api";
  import type { SessionStatus, SessionSummary } from "./api";
  import { tokens } from "./usage";

  let { name }: { name: string } = $props();

  let status: SessionStatus | null = $state(null);
  let summary: SessionSummary | null = $state(null);
  let problem = $state("");

  async function total(): Promise<void> {
    try {
      summary = await api.summary(name);
    } catch (e) {
      problem = e instanceof Error ? e.message : String(e);
    }
  }

  // Mounted only while its tab is shown. The totals change when a turn ends or a worker starts, which the
  // status stream reports.
  onMount(() =>
    api.watch<SessionStatus>(
      api.statusEventsUrl(name),
      (data) => {
        const changed = !status || status.busy !== data.busy || status.running !== data.running;
        status = data;
        if (changed && !data.busy) void total();
      },
      (text) => (problem = text),
    ),
  );

  const entries = (record: Record<string, number>): [string, number][] =>
    Object.entries(record).sort(([, a], [, b]) => b - a);
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
    <dt>Agent</dt>
    <dd>{status.agent ?? "not reported"}</dd>
    <dt>Model</dt>
    <dd>{status.model ?? "not reported"}</dd>
    <dt>Agent session</dt>
    <dd>{#if status.agent_session}<code>{status.agent_session}</code>{:else}none{/if}</dd>
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

{#if summary}
  <h3>All turns</h3>
  <p class="muted">What the agents reported; how much a turn's count covers differs by agent.</p>
  <dl>
    <dt>Turns</dt>
    <dd>{summary.turns} <span class="muted">({summary.starts} worker starts)</span></dd>
    <dt>Tokens</dt>
    <dd>
      {tokens(summary.input_tokens)} in · {tokens(summary.output_tokens)} out · {tokens(summary.cache_read_tokens)}
      cached · {tokens(summary.cache_write_tokens)} cache written
      {#if summary.requests}<span class="muted">({summary.requests} requests)</span>{/if}
    </dd>
    <dt>Models</dt>
    <dd>
      {#each entries(summary.models) as [model, turns], i (model)}{i ? ", " : ""}{model}
        <span class="muted">({turns} turns)</span>{:else}none reported{/each}
    </dd>
    <dt>Cost</dt>
    <dd>
      {#each entries(summary.cost) as [currency, amount], i (currency)}{i ? ", " : ""}{amount.toFixed(2)}
        {currency}{:else}none reported{/each}
    </dd>
    <dt>Agent sessions</dt>
    <dd>
      {#each summary.agent_sessions as id (id)}<code>{id}</code><br />{:else}none{/each}
    </dd>
  </dl>

  <h3>Files</h3>
  {#if summary.files.length}
    <table>
      <thead><tr><th>Path</th><th>Reads</th><th>Writes</th></tr></thead>
      <tbody>
        {#each summary.files as file (file.path)}
          <tr><td><code>{file.path}</code></td><td>{file.reads || ""}</td><td>{file.writes || ""}</td></tr>
        {/each}
      </tbody>
    </table>
  {:else}
    <p class="muted">No tool call has named a file.</p>
  {/if}
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
  h3 {
    font-size: 0.95rem;
    margin: 1.2rem 0 0.3rem;
  }
  table {
    border-collapse: collapse;
    font-size: 0.9em;
  }
  th,
  td {
    text-align: left;
    padding: 0.15rem 0.8rem 0.15rem 0;
    border-bottom: 1px solid var(--line);
  }
  td:not(:first-child),
  th:not(:first-child) {
    text-align: right;
  }
  td code {
    overflow-wrap: anywhere;
  }
  .muted {
    color: var(--muted);
  }
  .problem {
    color: var(--bad);
  }
</style>
