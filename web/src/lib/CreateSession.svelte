<script lang="ts">
  import * as api from "./api";
  import type { AgentCatalog, AgentKind, AgentSpec, PermissionMode } from "./api";

  interface Props {
    oncreated: (name: string) => void | Promise<void>;
  }

  let { oncreated }: Props = $props();

  let name = $state("");
  let kind: AgentKind = $state("claude-tty");
  let cwd = $state("");
  let command = $state("claude-agent-acp");
  let permission: PermissionMode = $state("ask");
  let agent = $state("");
  let catalog = $state<AgentCatalog | null>(null);
  let args = $state("");
  let trust = $state(false);
  let error = $state("");
  let busy = $state(false);

  const words = (text: string): string[] => text.split(/\s+/).filter((word) => word !== "");

  function spec(): AgentSpec {
    switch (kind) {
      case "acp": {
        const [program, ...rest] = words(command);
        if (program === undefined) throw new Error("the command is empty");
        return { kind, cwd, command: [program, ...rest], permission };
      }
      case "pydantic-ai":
        return { kind, cwd, agent, permission };
      case "claude-tty":
        return { kind, cwd, args: words(args), trust_cwd: trust };
    }
  }

  async function loadAgents(): Promise<void> {
    try {
      catalog = await api.agents();
      if (!catalog.agents.some((a) => a.name === agent)) agent = catalog.agents[0]?.name ?? "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  // Asked each time the kind turns to pydantic-ai: the catalog reflects the agents path as it is now.
  $effect(() => {
    if (kind === "pydantic-ai") void loadAgents();
  });

  const chosen = $derived(catalog?.agents.find((a) => a.name === agent));

  async function submit(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    busy = true;
    error = "";
    try {
      await api.create(name, spec());
      await oncreated(name);
      name = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<form onsubmit={submit}>
  <h2>New session</h2>
  <label>Name <input bind:value={name} required pattern={"[A-Za-z0-9_.\\-]{1,64}"} /></label>
  <label>
    Kind
    <select bind:value={kind}>
      <option value="claude-tty">Claude Code (interactive)</option>
      <option value="acp">ACP agent</option>
      <option value="pydantic-ai">pydantic-ai agent</option>
    </select>
  </label>
  <label>Working directory <input bind:value={cwd} required placeholder="/home/me/project" /></label>
  {#if kind !== "claude-tty"}
    <!-- ACP: the agent's own requests. pydantic-ai: aid.coding's commands and applied edits. -->
    <label>
      Tool permissions
      <select bind:value={permission}>
        <option value="ask">Ask me</option>
        <option value="allow">Allow all</option>
        <option value="deny">Deny all</option>
      </select>
    </label>
  {/if}
  {#if kind === "acp"}
    <label>Command <input bind:value={command} required /></label>
  {:else if kind === "pydantic-ai"}
    <label>
      Agent
      <select bind:value={agent} required disabled={catalog === null}>
        {#each catalog?.agents ?? [] as info (info.name)}
          <option value={info.name}>{info.name}</option>
        {/each}
      </select>
    </label>
    {#if chosen?.description}<p class="hint">{chosen.description}</p>{/if}
    {#if catalog !== null && catalog.agents.length === 0}
      <p class="hint">No agents on the daemon's agents path.</p>
    {/if}
    {#each catalog?.problems ?? [] as problem (problem)}
      <p class="error">{problem}</p>
    {/each}
  {:else}
    <label>Claude arguments <input bind:value={args} placeholder="--model opus" /></label>
    <label class="check"><input type="checkbox" bind:checked={trust} /> Trust the directory</label>
  {/if}
  <button type="submit" disabled={busy}>{busy ? "Starting…" : "Create"}</button>
  {#if error}<p class="error">{error}</p>{/if}
</form>

<style>
  label {
    display: block;
    margin: 0.4rem 0;
  }
  .hint {
    color: var(--muted);
    font-size: 0.9em;
    margin: 0 0 0.4rem;
  }
  .error {
    white-space: pre-wrap;
    font-size: 0.9em;
  }
  label.check {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  label.check input {
    width: auto;
  }
</style>
