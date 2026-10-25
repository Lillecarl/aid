<script lang="ts">
  import * as api from "./api";
  import type { AgentCatalog, AgentKind, AgentSpec, ModelEntry, PermissionMode, ProviderModels } from "./api";

  // A new session asks; its Permissions tab changes that, per tool for native sessions.
  const permission: PermissionMode = "ask";

  interface Props {
    oncreated: (name: string) => void | Promise<void>;
  }

  let { oncreated }: Props = $props();

  let name = $state("");
  let kind: AgentKind = $state("claude-tty");
  let dir = $state("");
  let command = $state("claude-agent-acp");
  let agent = $state("");
  let catalog = $state<AgentCatalog | null>(null);
  // Empty: the agent's own model.
  let provider = $state("");
  let providers = $state<ProviderModels[]>([]);
  let modelId = $state("");
  let models = $state<ModelEntry[]>([]);
  let args = $state("");
  let model = $state("");
  // One regex per line; empty offers every tool (Lillecarl/aid#50).
  let tools = $state("");
  let trust = $state(false);
  let error = $state("");
  let busy = $state(false);

  const words = (text: string): string[] => text.split(/\s+/).filter((word) => word !== "");

  function spec(): AgentSpec {
    // Completions end in "/"; Claude Code names its project directory after the cwd as given.
    const cwd = dir.length > 1 ? dir.replace(/\/+$/, "") : dir;
    switch (kind) {
      case "acp": {
        const [program, ...rest] = words(command);
        if (program === undefined) throw new Error("the command is empty");
        return { kind, cwd, command: [program, ...rest], permission };
      }
      case "native": {
        const patterns = tools.split("\n").map((line) => line.trim()).filter((line) => line !== "");
        return {
          kind,
          cwd,
          agent,
          permission,
          ...(provider && modelId ? { model: `${provider}/${modelId}` } : {}),
          ...(patterns.length ? { tools: patterns } : {}),
        };
      }
      case "claude-tty":
        return { kind, cwd, args: words(args), trust_cwd: trust };
      case "claude-sdk":
        return { kind, cwd, model, permission };
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

  // Providers with a key first: those are the ones a session can run on.
  async function loadProviders(): Promise<void> {
    try {
      const listing = await api.models();
      providers = listing.providers.toSorted((a, b) => Number(b.has_key) - Number(a.has_key));
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  // A reply that comes back after another provider was chosen is dropped.
  async function loadModels(chosen: string): Promise<void> {
    models = [];
    modelId = "";
    if (!chosen) return;
    try {
      const [row] = (await api.models(chosen)).providers;
      if (provider !== chosen) return;
      models = row?.models ?? [];
      modelId = models.find((m) => m.unsupported === null)?.id ?? "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  // Asked each time the kind turns to native: the catalog reflects the plugins path as it is now.
  $effect(() => {
    if (kind === "native") {
      void loadAgents();
      void loadProviders();
    }
  });

  const tokens = (count: number | null): string =>
    count === null ? "" : count >= 1000 ? `${Math.round(count / 1000)}K` : String(count);
  const modelLabel = (m: ModelEntry): string =>
    [m.id, m.context === null ? "" : `${tokens(m.context)} context`, m.unsupported === null ? "" : "unsupported"]
      .filter((part) => part !== "")
      .join(" · ");

  const chosen = $derived(catalog?.agents.find((a) => a.name === agent));

  // The working directory starts at aid web's prefix (the home directory by default), and the datalist offers
  // the directories what is typed can go on to. A reply that comes back after newer typing is dropped.
  let completions = $state<string[]>([]);
  let asked = 0;
  async function complete(typed: string): Promise<void> {
    const ask = ++asked;
    try {
      const found = await api.dirs(typed);
      if (ask === asked) completions = found.dirs;
    } catch {
      if (ask === asked) completions = [];
    }
  }
  void api.dirs("").then(
    (found) => {
      if (dir === "") dir = found.prefix;
      void complete(dir);
    },
    () => undefined,
  );

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
      <option value="claude-sdk">Claude Code (streaming)</option>
      <option value="acp">ACP agent</option>
      <option value="native">Native agent</option>
    </select>
  </label>
  <label>
    Working directory
    <input
      bind:value={dir}
      oninput={() => complete(dir)}
      list="cwd-completions"
      required
      placeholder="/home/me/project"
      autocomplete="off"
      data-testid="cwd-input"
    />
  </label>
  <datalist id="cwd-completions">
    {#each completions as dir (dir)}
      <option value={dir}></option>
    {/each}
  </datalist>
  {#if kind === "acp"}
    <label>Command <input bind:value={command} required /></label>
  {:else if kind === "native"}
    <label>
      Agent
      <select bind:value={agent} required disabled={catalog === null} data-testid="agent-select">
        {#each catalog?.agents ?? [] as info (info.name)}
          <option value={info.name}>{info.name}</option>
        {/each}
      </select>
    </label>
    {#if chosen?.description}<p class="hint">{chosen.description}</p>{/if}
    {#if catalog !== null && catalog.agents.length === 0}
      <p class="hint">No agents on the daemon's plugins path.</p>
    {/if}
    {#each catalog?.problems ?? [] as problem (problem)}
      <p class="error">{problem}</p>
    {/each}
    <label>
      Model provider
      <select bind:value={provider} onchange={() => loadModels(provider)} data-testid="model-provider">
        <option value="">the agent's own model</option>
        {#each providers as row (row.name)}
          <option value={row.name}>{row.name}{row.has_key || row.key_env === null ? "" : " (no key)"}</option>
        {/each}
      </select>
    </label>
    {#if provider}
      <label>
        Model
        <select bind:value={modelId} required data-testid="model-id">
          {#each models as m (m.id)}
            <option value={m.id} disabled={m.unsupported !== null} title={m.unsupported ?? m.name ?? ""}
              >{modelLabel(m)}</option
            >
          {/each}
        </select>
      </label>
    {/if}
    <label>
      Tools
      <textarea bind:value={tools} rows="2" placeholder="every tool; or one regex per line, e.g. ls|read|outline"
        data-testid="tool-ceiling"></textarea>
    </label>
  {:else if kind === "claude-sdk"}
    <label>Model <input bind:value={model} required placeholder="opus" /></label>
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
