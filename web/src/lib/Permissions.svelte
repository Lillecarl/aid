<script lang="ts">
  import { onMount } from "svelte";
  import * as api from "./api";
  import type { PermissionMode, SessionStatus, ToolMode } from "./api";

  let { name }: { name: string } = $props();

  const MODES: PermissionMode[] = ["allow", "ask", "supervisor", "ask_supervisor", "deny"];

  let status: SessionStatus | null = $state(null);
  let problem = $state("");
  let busy = $state(false);
  let checked = $state(new Set<string>());

  let others: string[] = $state([]);

  onMount(() => {
    const stops = [
      api.followStatus(
        name,
        (data) => (status = data),
        (text) => (problem = text),
      ),
      api.followSessions(
        (list) => (others = list.map((s) => s.name).filter((n) => n !== name)),
        (text) => (problem = text),
      ),
    ];
    return () => stops.forEach((stop) => stop());
  });

  const setSupervisor = (supervisor: string) => change(() => api.setSupervisor(name, supervisor || null));

  const groups = $derived.by(() => {
    const byGroup = new Map<string, ToolMode[]>();
    for (const tool of status?.tools ?? []) byGroup.set(tool.group, [...(byGroup.get(tool.group) ?? []), tool]);
    return [...byGroup.entries()].sort(([a], [b]) => a.localeCompare(b));
  });

  function toggle(names: string[], on: boolean): void {
    const next = new Set(checked);
    for (const n of names) {
      if (on) next.add(n);
      else next.delete(n);
    }
    checked = next;
  }

  async function change(work: () => Promise<void>): Promise<void> {
    busy = true;
    problem = "";
    try {
      await work();
    } catch (e) {
      problem = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  // The status stream brings the new modes back; nothing is updated here ahead of the daemon.
  const setChecked = (mode: PermissionMode | null) =>
    change(() => api.setPermissions(name, Object.fromEntries([...checked].map((n) => [n, mode]))));
  const setDefault = (mode: PermissionMode) => change(() => api.setPermissions(name, {}, mode));

  // The sandbox form edits a draft; the status stream resets it whenever the saved sandbox changes.
  let boxed = $state(false);
  let network = $state(true);
  let writable = $state("");
  let shown = "";
  $effect(() => {
    const saved = status?.sandbox ?? null;
    const key = JSON.stringify(saved);
    if (key === shown) return;
    shown = key;
    boxed = saved !== null;
    network = saved?.network ?? true;
    writable = (saved?.writable ?? []).join("\n");
  });
  const saveSandbox = () =>
    change(() =>
      api.setSandbox(
        name,
        boxed
          ? {
              network,
              writable: writable
                .split("\n")
                .map((line) => line.trim())
                .filter((line) => line !== ""),
            }
          : null,
      ),
    );
</script>

{#if problem}<p class="problem">{problem}</p>{/if}
{#if status === null}
  <p class="muted">Loading…</p>
{:else}
  {#if status.permission === null}
    <p class="muted">
      Claude reads these rules when it starts: a change restarts it, when its turn ends if one runs. An ask
      tool asks in the terminal pane and here.
    </p>
  {:else}
    <div class="bar">
      <label>
        {status.kind === "native"
          ? "Tools that ask for themselves"
          : status.kind === "claude-sdk"
            ? "What Claude asks about the rest"
            : "Requests about the rest"}
        <select
          value={status.permission}
          disabled={busy}
          data-testid="permission-default"
          onchange={(e) => setDefault(e.currentTarget.value as PermissionMode)}
        >
          {#each MODES as mode (mode)}<option value={mode}>{mode}</option>{/each}
        </select>
      </label>
    </div>
  {/if}
  <label class="supervisor">
    Supervisor
    <select
      value={status.supervisor ?? ""}
      disabled={busy}
      data-testid="supervisor"
      onchange={(e) => setSupervisor(e.currentTarget.value)}
    >
      <option value="">none</option>
      {#each others as other (other)}<option value={other}>{other}</option>{/each}
    </select>
    <span class="muted">reviews each turn end, and answers tools in supervisor and ask_supervisor mode</span>
  </label>
  {#if status.kind === "native"}
    <fieldset data-testid="sandbox">
      <legend>
        <label><input type="checkbox" bind:checked={boxed} data-testid="sandbox-on" /> Sandbox (bwrap)</label>
      </legend>
      <p class="muted">
        The shell, background tasks and scripts see a read-only system; they write only the working directory, a
        private /tmp and the paths below. Saving restarts the worker, when its turn ends if one runs.
      </p>
      <label><input type="checkbox" bind:checked={network} disabled={!boxed} /> Network</label>
      <label class="column">
        Also writable, one absolute path per line
        <textarea bind:value={writable} disabled={!boxed} rows="3"></textarea>
      </label>
      <button type="button" disabled={busy} onclick={saveSandbox} data-testid="sandbox-save">Save sandbox</button>
    </fieldset>
  {/if}
  {#if groups.length === 0}
    <p class="muted">
      {status.kind === "native"
        ? "Start the session to list its tools."
        : status.kind === "claude-sdk"
          ? "Send the session a prompt to list its tools."
          : status.kind === "claude-tty"
            ? "Its tools appear here as Claude uses them or asks about them."
            : "Its tools appear here as the agent asks permission for them."}
    </p>
  {:else}
    <div class="bar" data-testid="permission-actions">
      {#each MODES as mode (mode)}
        <button type="button" disabled={busy || checked.size === 0} onclick={() => setChecked(mode)}
          >{mode}</button
        >
      {/each}
      <button type="button" disabled={busy || checked.size === 0} onclick={() => setChecked(null)}
        >default</button
      >
      <button type="button" onclick={() => toggle(status?.tools.map((t) => t.name) ?? [], true)}
        >check all</button
      >
      <button type="button" onclick={() => (checked = new Set())}>uncheck all</button>
      <span class="muted">{checked.size} checked</span>
    </div>
    {#each groups as [group, tools] (group)}
      {@const all = tools.every((t) => checked.has(t.name))}
      {@const some = tools.some((t) => checked.has(t.name))}
      <fieldset data-testid="permission-group">
        <legend>
          <label>
            <input
              type="checkbox"
              checked={all}
              indeterminate={some && !all}
              onchange={(e) => toggle(tools.map((t) => t.name), e.currentTarget.checked)}
            />
            {group}
          </label>
        </legend>
        <ul>
          {#each tools as tool (tool.name)}
            <li data-testid="permission-tool">
              <label>
                <span class="name">{tool.name}</span>
                <span class="mode {tool.mode ?? ''}" class:default={!tool.set}
                  >{tool.mode === null ? "agent decides" : `${tool.mode}${tool.set ? "" : " (default)"}`}</span
                >
                <input
                  type="checkbox"
                  checked={checked.has(tool.name)}
                  onchange={(e) => toggle([tool.name], e.currentTarget.checked)}
                />
              </label>
            </li>
          {/each}
        </ul>
      </fieldset>
    {/each}
  {/if}
{/if}

<style>
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }
  fieldset {
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    margin: 0 0 0.75rem;
    padding: 0.25rem 0.75rem 0.5rem;
  }
  .column {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin: 0.5rem 0;
  }
  textarea {
    font-family: monospace;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  li label {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 1rem;
    align-items: center;
    padding: 0.15rem 0;
  }
  .name {
    font-family: monospace;
    overflow-wrap: anywhere;
  }
  .mode.allow {
    color: var(--ok);
  }
  .mode.deny {
    color: var(--bad);
  }
  .mode.ask,
  .mode.supervisor,
  .mode.ask_supervisor {
    color: var(--accent);
  }
  .mode.default {
    opacity: 0.7;
  }
  .supervisor {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
  }
  .muted {
    color: var(--muted);
  }
  .problem {
    color: var(--bad);
  }
</style>
