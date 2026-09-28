<script lang="ts">
  import * as api from "./api";
  import type { AgentKind, AgentSpec } from "./api";

  interface Props {
    oncreated: (name: string) => void | Promise<void>;
  }

  let { oncreated }: Props = $props();

  let name = $state("");
  let kind: AgentKind = $state("claude-tty");
  let cwd = $state("");
  let command = $state("claude-agent-acp");
  let target = $state("");
  let args = $state("");
  let trust = $state(false);
  let error = $state("");
  let busy = $state(false);

  const words = (text: string): string[] => text.split(/\s+/).filter((word) => word !== "");

  function spec(): AgentSpec {
    switch (kind) {
      case "acp":
        return { kind, cwd, command: words(command) };
      case "pydantic-ai":
        return { kind, cwd, target };
      case "claude-tty":
        return { kind, cwd, args: words(args), trust_cwd: trust };
    }
  }

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
  {#if kind === "acp"}
    <label>Command <input bind:value={command} required /></label>
  {:else if kind === "pydantic-ai"}
    <label>Target <input bind:value={target} required placeholder="package.module:agent" /></label>
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
  label.check {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  label.check input {
    width: auto;
  }
</style>
