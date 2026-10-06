<script lang="ts">
  import * as api from "./api";
  import type { PermissionDecision, PermissionRequest } from "./api";
  import { command, hasInput } from "./tools";

  interface Props {
    session: string;
    request: PermissionRequest;
    decision: PermissionDecision | null;
  }

  const BY: Record<PermissionDecision["by"], string> = {
    person: "",
    policy: " by the session's policy",
    timeout: ": nobody answered in time",
    auto: ": nobody answered in time, picked automatically",
    cancel: ": the turn ended",
    terminal: "",
    plugin: " by a plugin",
  };

  let { session, request, decision }: Props = $props();
  let sending = $state(false);
  let error = $state("");
  let words = $state("");

  const recommended = $derived(request.options.find((o) => o.recommended)?.name);

  const shell = $derived(command(request.input));
  const chosen = $derived(
    decision === null
      ? null
      : decision.by === "terminal"
        ? "Answered in the terminal"
        : decision.option_id === null
          ? (decision.text ?? "Cancelled")
          : (request.options.find((o) => o.option_id === decision.option_id)?.name ?? decision.option_id) +
            (decision.text ? `: ${decision.text}` : ""),
  );

  async function answer(optionId: string | null, text: string | null): Promise<void> {
    sending = true;
    error = "";
    try {
      await api.answerPermission(session, request.request_id, optionId, text);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      sending = false;
    }
  }
</script>

<div class="permission" data-testid="permission-box" class:open={decision === null}>
  <div class="head">
    <span class="label">Permission</span>
    {#if request.tool_name}<span class="tool">{request.tool_name}</span>{/if}
    <span class="title">{request.title ?? request.tool_call_id}</span>
  </div>
  {#if shell !== null}
    <pre>$ {shell}</pre>
  {:else if hasInput(request.input)}
    <pre>{JSON.stringify(request.input, null, 2)}</pre>
  {/if}
  {#if decision === null}
    <div class="choices">
      {#each request.options as option (option.option_id)}
        <button
          type="button"
          class={option.kind.startsWith("allow")
            ? "allow"
            : option.kind.startsWith("reject")
              ? "reject"
              : ""}
          disabled={sending}
          onclick={() => answer(option.option_id, words || null)}
          >{option.name}{#if option.recommended}{" (recommended)"}{/if}</button
        >
      {/each}
      <button type="button" disabled={sending} onclick={() => answer(null, null)} title="Cancel the request">
        Cancel
      </button>
    </div>
    {#if decision === null && recommended}
      <div class="note">If nobody answers, {recommended} is picked automatically.</div>
    {/if}
    <div class="words">
      <input
        type="text"
        data-testid="permission-text"
        placeholder="Your own words, with a pick or instead of one…"
        bind:value={words}
        disabled={sending}
      />
      <button
        type="button"
        data-testid="permission-send"
        disabled={sending || words === ""}
        onclick={() => answer(null, words)}>Send</button
      >
    </div>
  {:else}
    <div class="decided">{chosen}{decision.plugin ? ` by plugin ${decision.plugin}` : BY[decision.by]}</div>
  {/if}
  {#if error}<div class="error">{error}</div>{/if}
</div>

<style>
  .permission {
    font-size: 0.9em;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.4rem 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .permission.open {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  .head {
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    min-width: 0;
  }
  .label {
    font-weight: 600;
  }
  .tool {
    color: var(--muted);
  }
  .title {
    font-family: ui-monospace, monospace;
    overflow-wrap: anywhere;
  }
  pre {
    margin: 0;
    font-family: ui-monospace, monospace;
    background: var(--code-bg);
    padding: 0.4rem 0.6rem;
    border-radius: 0.3rem;
    overflow: auto;
    max-height: 16rem;
  }
  .choices {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .words {
    display: flex;
    gap: 0.4rem;
  }
  .words input {
    flex: 1;
    min-width: 0;
    font: inherit;
    color: inherit;
    background: var(--code-bg);
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.2rem 0.5rem;
  }
  .allow {
    border-color: var(--ok);
  }
  .reject {
    border-color: var(--bad);
  }
  .decided {
    color: var(--muted);
  }
  .note {
    color: var(--muted);
    font-size: 0.85em;
  }
  .error {
    color: var(--bad);
  }
</style>
