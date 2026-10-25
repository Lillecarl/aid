<script lang="ts">
  import { onMount } from "svelte";
  import * as api from "./api";
  import { costLine } from "./usage";

  interface Props {
    name: string;
    label: string;
    active: boolean;
    onselect: () => void;
  }

  let { name, label, active, onselect }: Props = $props();
  let badge = $state("");
  let working = $state(false);

  onMount(() =>
    api.followStatus(
      name,
      (status) => {
        const context =
          status.context_used !== null && status.context_size
            ? `${Math.round((100 * status.context_used) / status.context_size)}%`
            : "";
        badge = [costLine(status.cost), context].filter(Boolean).join(" · ");
        working = status.working;
      },
      () => undefined,
    ),
  );
</script>

<button
  type="button"
  role="tab"
  aria-selected={active}
  class:active
  class:working
  data-testid="linked-tab"
  title={name}
  onclick={onselect}
  >{label}{#if badge}<span class="badge" data-testid="linked-badge">{badge}</span>{/if}</button
>

<style>
  button {
    width: auto;
    font-size: 0.85em;
    padding: 0.1rem 0.5rem;
    border: 1px solid var(--line);
    border-radius: 1rem;
    background: none;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
  }
  .active {
    color: inherit;
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
  }
  .working {
    border-style: dashed;
  }
  .badge {
    margin-left: 0.4em;
    color: var(--muted);
    font-size: 0.9em;
  }
</style>
