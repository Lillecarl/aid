<script lang="ts">
  import * as api from "./api";
  import { parseFilter, tagLabels } from "./tags";

  interface Props {
    name: string;
    tags: Record<string, string>;
  }

  let { name, tags }: Props = $props();
  let draft = $state("");
  let error = $state("");

  async function change(update: Record<string, string | null>): Promise<void> {
    try {
      await api.setTags(name, update);
      error = "";
    } catch (e) {
      error = String(e);
    }
  }

  async function add(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const terms = parseFilter(draft);
    if (terms.length === 0) return;
    if (terms.some((t) => t.value === null)) {
      error = "write each tag as key=value";
      return;
    }
    await change(Object.fromEntries(terms.map((t) => [t.key, t.value || null])));
    if (!error) draft = "";
  }
</script>

<div class="tags" data-testid="session-tags">
  {#each tagLabels(tags) as label (label)}
    {@const key = label.slice(0, label.indexOf("="))}
    <span class="tag"
      >{label}<button type="button" aria-label="Remove tag {key}" onclick={() => change({ [key]: null })}>×</button
      ></span
    >
  {/each}
  <form onsubmit={add}>
    <input placeholder="+ key=value" aria-label="Add a tag" data-testid="tag-input" bind:value={draft} />
  </form>
  {#if error}<span class="error">{error}</span>{/if}
</div>

<style>
  .tags {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.85em;
    min-width: 0;
  }
  .tag {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0 0.15rem 0 0.4rem;
    border: 1px solid var(--line);
    border-radius: 0.6rem;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .tag button {
    border: 0;
    background: none;
    padding: 0 0.2rem;
    color: inherit;
    cursor: pointer;
  }
  input {
    width: 9rem;
    font-size: inherit;
  }
  .error {
    color: var(--bad);
  }
</style>
