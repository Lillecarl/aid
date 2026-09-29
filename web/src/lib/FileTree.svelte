<script lang="ts">
  import { onMount } from "svelte";
  import { SvelteMap, SvelteSet } from "svelte/reactivity";
  import * as api from "./api";
  import type { FileEntry } from "./api";

  interface Props {
    name: string;
    selected: string | null;
    onopen: (path: string) => void;
  }

  type Listing = FileEntry[] | { error: string } | "loading";

  let { name, selected, onopen }: Props = $props();

  // Directories load when first opened, and stay loaded until "Refresh".
  const listings = new SvelteMap<string, Listing>();
  const expanded = new SvelteSet<string>();

  const join = (dir: string, entry: string): string => (dir === "" ? entry : `${dir}/${entry}`);

  async function load(dir: string): Promise<void> {
    listings.set(dir, "loading");
    try {
      listings.set(dir, await api.listFiles(name, dir));
    } catch (e) {
      listings.set(dir, { error: e instanceof Error ? e.message : String(e) });
    }
  }

  function toggle(dir: string): void {
    if (expanded.has(dir)) {
      expanded.delete(dir);
      return;
    }
    expanded.add(dir);
    if (!listings.has(dir)) void load(dir);
  }

  export function refresh(): void {
    const open = ["", ...expanded];
    listings.clear();
    for (const dir of open) void load(dir);
  }

  onMount(() => void load(""));
</script>

{#snippet level(dir: string, depth: number)}
  {@const listing = listings.get(dir)}
  {#if listing === undefined || listing === "loading"}
    <li class="note" style:--depth={depth}>Loading…</li>
  {:else if "error" in listing}
    <li class="note error" style:--depth={depth}>{listing.error}</li>
  {:else}
    {#each listing as entry (entry.name)}
      {@const path = join(dir, entry.name)}
      <li>
        <button
          type="button"
          class:dir={entry.dir}
          class:selected={path === selected}
          style:--depth={depth}
          title={entry.dir ? path : `${path} · ${entry.size ?? "?"} bytes`}
          aria-expanded={entry.dir ? expanded.has(path) : undefined}
          onclick={() => (entry.dir ? toggle(path) : onopen(path))}
        >
          <span class="twisty">{entry.dir ? (expanded.has(path) ? "▾" : "▸") : ""}</span>{entry.name}
        </button>
      </li>
      {#if entry.dir && expanded.has(path)}
        {@render level(path, depth + 1)}
      {/if}
    {:else}
      <li class="note" style:--depth={depth}>Empty</li>
    {/each}
  {/if}
{/snippet}

<ul class="tree" role="tree">
  {@render level("", 0)}
</ul>

<style>
  .tree {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 0.9em;
  }
  button {
    display: flex;
    width: 100%;
    text-align: left;
    border: none;
    border-radius: 0.2rem;
    background: none;
    padding: 0.1rem 0.4rem 0.1rem calc(0.4rem + var(--depth) * 1rem);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font: inherit;
    color: inherit;
  }
  button:hover {
    background: var(--code-bg);
  }
  button.selected {
    background: color-mix(in srgb, var(--accent) 20%, transparent);
  }
  .twisty {
    display: inline-block;
    width: 1em;
    flex-shrink: 0;
    color: var(--muted);
  }
  .note {
    color: var(--muted);
    padding-left: calc(1.4rem + var(--depth) * 1rem);
  }
  .note.error {
    color: var(--bad);
  }
</style>
