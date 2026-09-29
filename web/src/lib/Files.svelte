<script lang="ts">
  import * as api from "./api";
  import type { FileView } from "./api";
  import FileTree from "./FileTree.svelte";

  interface Props {
    name: string;
  }

  let { name }: Props = $props();
  let tree: FileTree | undefined = $state();
  let selected: string | null = $state(null);
  let file: FileView | null = $state(null);
  let problem = $state("");
  let loading = $state(false);

  async function open(path: string): Promise<void> {
    selected = path;
    loading = true;
    try {
      file = await api.readFile(name, path);
      problem = "";
    } catch (e) {
      file = null;
      problem = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  function refresh(): void {
    tree?.refresh();
    if (selected !== null) void open(selected);
  }

  const size = (bytes: number): string =>
    bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / 1048576).toFixed(1)} MiB`;
</script>

<div class="files">
  <nav>
    <div class="bar">
      <span class="muted">Working directory</span>
      <button type="button" onclick={refresh} title="Read the tree and the open file again">Refresh</button>
    </div>
    <div class="tree"><FileTree bind:this={tree} {name} {selected} onopen={open} /></div>
  </nav>
  <section>
    {#if problem}
      <p class="error">{problem}</p>
    {:else if file === null}
      <p class="muted">{loading ? "Loading…" : "Choose a file."}</p>
    {:else}
      <div class="bar">
        <span class="path">{file.path}</span>
        <span class="muted">{size(file.size)}{file.truncated ? ", first 1 MiB shown" : ""}</span>
      </div>
      {#if file.text === null}
        <p class="muted">A binary file.</p>
      {:else}
        <div class="view">
          <!-- Its own chunk: CodeMirror is most of the page's weight, and many visits never open a file. -->
          {#await import("./CodeView.svelte")}
            <p class="muted">Loading the viewer…</p>
          {:then { default: CodeView }}
            <CodeView path={file.path} text={file.text} highlights={file.highlights ?? null} />
          {/await}
        </div>
      {/if}
    {/if}
  </section>
</div>

<style>
  .files {
    display: grid;
    grid-template-columns: minmax(12rem, 18rem) 1fr;
    gap: 0.75rem;
    flex: 1;
    min-height: 0;
  }
  nav,
  section {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
  }
  .tree {
    flex: 1;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.3rem 0;
  }
  .bar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    min-height: 2rem;
  }
  .bar button {
    padding: 0.1rem 0.5rem;
    font-size: 0.85em;
  }
  .path {
    font-family: ui-monospace, monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .muted {
    color: var(--muted);
    font-size: 0.85em;
    white-space: nowrap;
  }
  .view {
    flex: 1;
    min-height: 0;
  }
  @media (max-width: 700px) {
    .files {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(8rem, 35%) 1fr;
    }
  }
</style>
