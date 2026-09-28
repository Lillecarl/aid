<script lang="ts">
  import { onMount, tick } from "svelte";
  import { screenEventsUrl, screenStylesheetUrl, watch } from "./api";
  import type { PaneView } from "./api";

  let { name }: { name: string } = $props();

  let view: PaneView | null = $state(null);
  let problem = $state("");
  let box: HTMLDivElement | undefined = $state();

  /** Links in the pane come from the program in it (OSC 8). pymux keeps only safe schemes; open them apart. */
  async function isolateLinks(): Promise<void> {
    await tick();
    for (const link of box?.querySelectorAll("a") ?? []) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  }

  // Mounted only while its tab is shown.
  onMount(() =>
    watch<PaneView>(
      screenEventsUrl(name),
      (data) => {
        view = data;
        void isolateLinks();
      },
      (text) => (problem = text),
    ),
  );
</script>

<svelte:head>
  <link rel="stylesheet" href={screenStylesheetUrl(name)} />
</svelte:head>

{#if problem}<p class="problem">{problem}</p>{/if}
{#if view?.overlay}<p class="muted">pymux shows {view.overlay} over this pane, which this view does not draw.</p>{/if}
<div class="screen" bind:this={box}>
  {#if view === null}
    <p class="muted">Loading…</p>
  {:else}
    <!-- HTML that pymux drew: pyte escapes the pane's text and keeps links to safe schemes. -->
    {@html view.html}
  {/if}
</div>

<style>
  .screen {
    flex: 1;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
  }
  .screen :global(pre) {
    margin: 0;
    padding: 0.5rem;
    width: max-content;
  }
  .muted {
    color: var(--muted);
  }
  .problem {
    color: #d33;
  }
</style>
