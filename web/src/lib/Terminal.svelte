<script lang="ts">
  import "pymux-pane";
  import { onMount } from "svelte";

  let { name }: { name: string } = $props();

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const src = $derived(`${scheme}://${location.host}/api/sessions/${encodeURIComponent(name)}/pane`);

  let note = $state("Connecting…");
  // Mounted only while its tab is shown, and the element only while the browser tab is visible: removing it
  // closes its WebSocket, so nothing streams that nobody sees.
  let visible = $state(document.visibilityState === "visible");

  onMount(() => {
    const onvisibility = (): void => {
      visible = document.visibilityState === "visible";
    };
    document.addEventListener("visibilitychange", onvisibility);
    return () => document.removeEventListener("visibilitychange", onvisibility);
  });
</script>

<p class="muted">{note}</p>
{#if visible}
  <pymux-pane
    {src}
    onconnected={(event: Event) => {
      const pane = event.currentTarget as HTMLElementTagNameMap["pymux-pane"];
      note = pane.writable ? "Click the pane and type: keys go to the session." : "Showing only.";
    }}
    onclosed={() => (note = "The stream ended: the session stopped, or its pane went away.")}
    onerror={() => (note = "The stream failed.")}
  ></pymux-pane>
{/if}

<style>
  pymux-pane {
    flex: 1;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    padding: 0.5rem;
  }
  .muted {
    color: var(--muted);
    margin: 0 0 0.5rem;
    font-size: 0.85em;
  }
</style>
