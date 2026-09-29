<script lang="ts">
  import "pymux-pane";
  import { onMount } from "svelte";
  import * as api from "./api";
  import type { SessionStatus } from "./api";

  let { name }: { name: string } = $props();

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const src = $derived(`${scheme}://${location.host}/api/sessions/${encodeURIComponent(name)}/pane`);

  let note = $state("Connecting…");
  let problem = $state("");
  // null until the status stream first answers: neither the pane nor the Start button before then.
  let running: boolean | null = $state(null);
  let starting = $state(false);
  // Mounted only while its tab is shown, and the element only while the browser tab is visible: removing it
  // closes its WebSocket, so nothing streams that nobody sees.
  let visible = $state(document.visibilityState === "visible");

  onMount(() => {
    const onvisibility = (): void => {
      visible = document.visibilityState === "visible";
    };
    document.addEventListener("visibilitychange", onvisibility);
    const unwatch = api.watch<SessionStatus>(
      api.statusEventsUrl(name),
      (status) => {
        // A stopped session's pane is gone; the element mounts afresh, and connects, when it runs again.
        if (status.running && running === false) note = "Connecting…";
        running = status.running;
      },
      (text) => (problem = text),
    );
    return () => {
      unwatch();
      document.removeEventListener("visibilitychange", onvisibility);
    };
  });

  async function start(): Promise<void> {
    starting = true;
    try {
      await api.control(name, "start");
      problem = "";
    } catch (e) {
      problem = e instanceof Error ? e.message : String(e);
    } finally {
      starting = false;
    }
  }
</script>

{#if running === false}
  <p class="muted">The session is stopped, so it has no pane.</p>
  <div>
    <button type="button" onclick={start} disabled={starting}>{starting ? "Starting…" : "Start"}</button>
  </div>
{:else if running}
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
{/if}
{#if problem}
  <p class="error">{problem}</p>
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
