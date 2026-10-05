<script lang="ts">
  import { command, type DiffLine, diffLines, gist, hasInput, readable, type Tool } from "./tools";

  interface Props {
    tool: Tool;
  }

  const STATUS_ICON: Record<string, string> = { pending: "○", in_progress: "◐", completed: "✓", failed: "✗" };
  const LINE_CLASS: Record<DiffLine["op"], string> = { " ": "", "-": "del", "+": "add", "…": "gap" };

  let { tool }: Props = $props();
  const status = $derived(tool.status ?? "pending");
  const hint = $derived(gist(tool.input) ?? tool.paths[0] ?? null);
  const shell = $derived(command(tool.input));
  const showInput = $derived(shell === null && hasInput(tool.input) && !tool.diffs.length);
  const hasDetails = $derived(
    shell !== null || showInput || tool.output !== null || tool.diffs.length > 0 || tool.paths.length > 0,
  );
  // Only while open: a diff of a big file costs time, and most cards stay closed.
  let open = $state(false);
</script>

<details class="tool {status}" data-testid="tool-box" bind:open>
  <summary title={tool.id} class:plain={!hasDetails}>
    <span class="icon" aria-label={status}>{STATUS_ICON[status] ?? "•"}</span>
    {#if tool.kind}<span class="kind">{tool.kind}</span>{/if}
    <span class="title">{tool.title ?? tool.id}</span>
    {#if hint && hint !== tool.title}<span class="hint">{hint}</span>{/if}
  </summary>
  {#if open && hasDetails}
    <div class="body">
    {#if tool.paths.length}
      <ul class="paths" data-testid="tool-paths">
        {#each tool.paths as path (path)}<li>{path}</li>{/each}
      </ul>
    {/if}
      {#if shell !== null}
        <pre class="input">$ {shell}</pre>
      {:else if showInput}
        <div class="label">input</div>
        <pre class="input">{JSON.stringify(tool.input, null, 2)}</pre>
      {/if}
      {#each tool.diffs as diff (diff.path)}
        <div class="diff">
          <div class="path">{diff.old === null ? "new file " : ""}{diff.path}</div>
          <pre>{#each diffLines(diff.old, diff.new) as line, k (k)}<span class="l {LINE_CLASS[line.op]}"
                >{line.op === "…" ? "⋯" : `${line.op} ${line.text}`}</span
              >{/each}</pre>
        </div>
      {/each}
      {#if tool.output !== null}
        <div class="label">{status === "failed" ? "error" : "output"}</div>
        <pre class="output">{readable(tool.output)}</pre>
      {/if}
    </div>
  {/if}
</details>

<style>
  .tool {
    font-size: 0.85em;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    max-width: 100%;
    width: fit-content;
  }
  .tool[open] {
    width: auto;
  }
  summary {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.15rem 0.5rem;
    cursor: pointer;
    list-style: none;
    min-width: 0;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary.plain {
    cursor: default;
  }
  .title,
  .hint {
    font-family: ui-monospace, monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .title {
    flex-shrink: 0;
    /* Not a percentage: the card is fit-content, so a percentage resolves against the shrunk width. */
    max-width: 40ch;
  }
  .hint {
    color: var(--muted);
    min-width: 0;
  }
  .kind {
    color: var(--muted);
    text-transform: lowercase;
  }
  .icon {
    width: 1em;
    text-align: center;
  }
  .completed .icon {
    color: var(--ok);
  }
  .failed {
    border-color: var(--bad);
  }
  .failed .icon {
    color: var(--bad);
  }
  .in_progress .icon,
  .pending .icon {
    color: var(--accent);
  }
  .body {
    border-top: 1px solid var(--line);
    padding: 0.4rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  pre {
    margin: 0;
    font-family: ui-monospace, monospace;
    background: var(--code-bg);
    padding: 0.4rem 0.6rem;
    border-radius: 0.3rem;
    overflow: auto;
    max-height: 24rem;
    white-space: pre;
  }
  .output {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .label {
    color: var(--muted);
    font-size: 0.85em;
    margin-bottom: -0.3rem;
  }
  .failed .output {
    color: var(--bad);
  }
  .paths {
    margin: 0;
    padding-left: 1.2em;
    font-family: ui-monospace, monospace;
    color: var(--muted);
  }
  .diff .path {
    font-family: ui-monospace, monospace;
    color: var(--muted);
    margin-bottom: 0.2rem;
  }
  .l.add {
    color: var(--ok);
    background: color-mix(in srgb, var(--ok) 12%, transparent);
  }
  .l.del {
    color: var(--bad);
    background: color-mix(in srgb, var(--bad) 12%, transparent);
  }
  .l.gap {
    color: var(--muted);
  }
  .l {
    display: block;
  }
</style>
