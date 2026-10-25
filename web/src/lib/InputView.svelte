<script lang="ts">
  import Code from "./Code.svelte";
  import { inputFields, unifiedClass } from "./tools";

  interface Props {
    input: unknown;
  }

  let { input }: Props = $props();
  const fields = $derived(inputFields(input));
</script>

{#if fields === null}
  <pre data-testid="tool-input">{JSON.stringify(input, null, 2)}</pre>
{:else}
  {#each fields as field (field.key)}
    <div class="field" data-testid="tool-input">
      <div class="key">{field.key}</div>
      {#if field.diff}
        <pre class="diff">{#each field.text.split("\n") as line, k (k)}<span class="l {unifiedClass(line)}"
              >{line}</span
            >{/each}</pre>
      {:else if field.language || field.path}
        <pre><Code text={field.text} language={field.language} path={field.path} /></pre>
      {:else}
        <pre>{field.text}</pre>
      {/if}
    </div>
  {/each}
{/if}

<style>
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .key {
    color: var(--muted);
    font-size: 0.85em;
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
  .l {
    display: block;
  }
  .l.add {
    color: var(--ok);
    background: color-mix(in srgb, var(--ok) 12%, transparent);
  }
  .l.del {
    color: var(--bad);
    background: color-mix(in srgb, var(--bad) 12%, transparent);
  }
  .l.hunk,
  .l.file {
    color: var(--muted);
  }
</style>
