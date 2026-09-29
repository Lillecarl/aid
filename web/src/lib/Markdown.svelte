<script lang="ts">
  import type { MarkedToken } from "marked";
  import { decodeEntities, parse, safeHref, tokensOf } from "./markdown";

  interface Props {
    text: string;
  }

  let { text }: Props = $props();
  const tokens = $derived(parse(text));
</script>

{#snippet nodes(list: MarkedToken[])}
  {#each list as t, i (i)}
    {#if t.type === "paragraph"}
      <p>{@render nodes(tokensOf(t))}</p>
    {:else if t.type === "heading"}
      <svelte:element this={`h${Math.min(t.depth + 2, 6)}`}>{@render nodes(tokensOf(t))}</svelte:element>
    {:else if t.type === "code"}
      <pre data-lang={t.lang || null}><code>{t.text}</code></pre>
    {:else if t.type === "blockquote"}
      <blockquote>{@render nodes(tokensOf(t))}</blockquote>
    {:else if t.type === "list"}
      <svelte:element this={t.ordered ? "ol" : "ul"} start={t.ordered && t.start !== "" ? t.start : null}>
        {#each t.items as item, j (j)}
          <li class:task={item.task}>{@render nodes(tokensOf(item))}</li>
        {/each}
      </svelte:element>
    {:else if t.type === "table"}
      <div class="table">
        <table>
          <thead>
            <tr>
              {#each t.header as cell, j (j)}
                <th class={cell.align}>{@render nodes(tokensOf(cell))}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each t.rows as row, j (j)}
              <tr>
                {#each row as cell, k (k)}
                  <td class={cell.align}>{@render nodes(tokensOf(cell))}</td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else if t.type === "hr"}
      <hr />
    {:else if t.type === "checkbox"}
      <input type="checkbox" checked={t.checked} disabled />
    {:else if t.type === "strong"}
      <strong>{@render nodes(tokensOf(t))}</strong>
    {:else if t.type === "em"}
      <em>{@render nodes(tokensOf(t))}</em>
    {:else if t.type === "del"}
      <del>{@render nodes(tokensOf(t))}</del>
    {:else if t.type === "codespan"}
      <code>{t.text}</code>
    {:else if t.type === "br"}
      <br />
    {:else if t.type === "link"}
      {@const href = safeHref(t.href)}
      {#if href !== null}
        <a {href} title={t.title ?? null} target="_blank" rel="noopener noreferrer">{@render nodes(tokensOf(t))}</a>
      {:else}
        {@render nodes(tokensOf(t))}
      {/if}
    {:else if t.type === "image"}
      <!-- The CSP loads no images from elsewhere: link to it instead. -->
      {@const href = safeHref(t.href)}
      {#if href !== null}
        <a {href} target="_blank" rel="noopener noreferrer">🖼 {t.text || href}</a>
      {:else}
        {t.text}
      {/if}
    {:else if t.type === "text"}
      {#if t.tokens}{@render nodes(tokensOf(t))}{:else}{decodeEntities(t.text)}{/if}
    {:else if t.type === "escape"}
      {t.text}
    {:else if t.type === "html"}
      {#if t.block}<pre class="html">{t.text}</pre>{:else}{t.text}{/if}
    {/if}
  {/each}
{/snippet}

<div class="md">{@render nodes(tokens)}</div>

<style>
  .md {
    overflow-wrap: anywhere;
  }
  .md :global(:first-child) {
    margin-top: 0;
  }
  .md :global(:last-child) {
    margin-bottom: 0;
  }
  p,
  ul,
  ol,
  blockquote,
  pre,
  .table {
    margin: 0.5em 0;
  }
  ul,
  ol {
    padding-left: 1.5em;
  }
  li.task {
    list-style: none;
    margin-left: -1.2em;
  }
  li > :global(p) {
    margin: 0.2em 0;
  }
  code {
    font-family: ui-monospace, monospace;
    font-size: 0.9em;
    background: var(--code-bg);
    padding: 0.1em 0.3em;
    border-radius: 0.2rem;
  }
  pre {
    background: var(--code-bg);
    padding: 0.6em 0.8em;
    border-radius: 0.3rem;
    overflow-x: auto;
    white-space: pre;
    position: relative;
  }
  pre code {
    background: none;
    padding: 0;
  }
  pre[data-lang]::before {
    content: attr(data-lang);
    position: absolute;
    top: 0.2em;
    right: 0.5em;
    font-size: 0.75em;
    color: var(--muted);
  }
  pre.html {
    font-family: ui-monospace, monospace;
    font-size: 0.9em;
  }
  blockquote {
    border-left: 3px solid var(--line);
    padding-left: 0.8em;
    color: var(--muted);
  }
  .table {
    overflow-x: auto;
  }
  table {
    border-collapse: collapse;
  }
  th,
  td {
    border: 1px solid var(--line);
    padding: 0.25em 0.6em;
  }
  .center {
    text-align: center;
  }
  .right {
    text-align: right;
  }
  a {
    color: var(--accent);
  }
  hr {
    border: none;
    border-top: 1px solid var(--line);
  }
</style>
