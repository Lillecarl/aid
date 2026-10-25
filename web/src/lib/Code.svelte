<script lang="ts" module>
  import type { Span } from "./api";

  // Rows remount as the window moves, and a streaming block repeats its text: answers are kept by request.
  const CACHE_SIZE = 500;
  const cache = new Map<string, Promise<Span[] | null>>();
</script>

<script lang="ts">
  import * as api from "./api";

  interface Props {
    text: string;
    /** A fenced block's tag or a lexer name. */
    language?: string | null;
    /** The file the text comes from, when no language is named. */
    path?: string | null;
    /** Line numbers shown beside the text, one per line, kept out of a copy. */
    gutters?: string[] | null;
  }

  // A block still streaming changes on every delta; ask once it holds still this long.
  const SETTLE_MS = 300;

  let { text, language = null, path = null, gutters = null }: Props = $props();
  let spans: Span[] | null = $state(null);

  function lookup(key: string): Promise<Span[] | null> {
    let found = cache.get(key);
    if (found === undefined) {
      found = api.highlight(text, language, path).catch(() => null);
      cache.set(key, found);
      if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
    }
    return found;
  }

  $effect(() => {
    if (!language && !path) return;
    const key = JSON.stringify([language, path, text]);
    let current = true;
    const timer = setTimeout(
      () => void lookup(key).then((found) => current && (spans = found)),
      cache.has(key) ? 0 : SETTLE_MS,
    );
    return () => {
      current = false;
      clearTimeout(timer);
    };
  });

  /** The text cut at span edges; spans past the text (a stale answer) are dropped. */
  const pieces = $derived.by(() => {
    if (spans === null) return [{ text, cls: "" }];
    const out: { text: string; cls: string }[] = [];
    let at = 0;
    for (const [start, end, cls] of spans) {
      if (start < at || end > text.length) continue;
      if (start > at) out.push({ text: text.slice(at, start), cls: "" });
      out.push({ text: text.slice(start, end), cls });
      at = end;
    }
    if (at < text.length) out.push({ text: text.slice(at), cls: "" });
    return out;
  });

  /** The pieces cut again at line ends, one row per gutter. */
  const rows = $derived.by(() => {
    if (gutters === null) return [];
    let row: { text: string; cls: string }[] = [];
    const out = [row];
    for (const piece of pieces) {
      piece.text.split("\n").forEach((part, k) => {
        if (k > 0) out.push((row = []));
        if (part) row.push({ text: part, cls: piece.cls });
      });
    }
    return out;
  });
</script>

{#if gutters === null}<code class="hl"
    >{#each pieces as piece, i (i)}{#if piece.cls}<span class={piece.cls}>{piece.text}</span
        >{:else}{piece.text}{/if}{/each}</code
  >{:else}<code class="hl numbered"
    >{#each rows as row, n (n)}<span class="row"
        ><span class="gutter">{gutters[n] ?? ""}</span><span class="line"
          >{#each row as piece, i (i)}{#if piece.cls}<span class={piece.cls}>{piece.text}</span
              >{:else}{piece.text}{/if}{/each}{"\n"}</span
        ></span
      >{/each}</code
  >{/if}

<style>
  .numbered {
    display: block;
  }
  .row {
    display: flex;
  }
  .gutter {
    flex: none;
    min-width: 4ch;
    padding-right: 1ch;
    text-align: right;
    color: var(--muted);
    user-select: none;
  }
  .line {
    flex: 1;
    min-width: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
</style>
