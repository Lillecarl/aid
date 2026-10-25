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
  }

  // A block still streaming changes on every delta; ask once it holds still this long.
  const SETTLE_MS = 300;

  let { text, language = null, path = null }: Props = $props();
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
</script>

<code class="hl">{#each pieces as piece, i (i)}{#if piece.cls}<span class={piece.cls}>{piece.text}</span
      >{:else}{piece.text}{/if}{/each}</code
>
