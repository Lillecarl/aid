<script lang="ts">
  import { LanguageDescription, syntaxHighlighting } from "@codemirror/language";
  import { languages } from "@codemirror/language-data";
  import { Compartment, EditorState, RangeSetBuilder } from "@codemirror/state";
  import { Decoration, EditorView, type DecorationSet } from "@codemirror/view";
  import { tagHighlighter, tags as t } from "@lezer/highlight";
  import { basicSetup } from "codemirror";
  import { onMount } from "svelte";

  interface Props {
    /** Decides the language, by its extension or name. */
    path: string;
    text: string;
    /** aid web's tree-sitter spans; without them CodeMirror's own parser highlights what it knows. */
    highlights?: [number, number, string][] | null;
  }

  let { path, text, highlights = null }: Props = $props();
  let host: HTMLDivElement;
  let view: EditorView | undefined = $state.raw();
  const language = new Compartment();
  const marks = new Compartment();
  const scheme = new Compartment();
  const media = matchMedia("(prefers-color-scheme: dark)");

  // CodeMirror styles through style-mod, which adds a style element to a document, and the CSP blocks that; in a
  // shadow root it uses a constructed stylesheet (adoptedStyleSheets), which the CSP allows. The page's
  // variables inherit into the shadow root; /theme.css is linked inside it for the token classes.
  const layout = EditorView.theme({
    "&": { height: "100%", fontSize: "0.85em", color: "var(--text)", backgroundColor: "var(--surface)" },
    ".cm-scroller": { fontFamily: "ui-monospace, monospace" },
    ".cm-content": { background: "none" },
    ".cm-gutters": { backgroundColor: "var(--surface-raised)", color: "var(--muted)", borderColor: "var(--line)" },
    ".cm-activeLine, .cm-activeLineGutter": {
      backgroundColor: "color-mix(in srgb, var(--accent) 10%, transparent)",
    },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
      backgroundColor: "color-mix(in srgb, var(--accent) 30%, transparent) !important",
    },
  });

  // The same Pygments classes the server sends, for the files only CodeMirror's parsers know.
  const pygments = tagHighlighter([
    { tag: t.comment, class: "c" },
    { tag: t.docComment, class: "sd" },
    { tag: t.keyword, class: "k" },
    { tag: [t.controlKeyword, t.operatorKeyword], class: "k" },
    { tag: t.definitionKeyword, class: "kd" },
    { tag: t.moduleKeyword, class: "kn" },
    { tag: [t.bool, t.null, t.atom], class: "kc" },
    { tag: t.string, class: "s" },
    { tag: t.special(t.string), class: "ss" },
    { tag: t.escape, class: "se" },
    { tag: t.regexp, class: "sr" },
    { tag: t.number, class: "m" },
    { tag: t.operator, class: "o" },
    { tag: t.punctuation, class: "p" },
    { tag: t.function(t.variableName), class: "nf" },
    { tag: t.function(t.definition(t.variableName)), class: "nf" },
    { tag: t.standard(t.variableName), class: "nb" },
    { tag: t.className, class: "nc" },
    { tag: t.typeName, class: "nc" },
    { tag: t.standard(t.typeName), class: "kt" },
    { tag: t.namespace, class: "nn" },
    { tag: t.propertyName, class: "py" },
    { tag: t.attributeName, class: "na" },
    { tag: t.tagName, class: "nt" },
    { tag: t.labelName, class: "nl" },
    { tag: t.macroName, class: "fm" },
    { tag: t.constant(t.variableName), class: "no" },
    { tag: t.self, class: "bp" },
    { tag: t.meta, class: "cp" },
    { tag: t.heading, class: "gh" },
    { tag: t.emphasis, class: "ge" },
    { tag: t.strong, class: "gs" },
    { tag: t.link, class: "sx" },
    { tag: t.inserted, class: "gi" },
    { tag: t.deleted, class: "gd" },
    { tag: t.changed, class: "gu" },
    { tag: t.invalid, class: "err" },
  ]);

  function decorations(spans: [number, number, string][], length: number): DecorationSet {
    const builder = new RangeSetBuilder<Decoration>();
    for (const [from, to, cls] of spans) {
      if (to <= length && from < to) builder.add(from, to, Decoration.mark({ class: cls }));
    }
    return builder.finish();
  }

  const markExtension = (spans: [number, number, string][] | null, length: number) =>
    spans === null ? [] : EditorView.decorations.of(decorations(spans, length));

  // A --theme makes the page one scheme; without one it follows the browser.
  const isDark = () => {
    const declared = getComputedStyle(document.documentElement).colorScheme;
    return declared === "dark" || (declared !== "light" && media.matches);
  };
  const schemeExtension = () => EditorView.darkTheme.of(isDark());

  onMount(() => {
    const root = host.attachShadow({ mode: "open" });
    const colors = document.createElement("link");
    colors.rel = "stylesheet";
    colors.href = "/theme.css";
    root.append(colors);
    view = new EditorView({
      parent: root,
      root,
      state: EditorState.create({
        doc: text,
        extensions: [
          basicSetup,
          layout,
          syntaxHighlighting(pygments),
          EditorView.contentAttributes.of({ class: "hl" }),
          EditorState.readOnly.of(true),
          EditorView.editable.of(false),
          language.of([]),
          marks.of(markExtension(highlights, text.length)),
          scheme.of(schemeExtension()),
        ],
      }),
    });
    const onscheme = () => view?.dispatch({ effects: scheme.reconfigure(schemeExtension()) });
    media.addEventListener("change", onscheme);
    return () => {
      media.removeEventListener("change", onscheme);
      view?.destroy();
    };
  });

  // Text and spans in one transaction: spans for the new text must never land on the old one.
  $effect(() => {
    const doc = text;
    const spans = highlights;
    if (!view) return;
    const changes = view.state.doc.toString() === doc ? undefined : { from: 0, to: view.state.doc.length, insert: doc };
    view.dispatch({ changes, effects: marks.reconfigure(markExtension(spans, doc.length)) });
  });

  // Each language is its own chunk, loaded the first time a file needs it. Not when the server highlighted.
  $effect(() => {
    const found = highlights === null ? LanguageDescription.matchFilename(languages, path.split("/").at(-1) ?? path) : null;
    let current = true;
    if (!found) view?.dispatch({ effects: language.reconfigure([]) });
    else
      void found.load().then((support) => {
        if (current) view?.dispatch({ effects: language.reconfigure(support) });
      });
    return () => {
      current = false;
    };
  });
</script>

<div class="code" bind:this={host}></div>

<style>
  .code {
    height: 100%;
    min-height: 0;
    border: 1px solid var(--line);
    border-radius: 0.3rem;
    overflow: hidden;
  }
</style>
