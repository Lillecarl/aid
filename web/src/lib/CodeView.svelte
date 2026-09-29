<script lang="ts">
  import { LanguageDescription } from "@codemirror/language";
  import { languages } from "@codemirror/language-data";
  import { Compartment, EditorState } from "@codemirror/state";
  import { oneDark } from "@codemirror/theme-one-dark";
  import { EditorView } from "@codemirror/view";
  import { basicSetup } from "codemirror";
  import { onMount } from "svelte";

  interface Props {
    /** Decides the language, by its extension or name. */
    path: string;
    text: string;
  }

  let { path, text }: Props = $props();
  let host: HTMLDivElement;
  let view: EditorView | undefined = $state.raw();
  const language = new Compartment();
  const theme = new Compartment();
  const dark = matchMedia("(prefers-color-scheme: dark)");
  const themeFor = () => (dark.matches ? oneDark : []);

  // CodeMirror styles through style-mod, which adds a style element to a document, and the CSP blocks that; in a
  // shadow root it uses a constructed stylesheet (adoptedStyleSheets), which the CSP allows.
  const layout = EditorView.theme({
    "&": { height: "100%", fontSize: "0.85em" },
    ".cm-scroller": { fontFamily: "ui-monospace, monospace" },
  });

  onMount(() => {
    const root = host.attachShadow({ mode: "open" });
    view = new EditorView({
      parent: root,
      root,
      state: EditorState.create({
        doc: text,
        extensions: [
          basicSetup,
          layout,
          EditorState.readOnly.of(true),
          EditorView.editable.of(false),
          language.of([]),
          theme.of(themeFor()),
        ],
      }),
    });
    const onscheme = () => view?.dispatch({ effects: theme.reconfigure(themeFor()) });
    dark.addEventListener("change", onscheme);
    return () => {
      dark.removeEventListener("change", onscheme);
      view?.destroy();
    };
  });

  $effect(() => {
    const doc = text;
    if (view && view.state.doc.toString() !== doc) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: doc } });
    }
  });

  // Each language is its own chunk, loaded the first time a file needs it.
  $effect(() => {
    const found = LanguageDescription.matchFilename(languages, path.split("/").at(-1) ?? path);
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
