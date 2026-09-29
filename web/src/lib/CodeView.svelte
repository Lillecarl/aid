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

  // CodeMirror styles through constructed stylesheets (style-mod's adoptedStyleSheets), which the CSP allows.
  onMount(() => {
    view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: text,
        extensions: [
          basicSetup,
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
  .code :global(.cm-editor) {
    height: 100%;
    font-size: 0.85em;
  }
  .code :global(.cm-scroller) {
    font-family: ui-monospace, monospace;
  }
</style>
