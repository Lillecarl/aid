import { lexer, type MarkedToken, type Token } from "marked";

// Agent output becomes elements through Svelte, never through innerHTML: raw HTML in the markdown stays text.

export const parse = (text: string): MarkedToken[] => lexer(text) as MarkedToken[];

export const tokensOf = (token: { tokens?: Token[] }): MarkedToken[] => (token.tokens ?? []) as MarkedToken[];

const SAFE_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

/** The link target if it is safe to follow, else null: `javascript:` and friends render as text. */
export function safeHref(href: string): string | null {
  try {
    return SAFE_PROTOCOLS.has(new URL(href, location.href).protocol) ? href : null;
  } catch {
    return null;
  }
}

let decoder: DOMParser | undefined;
const ENTITY = /&(?:#\d+|#x[\da-f]+|[a-z][a-z\d]*);/gi;

/** Resolve character references (`&amp;`, `&#169;`): marked's lexer leaves them for its HTML renderer. Only the
 * references go through the parser; the rest of the text may hold a literal `<` that it would read as a tag. */
export function decodeEntities(text: string): string {
  if (!text.includes("&")) return text;
  decoder ??= new DOMParser();
  // An inert document: nothing in it loads or runs, and only its text is read.
  return text.replace(ENTITY, (ref) => decoder?.parseFromString(ref, "text/html").documentElement.textContent ?? ref);
}
