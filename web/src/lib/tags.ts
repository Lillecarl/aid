import type { SessionInfo } from "./api";

/** One term of a list filter: a key that must be there, with this value when one is named. */
export type TagTerm = { key: string; value: string | null };

/** `key` and `key=value` terms, split on whitespace, as `aid list --tag` reads them. */
export function parseFilter(text: string): TagTerm[] {
  return text
    .split(/\s+/)
    .filter((term) => term !== "")
    .map((term) => {
      const at = term.indexOf("=");
      return at < 0 ? { key: term, value: null } : { key: term.slice(0, at), value: term.slice(at + 1) };
    });
}

export function tagged(info: SessionInfo, terms: TagTerm[]): boolean {
  const tags = info.tags;
  return terms.every(({ key, value }) => key in tags && (value === null || tags[key] === value));
}

/** The tags in a stable order, as `key=value`. */
export const tagLabels = (tags: Record<string, string>): string[] =>
  Object.keys(tags)
    .sort()
    .map((key) => `${key}=${tags[key]}`);
