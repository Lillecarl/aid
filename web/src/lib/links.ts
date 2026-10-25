import type { SessionInfo } from "./api";

/** Sessions the list leaves out: every supervisor, and every session working for one (its classifier). They
 * show as tabs under the sessions they watch instead. */
export function linkedOnly(sessions: SessionInfo[]): Set<string> {
  const parents = new Map(sessions.map((s) => [s.name, s.parent]));
  const supervisors = new Set(sessions.flatMap((s) => (s.supervisor ? [s.supervisor] : [])));
  // Bounded by the list's length: parent links come from specs, and nothing here trusts them to end.
  const linked = (name: string, depth: number): boolean => {
    if (supervisors.has(name)) return true;
    const parent = parents.get(name);
    return parent != null && depth < sessions.length && linked(parent, depth + 1);
  };
  return new Set(sessions.filter((s) => linked(s.name, 0)).map((s) => s.name));
}

/** `name`'s tabs: itself and the sessions working for it, then its supervisor and the sessions working for
 * that (its classifier), each after its parent. */
export function family(sessions: SessionInfo[], name: string): string[] {
  const tabs: string[] = [];
  const visit = (n: string): void => {
    if (tabs.includes(n)) return;
    tabs.push(n);
    for (const s of sessions) if (s.parent === n) visit(s.name);
  };
  visit(name);
  const supervisor = sessions.find((s) => s.name === name)?.supervisor;
  if (supervisor && sessions.some((s) => s.name === supervisor)) visit(supervisor);
  return tabs;
}

/** A tab's label: a child's name without its parent's prefix (`superds.classifier` → `classifier`). */
export function tabLabel(sessions: SessionInfo[], name: string): string {
  const parent = sessions.find((s) => s.name === name)?.parent;
  return parent && name.startsWith(`${parent}.`) ? name.slice(parent.length + 1) : name;
}
