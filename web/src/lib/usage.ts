import type { Started, Usage } from "./api";

export const tokens = (n: number): string =>
  n < 1000 ? String(n) : n < 1_000_000 ? `${(n / 1000).toFixed(1)}k` : `${(n / 1_000_000).toFixed(2)}M`;

const part = (n: number | null, label: string): string | null => (n ? `${tokens(n)} ${label}` : null);

/** One line: the models, then only the counts the agent reported. */
export function usageLine(u: Usage): string {
  const context =
    u.context_used !== null && u.context_size
      ? `context ${Math.round((100 * u.context_used) / u.context_size)}% of ${tokens(u.context_size)}`
      : null;
  const cost = u.session_cost ? `session ${u.session_cost.amount.toFixed(2)} ${u.session_cost.currency}` : null;
  return [
    u.agent ? `subagent ${u.agent}` : null,
    u.models.join(", ") || null,
    part(u.input_tokens, "in"),
    part(u.output_tokens, "out"),
    part(u.cache_read_tokens, "cached"),
    part(u.cache_write_tokens, "cache written"),
    u.requests && u.requests > 1 ? `${u.requests} requests` : null,
    context,
    cost,
  ]
    .filter((p) => p !== null)
    .join(" · ");
}

export function startedLine(s: Started): string {
  return [
    s.resumed ? "Resumed" : "Started",
    s.agent,
    s.model,
    s.agent_session ? `session ${s.agent_session}` : null,
    `pid ${s.pid}`,
  ]
    .filter((p) => p !== null)
    .join(" · ");
}
