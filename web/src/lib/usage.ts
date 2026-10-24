import type { Started, Usage } from "./api";

export const tokens = (n: number): string =>
  n < 1000 ? String(n) : n < 1_000_000 ? `${(n / 1000).toFixed(1)}k` : `${(n / 1_000_000).toFixed(2)}M`;

const part = (n: number | null, label: string): string | null => (n ? `${tokens(n)} ${label}` : null);

/** Cents, or two significant digits below a cent: a native turn often costs a tenth of one. */
export const money = (amount: number, currency: string): string =>
  `${amount >= 0.01 || amount === 0 ? amount.toFixed(2) : amount.toPrecision(2)} ${currency}`;

/** Accumulated cost across turns, per currency, highest amount first; empty when nothing reported. */
export function costLine(record: Record<string, number>): string {
  return Object.entries(record)
    .sort(([, a], [, b]) => b - a)
    .map(([currency, amount]) => money(amount, currency))
    .join(", ");
}

/** One line: the models, then only the counts the agent reported. */
export function usageLine(u: Usage): string {
  const context =
    u.context_used !== null && u.context_size
      ? `context ${Math.round((100 * u.context_used) / u.context_size)}% of ${tokens(u.context_size)}`
      : null;
  const cost = u.cost ? money(u.cost.amount, u.cost.currency) : null;
  const session = u.session_cost ? `session ${money(u.session_cost.amount, u.session_cost.currency)}` : null;
  return [
    u.agent ? (u.agent.startsWith("fork ") ? u.agent : `subagent ${u.agent}`) : null,
    u.models.join(", ") || null,
    part(u.input_tokens, "in"),
    part(u.output_tokens, "out"),
    part(u.cache_read_tokens, "cached"),
    part(u.cache_write_tokens, "cache written"),
    u.requests && u.requests > 1 ? `${u.requests} requests` : null,
    context,
    cost,
    session,
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
