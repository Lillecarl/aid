"""The page, its script and its style. Python strings, because the package ships only .py files.

Agent output reaches the DOM through textContent only: it is untrusted text, and innerHTML would run it.
"""

from __future__ import annotations

from typing import Final

HTML: Final = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aid</title>
<link rel="stylesheet" href="/app.css">
<script src="/app.js" defer></script>
</head>
<body>
<header><strong>aid</strong><span id="who"></span><button id="logout" type="button">Log out</button></header>
<main>
  <aside>
    <h2>Sessions</h2>
    <ul id="sessions"></ul>
    <form id="create">
      <h2>New session</h2>
      <label>Name <input name="name" required pattern="[A-Za-z0-9_.\\-]{1,64}"></label>
      <label>Kind
        <select name="kind">
          <option value="claude-tty">Claude Code (interactive)</option>
          <option value="acp">ACP agent</option>
          <option value="pydantic-ai">pydantic-ai agent</option>
        </select>
      </label>
      <label>Working directory <input name="cwd" required></label>
      <label data-kind="acp">Command <input name="command" placeholder="claude-agent-acp"></label>
      <label data-kind="pydantic-ai">Target <input name="target" placeholder="package.module:agent"></label>
      <label data-kind="claude-tty">Claude arguments <input name="args" placeholder="--model opus"></label>
      <label data-kind="claude-tty" class="check"><input type="checkbox" name="trust"> Trust the directory</label>
      <button type="submit">Create</button>
      <p id="create-error" class="error"></p>
    </form>
  </aside>
  <section>
    <h2 id="title">No session selected</h2>
    <div id="log"></div>
    <form id="prompt">
      <textarea name="text" rows="4" placeholder="Prompt (Ctrl+Enter sends)" disabled></textarea>
      <div class="buttons">
        <button type="submit" disabled>Send</button>
        <button type="button" id="cancel" disabled>Cancel</button>
        <button type="button" id="stop" disabled>Stop</button>
        <button type="button" id="delete" disabled>Delete</button>
      </div>
    </form>
  </section>
</main>
</body>
</html>
"""

SCRIPT: Final = """"use strict";
let csrf = null;
let current = null;
const $ = (id) => document.getElementById(id);

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf ?? "" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) {
    location.href = "/login";
    throw new Error("login required");
  }
  return response;
}

function line(kind, text) {
  const div = document.createElement("div");
  div.className = kind;
  div.textContent = text;
  $("log").append(div);
  $("log").scrollTop = $("log").scrollHeight;
  return div;
}

async function problem(response) {
  const body = await response.json().catch(() => ({}));
  return typeof body.error === "string" ? body.error : JSON.stringify(body.error ?? response.statusText);
}

async function refresh() {
  const sessions = await (await api("GET", "/api/sessions")).json();
  const list = $("sessions");
  list.replaceChildren();
  for (const session of sessions) {
    const item = document.createElement("li");
    item.textContent = `${session.name} · ${session.kind} · ${session.running ? "running" : "stopped"}`;
    item.classList.toggle("current", session.name === current);
    item.addEventListener("click", () => select(session.name));
    list.append(item);
  }
}

function select(name) {
  current = name;
  $("title").textContent = name ?? "No session selected";
  $("log").replaceChildren();
  for (const element of document.querySelectorAll("#prompt textarea, #prompt button")) {
    element.disabled = name === null;
  }
  refresh();
}

async function send(text) {
  line("user", text);
  const response = await api("POST", `/api/sessions/${encodeURIComponent(current)}/prompt`, { text });
  if (!response.ok) {
    line("error", await problem(response));
    return;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let answer = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\\n\\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const kind = /^event: (.*)$/m.exec(block)?.[1] ?? "message";
      const data = JSON.parse(/^data: (.*)$/m.exec(block)?.[1] ?? "null");
      if (kind === "error") {
        line("error", data.error);
        continue;
      }
      switch (data.type) {
        case "text":
          answer ??= line("assistant", "");
          answer.textContent += data.text;
          break;
        case "thought":
          line("thought", data.text);
          break;
        case "tool_call":
          line("tool", `${data.title ?? data.tool_call_id} ${data.status ?? ""}`);
          answer = null;
          break;
        case "output":
          if (data.output !== null && typeof data.output !== "string") {
            line("assistant", JSON.stringify(data.output, null, 2));
          }
          line("meta", `[${data.stop_reason}]`);
          break;
      }
    }
  }
  refresh();
}

function showKindFields() {
  const kind = $("create").elements.kind.value;
  for (const label of document.querySelectorAll("#create [data-kind]")) {
    label.hidden = label.dataset.kind !== kind;
  }
}

function words(text) {
  return text.split(/\\s+/).filter((word) => word !== "");
}

async function create(event) {
  event.preventDefault();
  const form = event.target.elements;
  const kind = form.kind.value;
  const spec = { kind, cwd: form.cwd.value };
  if (kind === "acp") spec.command = words(form.command.value);
  if (kind === "pydantic-ai") spec.target = form.target.value;
  if (kind === "claude-tty") {
    spec.args = words(form.args.value);
    spec.trust_cwd = form.trust.checked;
  }
  $("create-error").textContent = "";
  const response = await api("POST", "/api/sessions", { name: form.name.value, spec });
  if (!response.ok) {
    $("create-error").textContent = await problem(response);
    return;
  }
  select(form.name.value);
}

async function action(verb, method = "POST") {
  const path = `/api/sessions/${encodeURIComponent(current)}` + (verb ? `/${verb}` : "");
  const response = await api(method, path);
  if (!response.ok) line("error", await problem(response));
  if (method === "DELETE" && response.ok) select(null);
  else refresh();
}

async function init() {
  const me = await (await fetch("/api/me")).json();
  csrf = me.csrf;
  $("who").textContent = me.email ?? "";
  $("logout").addEventListener("click", async () => {
    await api("POST", "/logout");
    location.href = "/";
  });
  $("create").elements.kind.addEventListener("change", showKindFields);
  $("create").addEventListener("submit", create);
  const prompt = $("prompt");
  prompt.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = prompt.elements.text.value.trim();
    if (text === "" || current === null) return;
    prompt.elements.text.value = "";
    send(text);
  });
  prompt.elements.text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) prompt.requestSubmit();
  });
  $("cancel").addEventListener("click", () => action("cancel"));
  $("stop").addEventListener("click", () => action("stop"));
  $("delete").addEventListener("click", () => action("", "DELETE"));
  showKindFields();
  select(null);
}

init();
"""

STYLE: Final = """:root { color-scheme: light dark; --muted: #888; --line: #8884; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 system-ui, sans-serif; }
header { display: flex; gap: 1rem; align-items: center; padding: .6rem 1rem; border-bottom: 1px solid var(--line); }
header #who { margin-left: auto; color: var(--muted); }
main { display: grid; grid-template-columns: minmax(16rem, 22rem) 1fr; height: calc(100vh - 3rem); }
aside { border-right: 1px solid var(--line); padding: 1rem; overflow: auto; }
section { display: flex; flex-direction: column; padding: 1rem; min-width: 0; }
h2 { font-size: 1rem; margin: 0 0 .5rem; }
ul { list-style: none; padding: 0; margin: 0 0 1.5rem; }
li { padding: .35rem .5rem; border-radius: .3rem; cursor: pointer; }
li:hover, li.current { background: var(--line); }
label { display: block; margin: .4rem 0; }
label input:not([type=checkbox]), label select, textarea { display: block; width: 100%; font: inherit; padding: .3rem; }
label.check { display: flex; gap: .4rem; align-items: center; }
#log { flex: 1; overflow: auto; border: 1px solid var(--line); border-radius: .3rem; padding: .5rem; margin-bottom: .5rem; }
#log div { white-space: pre-wrap; margin: .25rem 0; }
.user { font-weight: 600; }
.thought, .meta, .tool { color: var(--muted); font-size: .9em; }
.tool { font-family: ui-monospace, monospace; }
.error { color: #d33; }
.buttons { display: flex; gap: .5rem; margin-top: .5rem; }
@media (max-width: 700px) { main { grid-template-columns: 1fr; height: auto; } aside { border-right: 0; } }
"""
