# aid

AI server for persistent, cooperating agent sessions, with separate execution engines and extension hosts.
Runs external ACP agents (`claude-agent-acp`, `opencode acp`, …), native agents (aid's own loop), and Claude Code
through terminal and SDK backends, behind an async Python API, CLI and web UI.

## Architecture stance

Fixed: a plugin architecture, async (anyio). Everything else is a current shape, not a decision to defend.

- aid is agent glue. The daemon does as little as possible: sessions, routing, delivery, authorization. Behaviour
  that can live in an agent, a backend or a plugin goes there. Before adding to the daemon, ask what could own it.
- What can reasonably be a plugin is one, bundled and on by default where it matters (skills, chat bridges).
- Essential conventions (Agent Skills, AGENTS.md, MCP) follow the common baseline other harnesses share; go beyond
  it, never around it.
- Re-evaluate the architecture as work lands; each layer built on a weak shape makes it costlier to change. A better
  way found is a way taken: change the code, then this file.
- Architectural suggestions go to User before building them: what, why, cost, recommendation.

## Working method — read first

The goal is reliable agent work, not merely code that passes a happy-path test. Use this sequence:

1. **Define the observable outcome.** State what should happen and what must remain unchanged. A request to
   stop loops does not authorize an arbitrary turn cap; a request for plugins does not imply one universal ABC.
2. **Trace the actual path before editing.** Follow caller → protocol → daemon → worker → backend → stored
   state → response. Read the relevant tests too. A helper's docstring is not proof that callers use it correctly.
3. **Identify the owner and invariant.** Who owns the state, who may mutate it, how long does it live, and what
   happens on cancellation/restart? Prefer one owner and explicit transitions over two copies kept in sync.
4. **Reproduce with the smallest real path.** Use deterministic local models/fake agents, not paid API calls.
   Check the installed dependency version and runtime behaviour when a default or API is involved. Separate
   confirmed failures, hypotheses and recommendations in updates; do not present guesses as findings.
5. **Fix the cause at its owning layer.** Keep decisions pure where practical; keep IO and lifecycle effects
   explicit. Make small coherent changes. Do not rearrange directories, add abstractions or tune thresholds
   merely to make the code look organized.
6. **Prove the lifecycle, not just the return value.** Add a regression that fails before the fix. Check success,
   failure, cancellation, resume/restart and stale/duplicate input where relevant. Verify persisted state, usage,
   attribution and actual effects. A compaction test outside a turn does not prove the compact tool works inside one.
7. **Validate and close the change.** Run targeted tests, formatting/lint/types, then the wider affected suite.
   Regenerate schema/types when models change. Report what changed, the checks run and remaining limitations.
   Never claim a suite passed if only a subset ran, and never weaken a test just to obtain green output.

If an experiment fails, preserve the evidence and investigate the boundary it exposed. Do not patch blindly
around a failed check. Change a test's expectation only when the intended contract changed, and explain why.

## Decision guide

| Tempting shortcut | Better decision |
| --- | --- |
| Increase a request limit to stop errors | First check whether the user requested a limit, and whether `None` inherits an engine default. |
| Cancel whenever messages look similar | Raise a suspicion with evidence; let the supervisor judge progress. Polling and repetitive work can be legitimate. |
| Send a nudge to an indefinitely running turn | Check delivery timing. A queued message cannot redirect that turn; interruption needs an explicit, turn-scoped operation. |
| Retry a failed/interrupted tool automatically | Its effects may already have happened. Preserve the result/state and inspect before replaying a side effect. |
| Compare only the newest history entries | Define the signal's unit. Ten tool events are not ten answers; load/cache the relevant completed units. |
| Catch every error and continue | Distinguish normal outcomes from faults. Preserve work and usage, propagate cancellation, and log faults with their original cause. |
| Reconstruct every plugin during reconciliation | Select eligible classes first; the lifecycle host constructs only instances it needs to start. |
| Put critical control on PUB | Use addressed request/reply or the reliable inbox. PUB is observation and may drop events. |

## Ownership and extension boundaries

- **Daemon:** durable session state, routing, supervision links and authorization of daemon operations.
- **Worker:** one session's execution lifecycle and backend capabilities. Recheck turn identity here before an
  interruption: a daemon check can become stale while a request travels.
- **Backend:** engine adaptation, engine conversation history, cancellation and usage. Do not spread engine
  assumptions across the daemon, CLI and UI. Different engines expose different capabilities; say so explicitly.
- **Launcher:** process/environment setup and placement, not the agent's conversation or supervision policy.
- **Extensions:** use the appropriate ABC/contract for their host (`Plugin`, `AgentDefinition`,
  `InstructionsProvider`, backend capabilities), not a catch-all object. Specify construction, context,
  shutdown, restart and failure ownership before adding a new extension type. Discovery is not instantiation.

The daemon transcript and the engine's resumable conversation are different stores; preserving one does not
prove the other survived. A status flag is not enforcement, and a grant checked on one socket does not secure
another socket. Missing telemetry is unknown, not proof that nothing changed.

The layout below describes the current implementation, not a mandate to preserve every historical design.
When prose and code disagree, reproduce the behaviour, fix the contract and update both. Keep this file focused
on reusable decisions and current boundaries; put long investigations and dated measurements in `docs/`.

## Layout

- `src/aid/spec.py` — `AcpSpec` / `NativeSpec` / `ClaudeTtySpec` / `ClaudeSdkSpec`: what a session runs.
- `src/aid/protocol.py` — pydantic messages on the zmq sockets; `AidError`.
- `src/aid/daemon.py` — ROUTER for clients (`control.sock`), ROUTER for workers (`workers.sock`), session state.
- `src/aid/launcher.py` — `Launcher` protocol; `ForkserverLauncher`.
- `src/aid/worker.py` — worker entry `main(endpoint, server_key, public_key, secret_key, spec_json, state_dir)`;
  DEALER to the daemon.
- `src/aid/zap.py` — every worker connection is CURVE. The daemon issues a keypair per launch; its ZAP handler
  names the session as the connection's User-Id, and the daemon routes by that, never by a worker's routing id.
  A `wss://` worker sets WSS_HOSTNAME: without it libzmq accepts a trusted certificate for any name. Domains:
  workers, plugins, plugin events (wants `read`), plugin private events (`read` + `private`).
- `src/aid/plugins.py` — plugins: a registered CURVE public key + grants (`read`, `prompt`, `message`,
  `permissions`, `manage`, `supervise`, `private`; `NEEDS` maps op → grant, unlisted ops no plugin may send). Sockets
  `plugins.sock` (ROUTER), `plugin-events.sock` and `plugin-private-events.sock` (PUB), all CURVE; `_plugin_loop`
  checks each request against the grants as they are now → `forbidden`. The daemon stamps what a plugin cannot
  claim: a message's sender is `plugin:<name>`, `AnswerPermission.plugin` is the connection's (None on control) →
  `PermissionDecision.by=plugin` + name.
  Calls: a plugin overriding `Plugin.call(caller, method, args)` serves them (`plugin_host.run` → `serve_calls`,
  `ServeCalls` renewed every `SERVE_RENEW`, no grant needed); `CallPlugin` from a session tool or control gets
  `caller` stamped (agent connection → its session; control → as sent, None = operator), goes out as an
  `Invocation` under a daemon id, answered by `CallResult` from that plugin only, bounded by `PLUGIN_CALL_TIMEOUT`.
  Plugins themselves may not send `CallPlugin`. The plugin authorizes by `caller`. `supervise` (trusted: watching
  over sessions) adds `spec`, `attention`, `publish_supervision`, the `message` hook (`Plugin.on_message`, asked
  before each `SendMessage` is accepted; refusal = its code, silence/absence = pass), a kept `SendMessage.sender`
  and `AnswerPermission.supervisor`. plugin.toml `autostart = true`: hosted with declared grants, registered in
  memory each start; removal keeps it off for the daemon's lifetime.
  Registry ops (`add_plugin`, `plugins`, `remove_plugin`) only on control.sock, never in `PageRequest`. State:
  `state_dir/plugins/<name>/plugin.json` + `secret` (0600, `aid plugin add` makes it); the daemon writes its public
  key to `runtime_dir/server.key` at start (new each start). Client: `aid.connect(plugin=NAME)`,
  `register_plugin`. Grants bind only what arrives on the plugin sockets: a same-user local plugin can open
  control.sock, which checks nothing, until confined (Lillecarl/aid#4). The events PUB cannot drop a subscriber: a
  plugin losing `read` hears events until it reconnects.
- `src/aid/discovery.py` — the plugins path (`AID_PLUGINS_PATH`, `~/.config/aid/plugins`, `AID_PLUGINS_DIST`):
  a plugin is a directory named as an import name, with `__init__.py` and/or optional `plugin.toml` (`name` = dir,
  `version`, `description`, `requires`). `resolve` reads manifests only (first tier wins a name; missing requirement
  or cycle drops dependents); `discover` imports packages in dependency order, under their own names, never via
  `sys.path`, and sorts by interface: `AgentDefinition`, `InstructionsProvider`, `ModelProvider` (by the prefix it
  serves, `Catalog.model_providers`), `@aid.mcptool`, `Plugin` (≤1 per
  plugin; `plugin_name()` = top-level package); abstract and `_`-named classes are bases, skipped. Data-only plugins (`skills/`) need no `__init__.py`. Workers and
  `aid.mcp_server` import everything; the daemon never imports a plugin without a task (below).
- `src/aid/catalog.py` — `python -m aid.catalog [--path DIR]`, the catalog as JSON (`AgentCatalog`, plugins
  included); `listing()` runs it as a subprocess for the daemon. `session_catalog()` is what a session gets.
- `src/aid/plugin_host.py` — in-process plugins: `hosted()` lists the daemon's `plugin_dirs` (None means the standard
  tiers) in a subprocess and imports only plugins with a task, plus what they require. Registered ones run as
  supervised tasks speaking the same protocol over
  `inproc://aid-plugins` / `inproc://aid-plugin-events`, through the same loop (identity is the DEALER name the
  daemon set). `_reconcile_plugins` runs at startup and after every registry change: unregistered, under-granted
  and declined code stays outside with a log; removal stops the task at once. Failures restart after
  `RESTART_DELAY`, give up after `MAX_FAILURES` in `FAIL_WINDOW`. Tests pass `plugin_dirs=[]`; the daemon's own
  environment is what hosted plugins read.
- `src/aid/readonly.py` — pure rules: does a call only read? shlex with punctuation: pipes, `;`, `&&`, `||`;
  `NAME=literal` statements expand later `$NAME` (never before a program: its environment; never a `RUNS_CODE`
  name); output only to /dev/null or another fd. Each program: `READERS` (any args: nothing in them may write or
  run a program), `PROGRAMS` rules (git/jj/nix subcommands, sed without -i/w/e/-f, `_without` options that write
  or run programs: sort -o, fd -x, rg --pre, jj --config, nix unsafe flags, …), `--help`/`--version` alone;
  credential-shaped paths (`SENSITIVE`) never. Anything unclear is False. A parser, not enforcement: confinement
  is parked in Lillecarl/aid#44.
- `src/aid/guard.py` — `aid guard`: plugin (read, permissions) that allows what `aid.readonly` calls a read, and
  leaves the rest for a person. Seed for aid#3.
- `plugins/discord/` — the Discord plugin (read, message): `gateway.py` (the gateway over anyio streams, no
  Discord library — wsproto frames, anyio owns the sockets; resume policy follows the close-code table),
  `rest.py` (channels, messages, interaction answers; 429s honored and retried, the rest raises at once),
  `channels.py` (one channel per session under one category, id-mapped on disk, reconciled at startup and
  on list changes), `serve.py` (bare text in session channels as messages, `!`-commands everywhere,
  taps as new messages or in-place rewrites, settled notices to session channels). Custom ids hold
  `action:session` directly (100 bytes); Discord messages clip at 1900/2000.
- `plugins/skills/` — Agent Skills, bundled: an `InstructionsProvider` lists each skill's name and description, the
  `skill` tool loads a body or file. Spec-checked frontmatter (YAML); project `.aid|.agents|.claude/skills` up to the
  git root, then `AID_SKILLS_PATH`, then the global dirs, then each resolved plugin's `skills/`. Lists nothing when
  `InstructionsContext.tools` (the offered names after the ceiling; None unknown) lacks `skill`.
- `plugins/forks/` — short-lived forks, bundled, `autostart` (read, prompt, message, manage, supervise). Tool
  `fork(instructions)`: `Coding.snapshot` writes the conversation inside the call (history.json lacks the running
  turn) to `state_dir/forks/`; the plugin creates `<parent[:52]>.fork-<6hex>` from the parent's `NativeSpec` with
  `seed_history` (a seeded session skips instruction composing: same cache prefix) and `parent`, prompts it, sends
  the answer to the parent as the fork with its turn `Usage` (`agent="fork <name>"`; `SendMessage.usage`, kept
  only from control and `supervise` plugins, recorded after the message), deletes it. A session with a `parent`
  can't fork. At plugin start, leftover forks answer from their recorded Output/TurnError (else "lost") and are
  deleted. Spec `parent` (also on `SessionInfo`/`SessionStatus`): chat plugins skip such sessions (no Discord
  channel, no settled-turn notice, not in session lists). The parent must exist at create (`unknown_parent`);
  `DeleteSession` takes its descendants (`_delete_orphans`), and a start deletes sessions whose parent's dir is
  gone (a parent whose spec fails to load keeps its children).
- `plugins/spawn/` — persistent children, bundled, `autostart` (read, message, manage, supervise). Tool
  `spawn(label, instructions, tools, model?, task?)` → `<parent[:47]>.<label>`, an instructions spec from
  `aid.spawn.child_spec`: model = parent's or one `NativeSpec.spawn_models` (person-set, inherited) fullmatches;
  `tools` = the parent's offered names (`SessionStatus.tools`, post-ceiling) the patterns match, stored
  `re.escape`d, a pattern matching none refused; modes, sandbox, hooks, supervisor, env inherited as is;
  `MAX_DEPTH` 3 ancestors, `MAX_CHILDREN` 8 live per parent. `task` goes to the child as a message from the parent.
  Each child's `Output`/`TurnError` goes to the parent as a message from the child with its `Usage`
  (`agent="spawned <name>"`); `children.json` in the plugin dir keeps each child's parent and last relayed seq,
  saved after the send (a crash between repeats one). A relay checks presence after `PRESENCE_CHECK` quiet.
  `dismiss(session)`: the parent only.
- `plugins/findings/` — reported issues, bundled, `autostart` (supervise: `spec`). Tool `report_finding` writes one
  row of `findings.db` (plugin dir, 0600: findings may quote secrets); `repo` is the caller's resolved `cwd`, never
  an argument; path confined to it (absolute allowed inside), lines within the file, `evidence` must occur in it
  (whitespace squeezed) else refused. Duplicates kept, grouped by `dedup`. `verdicts` for review passes. `query`
  (`aid findings query`) operator only. Plan: `docs/audit-campaigns.md`.
- `plugins/campaign/` — audit campaigns, bundled, `autostart` (read, prompt, manage, private; operator calls only:
  `start`/`status`/`pause`/`resume`, `aid campaign`). `CampaignConfig` (name, root, repos or every `.git` dir, model +
  settings, `hosts` → every session private, env, python_path, `Limits`, concurrency, attempts, supervise, backoff).
  Root session `<name>`: agent `supervisor` on the campaign model, ceiling `SUPERVISOR_TOOLS` (no man/tldr: they run
  programs). Per repository `<name>.<slug>` (`session_names`): instructions spec, ceiling `AUDITOR_TOOLS`,
  `fresh_turns`, `instruction_providers=[]` (nothing from the audited tree composes), parent root, supervisor root if
  `supervise`. `units.py` (pure): `git ls-files`, skip excluded/links/oversized/binary, chunk by file and byte caps;
  `uncovered`: a file is read when completed `read` calls' `[offset, offset+limit)` ranges cover every line, events
  merged per (turn, tool_call_id) (models renumber calls each turn); `credited`: only turns ending in `Output` with
  `end_turn` earn their reads (failed, cut off at max_tokens or never ended earn nothing: one rule, no cases). Brief
  turn, then a turn per unit; unread files prompted again (brief included: turns are fresh) until `attempts`; failed
  turns back off; a busy refusal costs no attempt. `store.py` (`campaign.db`, migrations): units keep `covered` and
  the running attempt's `start_seq`; a resume credits what that attempt recorded. Calls wait for `_ready`.
  Splitting (`subsystems.py`, pure): a briefed repository with more than `split_units` units, none started, gets a map
  turn (`overview` capped at `OVERVIEW_LINES`); `parse_map` takes 2–12 subsystems, a bare dir or one-level glob
  covers its tree, first match wins, the rest is `OTHER`; more than `OTHER_SHARE` unmatched is refused with a sample
  and asked again, no map after `attempts` leaves it whole. Units re-plan per subsystem (`store.split`, rows in
  `subsystems`); each subsystem runs in its own child session (`subsystem_session`: campaign, slug, path hash,
  index; never model text), parent the repository session, `subsystems_at_once` per repository, prompts carrying
  the brief and `subsystem_context`. Unit answers are stored; an integration turn on the repository session reads
  them (clipped) for issues across subsystems. States: briefed → mapped → integrated → done.
- `src/aid/slots.py` — model slots: `aid daemon --model-slots HOST=N` caps requests in flight to a model host across
  every session (a self-hosted server's best batch load). The table reaches a worker at launch (`WorkerArgs.model_slots`
  → `AID_MODEL_SLOTS`; none outside a worker); `open_native` wraps a limited host's model in `Slotted`, calling
  `host()` only when the table names hosts: each request (stream, request, compaction)
  holds a slot (`AcquireModelSlot`, FIFO) and releases it in a shielded `finally`, a cancelled wait too. The daemon's
  `ModelSlots` frees a session's slots when its worker exits (waits first, else its own next waiter inherits) and
  refuses a late acquire whose release overtook it (separate connections). Pricing asks the transport
  (`Transport.cache_reads_in_prompt`), never its class: wrappers hide classes.
- `src/aid/backends/` — `acp.py`, `native.py`, `claude_tty.py`, `claude_sdk.py`; each implements `base.Backend`.
- `src/aid/model.py` — the native backend's transports: `AnthropicModel` (Messages API, native blocks,
  `extra_headers` carry betas, automatic caching on), `OpenAIChatModel` (Chat Completions) and `GoogleModel`
  (Gemini API over google-genai, imported on first use: +0.5 s per worker otherwise; aid's own httpx client, AFC
  off; `thought_signature` kept base64 on the Text/ToolCall part that carried it, Gemini's call id as `google_id`
  beside aid's; prompt counts include cache reads; `_UnionGenericAlias` warning filtered in pytest,
  googleapis/python-genai#1640), each taking a
  `Provider` (base URL, key variable, default request settings, headers). A provider's quirks subclass
  `OpenAIChatModel` (`request_settings`, `reasoning`, `usage`, `wire`), never branch in it. `OPENAI_COMPATIBLE`:
  prefixes models.dev gives no URL for, and local servers. `model_settings` (JSON): `ClientSettings`
  (`api_key`, `api_base`/`base_url`, `max_retries`; mistyped → `bad_model_settings`) build the client, the rest goes
  into each request, cast once to the SDK's `…CreateParamsStreaming` (validating it would drop
  `reasoning_content`). `model_for(name)`: the table alone, for direct use.
- `src/aid/providers.py` — `resolve(name, catalog.model_providers)`, in `open_native`: an `aid.ModelProvider`
  plugin claiming the prefix (async `transport(model)`), else `anthropic/`, else `google/` (`GEMINI_API_KEY`;
  models.dev's id, so prices and lists need no mapping; `gemini/` stays the Chat Completions route, unpriced;
  measured 2026-10-24 on `gemini-3.1-flash-lite`, the cheapest a new key may use (2.5-flash-lite: 404 "no longer
  available to new users", still in models.dev's list): tool loop with signed calls replayed, thinking tokens
  counted, no thought text even with `include_thoughts`), else
  `OPENAI_COMPATIBLE`, else a
  models.dev `@ai-sdk/openai-compatible` provider by its `api` and first `env`. `ModelProvider.catalog_id` names a
  plugin's models.dev id where its prefix differs: that id then resolves to an error naming the plugin's prefix, is
  left out of listings, and prices and backfills the plugin's models (`catalog_model`, `price_as`). Providers are
  stateless and built per use in workers; the transport is the backend's to close and price.
- `src/aid/models_dev.py` — models.dev's catalog (`~/.cache/aid/models.dev.json`, refetched daily, stale kept on
  failure; `AID_MODEL_PRICES` = URL, file or `off`, tests set `off`), typed per provider and model. `prices.py`
  prices from it: the native backend prices each request of a model it resolved from a name (context tiers
  included) into `Usage.cost`, a per-turn amount that summaries add; `session_cost` stays the agent's own total.
- `src/aid/model_lists.py` — `ListModels` (`aid models [PROVIDER]`, page, plugin `read`, agents) answered by
  `python -m aid.model_lists` in a subprocess: providers (plugin, builtin, catalog; whether the key variable is set
  in the daemon's environment) and one provider's models: `ModelProvider.fetch_models` cached a day
  (`~/.cache/aid/models/<provider>.json`, stale kept on failure with `problem`), backfilled from models.dev, else
  models.dev's list. `AID_MODEL_LISTS=off` fetches nothing (tests; they also isolate `XDG_CACHE_HOME`).
- `plugins/deepseek/`, `plugins/openrouter/`, `plugins/opencode/` — bundled model providers. DeepSeek: output-limit
  default, `prompt_cache_hit_tokens`, its keyed `/models`. OpenRouter: `usage.include`, its `reasoning` field, no
  reasoning replay (needs `reasoning_details`), public `/models` priced per token. OpenCode: two gateways, two prefixes and
  keys: Zen `opencode-zen/` (models.dev's `opencode`; `OPENCODE_API_KEY`, public `/zen/v1/models`) and Go
  `opencode-go/` (`OPENCODE_GO_API_KEY`, public `/zen/go/v1/models`); per-model protocol from models.dev's `provider.npm`/`api` (Anthropic Messages, Chat
  Completions; Responses and Gemini models listed `unsupported`, and Zen's free models: Zen serves them to OpenCode
  alone, 403 FreeTierError; aid does not pose as OpenCode). Both send `User-Agent: aid/<version>` and
  `x-opencode-session` (uuid5 of the aid session, `ModelProvider.session`), as Go asks of other agents; Go answers
  400 MissingSessionID without it. A list's `unsupported` marks are fetched with it: a plugin change shows after
  the day's cache. Measured 2026-10-24: Zen answers over both routes (Anthropic Messages with `x-api-key`, Chat
  Completions), priced through models.dev's `opencode`; Go and OpenRouter inference unmeasured (no working key). A plugin
  named like a bundled one in an earlier tier shadows it: the example agent is `examples/deepseek_agent`.
- `NativeSpec.model` (`provider/model`) runs the agent on another model (`session_agent`); the agent's
  `model_settings` follow only within one provider, `NativeSpec.model_settings` go over them. `aid new-py --model`,
  the create form's picker.
- `src/aid/mcp.py` — a spec's `mcp_servers` as ACP `session/new` params and as Claude Code `--mcp-config` JSON,
  plus the built-in `aid` server (`session_servers`). claude-agent-acp restarts its query when `session/load` gets
  servers other than `session/new` did.
- `src/aid/tools.py` — `@aid.mcptool`, a marker attribute; `discovery.discover` collects the marked functions.
  Keep it free of `mcp` imports: every worker and the daemon import it.
- `src/aid/mcp_server.py` — `python -m aid.mcp_server`, the tools as a stdio MCP server. The agent starts it; it
  imports user code, so never the daemon. Native sessions get the same functions in-process as a toolset.
  For interactive Claude it is also the session's channel (research preview): it long-polls the daemon for the
  session's messages and pushes `notifications/claude/channel`.
- `src/aid/builtin_tools.py` — aid's own tools, in every session: `send_message`, `list_sessions`.
- Request ids route replies across all clients: the daemon refuses an id in flight (`duplicate_id`); relays and
  plugins never pass a caller's id through unchecked.
- Messages: `SendMessage` → recipient's history + inbox. `Delivery`: `interrupt` (default) or `wait_for_turn`.
  A native turn takes waiting interrupts before each model request after its first (`ReceiveMessages`, wait 0;
  `ARRIVED`, skipped by `open_turn`); the rest, and ACP and claude-sdk sessions, get a daemon-started wake turn after
  the running turn (claude -p queues stdin lines until its turn ends: measured 2.1.309); interactive Claude gets a
  channel event, `wait_for_turn` held while `working`. A busy session's `ReceiveMessages` claims interrupts only;
  the inbox checkpoint keeps `delivered` seqs past `through`, so out-of-order claims survive a restart. A wake turn has no client (`_Route.client` None).
  `aid.messaging.render_message`: a person's and background reports plain; another session's or a plugin's in
  `<aid-message from kind>` with an untrusted note (never `<system-reminder>`, which Claude reads as its harness).
- Directory moves: `ChangeDirectory` (control, page `/cd`, `aid cd`, plugin `manage`) saves `spec.cwd`, restarts a
  running worker there, records `Lifecycle(moved)`; refused while `busy`. Agents only `RequestDirectory` their own
  session (`change_directory` tool); a person answers (`AnswerDirectory`: page card, `aid cd --approve/--decline`,
  plugin `permissions`) under every permission mode, allow-all included, until the permissions rework. Approved
  mid-turn → applied at turn end (or when claude-tty stops `working`), then aid tells the agent (`aid:` sender,
  plain). Continuity measured: Claude Code 2.1.310 resumes across directories (the transcript stays under the old
  project; `find_transcript` globs every project); claude-agent-acp 0.88.0 `session/load`s the same agent session
  in the new cwd and knows it moved; native history is aid's; native turns name `cwd` in metadata. opencode 1.19.4
  keeps the conversation but its tools stay in the session's first directory (unsupported; not worked around).
- Tags: `spec.tags` (`TagKey`/`TagValue`), labels aid never acts on. `SetTags` (control, page, `aid tag`, plugin
  `manage`; never agents) merges, None removes; `SessionInfo.tags`; `aid list --tag`, the page's list filter.
- `src/aid/transcript.py` — Claude Code transcript entries → aid events.
- Recording: each turn ends with `Usage` (tokens, models, context, the agent's session cost) before `Output`; each
  worker start records `Started` (agent session id, agent, model) under a turn id of its own. What a turn's tokens
  cover differs per agent: `Usage`'s docstring says how, measured. The web UI must render every `HistoryItem` type:
  an unknown one breaks Chat.
- claude-tty hooks: `aid/hook.py` runs as a file (no aid import: 0.04 s against 0.2 s per event), sends a `hook`
  request, prints the answer, and fails open (exit 0, no output). The daemon routes it to the worker
  (`HookBackend.hook`); worker state goes back as `Activity` (working, attention). Events: `HOOK_EVENTS`.
- `usertee` (`aid/usertee.py`, a file like hook.py, zmq only): agents pipe output through it so a person sees what
  `tail` hides. `launcher.install_usertee` writes a shim to `runtime_dir/bin`, first on every worker's PATH (not the
  venv's bin: its `python` would shadow the project's). Frames go to the worker's hook socket (`usertee_text`), on
  to the daemon as `Live`, out on the live topic as `LiveChunk`; the daemon keeps `LIVE_LIMIT` of the current turn
  for `GetLive`. Never recorded; fails open (cat). Agents learn of it from `instructions.USERTEE`;
  a native session only when offered an `execute`-kind tool (`environment(runs_programs=)`, from `session_tools`).
- claude-tty follows its transcript for the worker's life (`FollowingBackend.follow`): a turn aid's `prompt()` waits
  for is aid's; any other (typed into the pane, woken by a channel event) goes to the daemon as `Observed` entries
  under a turn id of its own.
- `src/aid/agents.py` — `aid.AgentDefinition` and `aid.InstructionsProvider`, interfaces plugins implement. Never
  import agent plugins in the daemon itself: it asks `aid.catalog` in a subprocess. `AgentDefinition.
  instruction_providers` picks the providers composed for its sessions (the supervisor: none); a spec's overrides.
  `fresh_turns`: each native turn starts from the instructions alone (engine history cut back before the prompt;
  the transcript keeps every turn), one cached prefix for all.
- `src/aid/env.py` — agent environments, minus the markers of the Claude session that started us.
- `src/aid/client.py` — public API: `aid.connect()`, `Client`, `Session.run/stream`.
- `src/aid/client.py:daemon_call` — what hosted loops await daemon calls through: silence past
  `DAEMON_CALL_TIMEOUT` exits loudly, so the supervisor restarts into a fresh baseline.
- `src/aid/settle.py` — settled turn ends, platform-agnostic: `Tracker` fed by turn and supervision events,
  `settled_turns` calling back per settled episode. Both chat plugins share it.
- `plugins/supervision/` — supervision, bundled, `autostart` (read, message, prompt, permissions, supervise). Follows each
  supervised session's history (gap-filled), from the last entry before it started: turn ends (`Output`,
  `TurnError`, once per turn) message the supervisor as the supervised session with tail/goals/loop score;
  `loops.py` exact tool-call/cycle suspicions (never cancels); permission routes; stuck episode → attention;
  nudge budget via the `message` hook. Goals in its plugin dir (`goals/<session>.json`, legacy session-dir
  `goals.json` read). `aid.supervisor_tools` (core) call it via `CallPlugin`; it authorizes by `caller`.
  `interrupt` = `Cancel(turn=…)`, the worker rechecks the turn. Daemon keeps only `spec.supervisor` links and
  their validation (`supervision_cycle`); `SetSupervisor` (control, page, `aid supervise`, plugin `manage`; never
  agents) changes one later, and the plugin refollows that session from its newest entry (`SessionInfo.supervisor`).
  State is in memory: a restart starts a fresh episode.
  Permission routes (`ROUTES`): modes `supervisor` (supervisor → person → supervisor) and `ask_supervisor`
  (person → supervisor → person); engines treat both as `ask` (`PermissionMode.waits`) and know no routing; a
  request carries its `mode` (claude-tty's none: the plugin reads the spec). A holder hands on by
  `forward_permission` or silence (`SUPERVISOR_WINDOW`; a person: `supervisor_permission_timeout` / absent); the
  last decides; a handover to a person raises attention, the decision clears it; the worker's
  `permission_timeout` bounds the route (`_worker_bound`). The supervisor answers only routed requests
  (`person_only`). Stored `supervisor_permissions` (gone): the spec's before-validator maps listed `ask` tools to
  `ask_supervisor`, and the daemon writes each spec file back as the model reads it.
  A routed request `aid.readonly` calls a plain read (and offering `allow_once`) gets the single holder `rules`:
  the plugin allows it at once, no model asked (`PermissionDecision.by` plugin `supervision`). Else, with
  `spec.supervisor_classifier` (agent name or `module:attr`; default the bundled `classifier`, `classifier.py`,
  `fresh_turns`, structured `ClassifierVerdict`), allow/reject options and a tool outside `TASK_JUDGED` (`remove`:
  a plan's soundness depends on the task; measured, the classifier allowed one it was told to escalate), holder
  `classifier` comes first: the
  supervisor's session `<supervisor[:52]>.classifier` (parent = supervisor, made on use, remade for another
  classifier, deleted with the supervisor; one request at a time) gets one turn per request. allow/deny answer as
  the plugin (a deny's reason reaches the agent as `aid's supervision check said:`); escalate forwards with the
  reason as note; failure or `CLASSIFIER_WINDOW` hands on silently. Tests' `py_spec` sets it None.
  Rulings (`rulings.py`, `plugin_dir/rulings.db`): every routed request (`rulings`: session, turn, seq, mode, input,
  remove's `plan`, end, allowed, decided_by, said) and its holders (`ruling_steps`: forwarded with note, silent,
  decided, gone); a failed write is logged, never fatal. `query` (`aid.sql.ask`): one read-only statement over
  `rulings` and, with a session, its `removals.db`; a supervisor names only its own sessions and must supervise one,
  and reads `rulings`/`ruling_steps` as temp tables of its own sessions' rows, copied before the question with the
  file detached (`aid.sql.Excerpt`; not views: SQLite flattens a view and names it as reader of a joined raw table,
  so an authorizer leaks); the operator (`aid supervision query`, JSON) any, unscoped.
  Bundled agent `supervisor` (`agent.py`, `deepseek/deepseek-flash`): supervisor tools plus `reference.py`'s
  read-only `man_page` (`man -P cat`, fixed argv, command-shaped names only) and `tldr_page`
  (`AID_TLDR_PAGES`, `nix/tldr-pages.nix`, pinned tldr-pages main); no shell, files or editor by design (User).
- `src/aid/mutes.py` — runtime mutes per chat plugin, on disk beside the registration.
- `src/aid/events.py` — the daemon's PUB socket (`Paths.events`): `[topic, JSON]` for the session list, each
  status, each history entry. The daemon compares state with what it last published after every request,
  worker reply and worker exit (`_publish_changes`); history publishes from `HistoryLog.on_append`. Readers use
  `Client.follow_sessions` / `Session.follow_status` / `Session.follow_history`: subscribe, then fetch the
  baseline, then follow; a jump in seq is fetched. `idle=` yields None on quiet spells (never cancel the
  generator's receive). One SUB socket per follower, closed with it.
- `src/aid/web/` — `aid web`: Starlette on hypercorn, OIDC login (`auth.py`), HTTP for what aid web answers itself
  (`app.py`: files, speech, pane).
- `src/aid/web/zws.py` — the daemon for the page, as ZWS 2.0 (flags byte + body per WebSocket message; bare
  `ZWS2.0`, no mechanism): `/api/zws/control` a DEALER per WebSocket, `/api/zws/events` a SUB (`\x01`/`\x00` + topic).
  Login + Origin gate both. Control admits only `PageRequest` (the allowlist; never `Hook`, `SendMessage`,
  `ReceiveMessages`, pane requests) and swaps in its own request ids: the daemon routes and names turns by id across
  clients. Page side: `web/src/lib/zws.ts` (socket, PING heartbeat, backoff) and `api.ts` (`call`, `follow*`:
  subscribe, then fetch the baseline, changes that beat it applied after; a dropped request fails, never resent).
- `src/aid/schema.py` — `python -m aid.schema`: JSON Schema of what the page reads (serialization mode: defaults
  required) and sends (validation mode). Committed as `web/src/lib/protocol.schema.json`; `npm run types` in web/
  generates `protocol.ts` from it, which `api.ts` re-exports. Never hand-edit either: `tests/test_schema.py` and the
  web build's `types:check` fail on drift. After changing a model: regenerate both.
- `web/` — the Svelte 5 UI (runes, TypeScript, Vite). `web/default.nix` builds it; `aid web` serves the result
  from `AID_WEB_ASSETS`, which the installed `aid` wrapper sets. Linked sessions (`links.ts`): supervisors and
  sessions under one leave the list (a toggle shows them) and show as tabs in Chat's head of each session they
  watch, badged from `SessionStatus` (cost, `context_used`); the picked tab drives every session tab. Each picked
  session's Chat stays mounted, so several `chat-input`s exist: browser tests select `:visible`.
- `src/aid/sql.py` — apsw (`nix/apsw.nix`, its own SQLite): `Database`, one owner's writes in a worker thread, WAL;
  `ask`, one statement over files attached `mode=ro` under a read-only authorizer (ATTACH/PRAGMA still work on ro
  files), capped in rows, cell size and time (progress handler). `QueryResult` in protocol.
- `src/aid/removals.py` — `remove`'s plans: paths, globs (`glob.translate`, hidden included), regexes (searched),
  excludes and filters (kind, age, git tracked/ignored from two `git ls-files`) → every entry that goes, before
  anyone is asked. A matched directory goes whole (rmtree at execution, whatever it then holds); one holding a kept
  entry stays and its other entries go one by one; patterns never match `.git`. Execution removes the plan's top
  rows only, rechecking kind and parents. Rows in the session dir's `removals.db` (apsw, the worker writes:
  `removal_plans` with the permission `request_id`, `removal_entries` with `top`/`within`/`tracked`/`outcome`,
  `removal_kept`); the request's input is a `RemovalPreview` naming the plan. `preview=True` plans without asking.
- `src/aid/coding.py` — `aid.coding_tools`, a toolset native agents opt into: list, read and `search` (`re` per
  line in the edit thread, no program run: skips `.git`, gitignored paths, binaries, >4 MiB files, long lines;
  stops at `limit` or `SEARCH_SECONDS`), pyedit edits through its
  library (`EditSession`, staged until `apply_edits`; no agent code runs in the worker), and `python` (pyrun
  `run_script` in a child, each command asked under `python`; file reads and writes outside the cwd under
  `python:read` / `python:write` via pyrun's audit-hook guard, `pyrun/src/pyrun/files.py`: async mode only, not
  `sync`/`xonsh`/`in-loop`; quiet for the run store and reads of interpreter paths, /proc, /dev, /sys).
  `NativeSpec.sandbox` runs what the tools start under bwrap (`src/aid/sandbox.py`, `Coding.confine`: the shell's
  bash, `background`, `python` scripts with their run store bound; root read-only, cwd and `writable` bound,
  private /tmp, optional `--unshare-net`); `in-loop` is refused; no working bwrap → the worker refuses to start.
  Unix sockets still connect (nix daemon, aid's sockets). `SetSandbox` (control, page, `aid sandbox`, plugin
  `manage`; never agents) checks bwrap in the daemon, saves the spec, records `Lifecycle(sandboxed)` and restarts a
  running worker, at once when idle, else at turn end (`restart_pending`, `_settle_restart`).
  Results over `SPILL_LIMIT`
  spill to content-addressed files under the session, sent as a stable head plus the path. `compact`
  summarizes the history into a digest through a direct model call. One `Coding` per session, set per
  turn in the `CODING` contextvar; tool errors go back to the model as `ModelRetry`. pyedit is built into the
  set from Lillecarl/pyedit (`nix/pyedit.nix`); its grammars come from `tree-sitter-grammars` (`grammarsByName`).
- `pyrun/` — async process library, a project of its own (own pyproject, `pyrun/default.nix`, tests) to be
  extractable; contract in `pyrun/README.md`. Kills by the PYRUN_SCOPE/PYRUN_ID env marks found in /proc (no
  subreaper, no process-wide state). Check: `cd pyrun && ruff check src tests && pyright && python -m pytest`;
  sandbox: `nix build --file . pyrun-tests`.
- `default.nix` — a pyproject.nix set from pyterm's builders (`mkPythonSet`, `mkProject`, its `overlay`);
  `aid/default.nix` — the aid project in it; `nix/` — NixOS and home-manager modules.

## Commands

```sh
nix develop --file . shell --command sh -c 'ruff format src tests plugins && ruff check src tests plugins && pyright'
nix develop --file . shell --command python -m pytest -p no:cacheprovider -n auto  # ~2 min; serial ~10
nix build --file . tests      # the same suite in the sandbox, against a real pymux
nix run --file . pyrefly -- check   # project mode: files and import roots come from pyrefly.toml
```

The dev shell puts `src/` ahead of the aid its venv carries. Dependencies go in `pyproject.toml`; the set
reads them. Test-only ones go in the `test` extra.

For a focused change, run its test module first, not the entire suite on every edit. Before finishing, expand
coverage to the affected daemon/client/backends. A generated schema change also needs `tests/test_schema.py`
and `npm run types:check` in `web/`. Run code generators in an initialized environment: outer-shell `>` can
truncate an input file before `nix develop` evaluates the repository. Stage generated output outside the repo
or perform generation inside the already initialized shell, then regenerate its dependent types.

UI: `cd web && npm run check && npm run dev` (proxies the API to `aid web` on `AID_WEB_BACKEND`, default
127.0.0.1:8080). After changing `package-lock.json`, update `npmDepsHash` in `web/default.nix`.

## Rules

- Architecture beats velocity. When the quick fix and the right fix diverge, build the right one: a shortcut
  taken once becomes permanent tax at the scale of every session on the daemon, and scale is the whole point.
  Precedents, all slower than the alternative: per-session worker commands ship behind `--allow-worker-command`
  because the command receives worker credentials; history filtering is store-full plus a deterministic send
  policy (monotonic sends keep the prefix cache) instead of mutating stored history or sliding windows; user
  compaction is one daemon op behind CLI, page and tool instead of a web-only hack; schema and web types
  regenerate through the project's own commands, never by hand.
- anyio only. ruff bans `asyncio` and `subprocess` (TID251). The asyncio seam lives inside the `acp` library.
- Worker isolation is a process per session, forked from a forkserver that preloads only the transport
  (`launcher.PRELOAD`, default `zmq.asyncio`). Everything else — aid itself, the model SDKs, `acp` — imports fresh
  in each worker, so restarting a worker picks up new code. `NativeSpec.python_path` is prepended per worker,
  but a preloaded module and its dependencies win over it. `aid daemon --preload-module` shares more pages
  copy-on-write, `--no-preload` starts bare.
- Keep workers subinterpreter-ready, for when pydantic-core and pyzmq load in more than one interpreter
  (measured 2026-09 on 3.14.7: both refuse isolated subinterpreters; PyO3 allows one interpreter per process):
  - `worker.main` takes only `str` arguments and mutates no process-wide state (cwd, `os.environ`, signals).
  - Process-wide setup (chdir, environ) lives in `launcher.process_main`.
  - The daemon depends on `Launcher`/`WorkerHandle`, never on multiprocessing.
- ACP `session/update` must be read through the connection observer (`AcpBackend.observe`). The library runs
  `Client.session_update` as separate tasks, which can land after `prompt()` returns.
- Native: an unset `max_requests` explicitly disables the engine's default request cap. Turn cleanup
  persists available new messages and usage on success, failure and cancellation. Compaction owns the old
  history prefix; finalization appends only the run's new messages, never overwrites that prefix. Unanswered
  tool calls are closed as interrupted, not replayed: their effects may already have happened.
- multiprocessing re-imports the parent's `__main__` in each worker: every entry script needs a
  `if __name__ == "__main__":` guard.
- Daemon shutdown: `_stop_all` runs inside the task group; worker watchers are shielded and must see every exit.
- Permission modes: `allow`, `deny`, `ask`, `supervisor`, `ask_supervisor` (the last two route through the
  supervision plugin, above; without a supervisor they are `ask`). Engines branch on `PermissionMode.waits`, never
  on `ask` alone; Claude's rules ask for every waiting mode (`claude_rules`).
- Private sessions (`NativeSpec.private`, `Private.hosts`; docs/audit-campaigns.md): a viewer is cleared if it is the
  operator (control, no agent key), a plugin with `private`, or a private session's agent. `_dispatch` hides private
  sessions from the rest: `ListSessions` filtered, any request naming one (`session`, `to`, `sender`, `supervisor`,
  spec `parent`/`supervisor`, a supervision event's session) → `not_found` (never `forbidden`: names leak repos); a
  private `CreateSession` → `forbidden`. Always: a private sender messages only private recipients; a private caller
  calls only `private` plugins; message hooks of other plugins are skipped for private traffic; parent and child match
  in privacy (`_create`); a private session's supervisor is private (`_check_supervisor`). Events: `_publish` sends a
  session topic to the open plugin PUBs only if the session is known and open (unknown = private); the full list on
  operator + private PUBs, `OPEN_SESSIONS` (the open list) on the open PUBs as `sessions/`. A plugin client follows one
  events socket, chosen by its grants (`plugin.json`, or the hosted class's); the private ZAP domain wants `read` +
  `private`. Worker: `open_native` refuses to start unless `check_host` finds the model's client host
  (`Hosted.host(settings)`: OpenAI/Anthropic `base_url`, google-genai's `_http_options`) in `hosts`; a model naming no
  host is refused (test models say `model.test`). Not covered: a same-user process (a non-private session with a
  shell) can open control.sock or read the state dir (Lillecarl/aid#4, #44). `aid new-py --private HOST`.
- Tool ceiling (Lillecarl/aid#50): `NativeSpec.tools`, regexes fullmatched against tool names
  (`offers`); the backend never offers a tool none matches, aid's and plugins' included, so a call to one is an
  unknown tool. Above modes, not one of them: `deny` still offers. Set at creation (`aid new-py --tool`, the create
  form's Tools, one regex per line); forks inherit it; agents never set it. Native only: other engines lack the
  field until each enforces it (Claude: `--tools` needs its built-in names; ACP: refuse).
  Spec as the agent: `NativeSpec.instructions`, a third source beside `agent`/`target`, requires `model` and
  `tools`; `open_native` builds `Agent(model, instructions, [coding_tools, supervisor_tools])` plus aid's toolset, so the ceiling
  picks from every named tool and `tools=[]` offers none. `spec.runs` names it "instructions" (`Started.agent`,
  `SessionStatus.runs`). `aid new-py NAME --instructions[-file] … --model … --tool …`; the form's agent choice
  "none". Agents spawn such specs through `plugins/spawn/`, attenuated by `aid.spawn.child_spec` (pure).
- Native tool permissions: a mode per tool name. `NativeSpec.tool_permissions` overrides; unset, a `Tool.asks`
  tool (python, shell, background, monitor, apply_edits, remove, ask_user: they permit inside, with their own titles;
  `remove` deletes at once from a plan, below)
  follows `permission`, any other is allowed. The loop gates every non-`asks` call through `Coding.permit`, so
  `*` hooks cover all tools. The worker lists its tools in `Hello` (`SessionTool`: name, group by module,
  asks); the daemon keeps them in `tools.json` and shows `SessionStatus.tools` with the spec's modes.
  `SetPermissions` (control, page, `aid permissions`, plugin `permissions`; never agents) writes the spec under
  `spec_lock`, then the running worker applies it to its next permit. claude-sdk: `ClaudeSdkSpec.tool_permissions`
  become Claude Code's own rules per prompt (allow `--allowedTools`, deny `--disallowedTools`, which drops the tool
  from the model's list, ask a `permissions.ask` setting), an unset tool follows Claude's defaults (`ToolMode.mode`
  None), and what Claude asks comes as `can_use_tool` only with `--permission-prompt-tool stdio` (2.1.310). Its
  tools come from each prompt's `system init`; the worker sends `ToolsReported` before the prompt's reply when
  they change. ACP lists no tools: the daemon learns each name from a `PermissionRequest`
  (`_learn_tool`, `tool_group` in spec.py, no `acp` import in the daemon) and `AcpSpec.mode_for` answers it,
  changed live through the worker. aid answers only what the agent asks. claude-tty: `ClaudeTtySpec.tool_permissions`
  as the same rules at launch (`claude_rules`; ask merged into aid's `--settings` file), no default (Claude's;
  measured 2.1.310: a denied tool is gone, an ask tool reaches aid as a PermissionRequest);
  names learned from `PermissionRequest` and `ToolCall.tool` (transcript `tool_use`); a change restarts the
  worker once neither aid's turn nor `working` runs (`restart_pending`, settled from `Activity` too).
  The page's Permissions tab is the only place a person sets modes; the create form sends `ask`.
- ACP permission requests: `AcpSpec.permission` allow/deny answers at once; `ask` waits for `AnswerPermission` (web
  card, `aid answer`), refused after `permission_timeout`. The worker's `_Client` emits `PermissionRequest` and
  `PermissionDecision` into the running prompt's events, whatever the mode; the daemon keeps the pending ones in
  `SessionStatus.permissions`. Turn cancel/end answers pending ones `cancelled` (ACP requires it). Requests only
  come mid-prompt; outside one, `ask` falls back to deny. claude-tty asks in its pane and through the PermissionRequest hook.
- claude-tty depends on Claude Code's screen and transcript, neither an API. `claude_tty.py` and
  `transcript.py` say what was measured, on which version; re-measure before changing them.
- libpymux and the `pymux` binary come from one pyterm pin (`default.nix`): the wire protocol still moves.
- Turn text and thinking reach every watcher on `session/<name>/stream/` (`StreamChunk`), published after the
  recorder took the delta, so a block's history entry precedes the next block's first delta; Chat shows chunks of
  turns it did not send as unrecorded rows and drops them on each recorded entry (one events socket keeps order).
- Web live views stream only while their tab is mounted and the browser tab is visible: status, list and history by
  `api.follow*` (the events WebSocket closes on a hidden tab), Terminal by `<pymux-pane>` (pyterm's `pymux-element`, linked in as `node_modules/pymux-pane`)
  over `/api/sessions/{name}/pane`, which relays `libpymux.PaneStream` without reading it. The element styles
  through the CSSOM, so the CSP stays `default-src 'self'`: do not add inline allowances.
- Web: a session runs commands on the host, so login needs a verified email on the allowlist; every WebSocket checks
  the Origin (it carries no CSRF header), and a mutating HTTP request (logout) needs the CSRF header. The CSP forbids inline script; agent output is text, never HTML.
  Markdown goes through marked's lexer into Svelte elements (`Markdown.svelte`), never `{@html}`.
- CodeMirror (`CodeView.svelte`) must live in a shadow root: on a document style-mod adds a style element,
  which the CSP refuses; in a shadow root it adopts a constructed stylesheet.
- Colours: `aid web --theme pygments:<name>|base16:<name>`, pymux's spelling, served as `/theme.css`
  (`web/theme.py`): page variables from pymux's roles, code as Pygments short token classes under `.hl`.
  Components use the variables (`--bad`, `--muted`, …), never literal colours.
- Highlighting: `/file` carries `highlights`, tree-sitter spans (`web/highlight.py`) as UTF-16 offsets and Pygments
  classes. Grammars: `nix/tree-sitter-grammars.nix` → `AID_TREE_SITTER_GRAMMARS` (wrapper, shell, tests). Same node:
  the later pattern wins, as tree-sitter-highlight does. Unknown capture names fall back by prefix (`CLASSES`).
  A grammar's `injections.scm` (packaged where upstream ships one; js's for ts/tsx) adds layers drawn over the host,
  their plain text keeping its class: language by `#set!` or the captured text (a Nix `/* bash */` comment),
  `injection.combined` one parse per pattern, `INJECTION_DEPTH` deep; unknown languages skipped.
  Measured 2026-10-25: 0.7 MiB of Python 0.6 s; 1 MiB of Nix 4.4 s, 3.3 s of it the injections query (upstream's
  builder patterns, their structure, not `#match?`). The chat's code (`Code.svelte`: fenced blocks by tag, shell
  commands as bash, tool arguments by key or by the call's `path`, a numbered output of a one-path call as that file
  with the numbers in a gutter: `tools.numbered`) POSTs to `/api/highlight` (CSRF header, 256 Ki
  chars): a grammar by name or file type, else Pygments' lexer (`highlight.highlight`), else plain. The page caches
  answers and waits for a streaming block to settle.
- `/api/sessions/{name}/files` and `/file` (`web/files.py`) read the session cwd in aid web itself; every path
  must resolve inside it.
