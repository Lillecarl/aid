# Audit campaigns

Goal: a self-hosted model reads a whole tree of private repositories, one unit at a time, under supervision. It
records each issue it finds in a local database. A second AI pass and then a person review those findings. No
code and no finding leaves the operator's infrastructure.

## Decisions

- **A plugin drives the walk, not a model.** The campaign plugin lists the repositories and their tracked files
  and prompts one unit at a time. It measures coverage from the reads the session recorded, not from what the
  model says it did. Its state is in a database, so a daemon restart resumes the campaign.
- **Findings come in through one tool.** `report_finding` writes one row to the findings database. It takes no
  path to write to. The plugin takes the repository from the caller's spec, and the daemon stamps the caller.
  A model with a read-only ceiling gets this tool and no other write.
- **Every model in the chain is self-hosted.** This covers the worker, its supervisor, compaction and the review
  pass. The bundled supervisor runs on a hosted model, so a campaign overrides its model. A campaign offers no
  tool that asks, so no classifier runs.
- **Privacy is enforced, not incidental.** A session marked private stays out of every relay: chat plugins,
  settled-turn notices, and plugins that only hold `read`. A person sets an allowlist of model endpoints, and the
  campaign refuses any other.

## Phases

1. **Findings store.** `plugins/findings/`: an `aid.sql` database in the plugin directory, with `findings`
   (repository, path, lines, category, severity, title, detail, evidence, session, turn, model, time, dedup key)
   and `verdicts` (pass, status, note, by). The `report_finding` tool. `aid findings` queries and exports.
   Duplicates are kept and marked, never refused: a refusal teaches the model to vary its titles.
2. **Read-only search.** A `search` tool in `aid.coding_tools`: regex over files, in process, confined by
   `Coding.resolve`. Without it, a model must read every file to find one name.
3. **Privacy.** The `private` spec field, honoured by the daemon's event delivery and by every relay plugin. The
   model endpoint allowlist.
4. **Campaign plugin.** Units from `git ls-files`, with size and binary filters. One session per repository, on
   an instructions spec with a read-only ceiling. A per-repository brief first, then one turn per unit.
   Supervision on a self-hosted model. Bounded concurrency.
5. **Pilot.** Two or three small repositories. Measure context growth per unit, the server's concurrency, cost in
   time, and the false-positive rate. Size the full run from these numbers.
6. **Review.** A second campaign over the findings: confirm, reject or mark duplicate, with a reason. Then a
   person's verdicts, in the web UI and the CLI.

## Limits known now

- Findings can hold secrets, because leaked secrets are one kind of finding. The database stays in the plugin's
  state directory, mode 0600, and never in a reviewed repository.
- A plugin with the `read` grant hears every session's events today. Phase 3 has to close that for private
  sessions before a real run.
