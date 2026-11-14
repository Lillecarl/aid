# nixpkgs PR review

Goal: a self-hosted model reviews open nixpkgs pull requests from several angles. A judge merges the angle reviews
into one review. Nothing reaches GitHub: the judge's "post" tool records the review in a local database, so stronger
models and people can analyse the reviews before any posting is built. A person approves or rejects each review.

## Use

```sh
aid prreview start run.json       # {"name": "nixrev", "model": "vllm/glm-5.3-flash", "max_prs": 3}
aid prreview status nixrev
aid prreview query --table "SELECT number, event, body FROM reviews"
aid prreview approve nixrev 584792 --note "fair"   # or reject; the PR's latest reviewed head
aid prreview pause nixrev         # stops at once; the PR is reviewed again on resume
aid prreview resume nixrev
```

The daemon needs `gh` (logged in), `git` and `pynix-review` on its PATH, or `commands` in the config names them.
pynix-review builds through the local build daemon (pynixd), so the aid daemon's user must reach its socket.

## Decisions

- **A plugin drives it, like a campaign.** `plugins/prreview/` picks PRs, prepares each one, runs the reviewer
  turns and records the results. It drives sessions through `aid.drive`, so a model outage costs no attempt.
- **One PR at a time.** Disk stays bounded. The plugin fetches only the PR's head (`--depth 1`) into a bare
  repository of its own and checks it out as a worktree. After the PR, the worktree and the fetched objects go.
  The built outputs stay: the build daemon collects them.
- **Every review stays readable in the UI (#57).** Each PR gets a holder session `<run>.<PR>` under the run's root;
  the duplicate check, the angles and the judge are its children, shown as its tabs. They stop after the PR and
  stay. The plugin writes notes, which need no model turn: on the holder, the start, the builds, the review and
  the end; on the root, one line per PR. The root's Chat is the run's log.
- **Open, green PRs.** Open, every check green (`--checks success`), not a draft. The build has a fixed budget,
  so a PR of any size qualifies, up to `max_derivations` (1000) changed derivations.
- **Picked across orders, by chance.** Over 1000 PRs match, and GitHub's search answers at most 1000 in one order,
  so one order hides most of them for good. Each pick searches three orders, 100 hits each: recently updated
  (weight 0.5), most thumbs-up (0.3), longest waiting (0.2). It picks an order at random by weight, then one of
  that order's first 10 unreviewed PRs at random. Reviewed PRs come last, to catch a head that moved. The weights
  are the run config's `sources`.
- **pynix-review builds.** `pynix-review pr N` reads CI's rebuild list (the PR workflow's `comparison` artifact,
  `rebuildsByPlatform.x86_64-linux`), diffs the head against its merge base, and builds only the derivations that
  differ. Builds go through the local daemon, which spreads them over the remote builders. No build starts after
  `wall` (3600 s); the ones still running stop at `kill_after` (5400 s). A PR without the artifact, over
  `max_derivations`, or at a head that moved since the pick is skipped. A PR CI lists no rebuild for is reviewed
  unbuilt. The reviewers read the report as `built.md`, the comment pynix-review writes from `built.json`: without
  its graph link, clipped at `builds_limit`.
- **Duplicates first.** Before the builds, the plugin searches for PRs whose title names the same package or
  rebuilt attribute: open ones, and ones merged in the last 30 days. It keeps those that touch one of the PR's
  files. A read-only session compares their diffs and names the duplicates and the better change. A merged
  duplicate always wins, as master has the change. Between open ones the model's choice wins; when it calls them
  equal, the most recently updated PR wins. A PR that loses ends as a duplicate of the winner: nothing is built or
  reviewed. Every check is in `duplicate_checks`.
- **Reviewers only read.** The model is abliterated, so its sessions get `ls`, `read`, `outline` and `search`, and
  nothing that writes or runs a program. The plugin runs git, gh and Nix, never the model.
- **Angles are sessions, the judge is one more.** The default angles are security, conventions, alternatives and
  correctness; the config can replace them. Each angle is a session with fresh turns in the worktree, two at a time.
  The judge gets the PR, the diff, the build results and every angle's answer. Its `submit_review` tool records one
  review: an event (APPROVE, COMMENT, REQUEST_CHANGES), a body and line comments. The plugin refuses a comment on a
  line the diff does not show, as GitHub would, and the judge tries again in the same turn. A judge turn that ends
  without a review is repeated, up to `attempts`.
- **Nix guides on demand.** The instructions carry a short block of Nix knowledge. Eight guides in the Agent
  Skills format (`review-skills/`: packaging, updates, Python, Rust, Go, JavaScript, NixOS modules, binary
  packages) load with the `skill` tool. The plugin names the ones a diff calls for in the prompt, from the paths and
  builders it touches. They are the reviewers' only skill source (`AID_SKILLS_ONLY`): a PR can add skill
  directories to its own checkout.
- **PR text is untrusted.** The title, body, diff and files are data. The reviewer tools cannot act on them. The
  worst a poisoned PR can do is a wrong review, which is recorded and not posted.

## State

`prreview.db` in the plugin directory: `runs`, `prs` (number, head, title, body, diff, state, why skipped,
`duplicate_of`, error), `duplicate_checks` (candidates, the model's answer, the winner), `build_reports` (merge
base, `built.json`, `built.md`, the graph link), `angle_reviews` (angle, answer, the turn's usage), `reviews` (event,
body, comments, session, and a person's decision: approved or rejected, by whom, when, a note). `builds` holds
per-attribute rows from before pynix-review. Every row is keyed by run, PR number and head, so a head that moved is
a new review. An interrupted PR starts again, but keeps the angle answers it has.

## Open

- Posting: a later operator command, for approved reviews only.
- A calibration set: merged PRs and their human reviews, to measure the judge against.
- A failed PR is not retried.
