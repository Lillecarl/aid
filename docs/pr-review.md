# nixpkgs PR review

Goal: a self-hosted model reviews open nixpkgs pull requests from several angles. A judge merges the angle reviews
into one review. Nothing reaches GitHub: the judge's "post" tool records the review in a local database, so stronger
models and people can analyse the reviews before any posting is built.

## Use

```sh
aid prreview start run.json       # {"name": "nixrev", "model": "vllm/glm-5.3-flash", "max_prs": 3}
aid prreview status nixrev
aid prreview query --table "SELECT number, event, body FROM reviews"
aid prreview pause nixrev         # stops at once; the PR is reviewed again on resume
aid prreview resume nixrev
```

The daemon needs `gh` (logged in), `git` and `nix` on its PATH, or `commands` in the config names them.

## Decisions

- **A plugin drives it, like a campaign.** `plugins/prreview/` picks PRs, prepares each one, runs the reviewer
  turns and records the results. It drives sessions through `aid.drive`, so a model outage costs no attempt.
- **One PR at a time.** Disk stays bounded. The plugin fetches only the PR's head (`--depth 1`) into a bare
  repository of its own and checks it out as a worktree. After the PR, the worktree, the fetched objects and the
  built outputs go (`nix store delete` on those paths only).
- **Only PRs that build cheaply.** Open, every check green (`--checks success`), not a draft, label
  `10.rebuild-linux: 1-10`. No public cache holds PR outputs, but for such a PR only its own derivations build
  here. Everything else comes from cache.nixos.org.
- **CI says what to build.** The PR workflow's `comparison` artifact lists `rebuildsByPlatform.x86_64-linux`. The
  plugin builds each attribute from the worktree, with its own nixpkgs config: unfree allowed, as CI lists
  unfree packages too. It does not evaluate nixpkgs itself.
  A PR without the artifact, or with more than `max_builds` rebuilds, is skipped. A failed or timed-out build is a
  result the reviewers read.
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
- **PR text is untrusted.** The title, body, diff and files are data. The reviewer tools cannot act on them. The
  worst a poisoned PR can do is a wrong review, which is recorded and not posted.

## State

`prreview.db` in the plugin directory: `runs`, `prs` (number, head, title, body, diff, state, why skipped,
`duplicate_of`, error), `duplicate_checks` (candidates, the model's answer, the winner), `builds` (attribute,
outcome, log tail, outputs, closure size), `angle_reviews` (angle, answer, the turn's usage), `reviews` (event, body,
comments, session). Every row is keyed by run, PR number and head, so a head that moved is a
new review. An interrupted PR starts again, but keeps the angle answers it has.

## Open

- Posting: a later operator command, after the recorded reviews are checked.
- A calibration set: merged PRs and their human reviews, to measure the judge against.
- A failed PR is not retried.
