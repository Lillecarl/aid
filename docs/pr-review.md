# nixpkgs PR review

Goal: a self-hosted model reviews open nixpkgs pull requests from several angles. A judge merges the angle reviews
into one review. Nothing reaches GitHub: the judge's "post" tool records the review in a local database, so stronger
models and people can analyse the reviews before any posting is built.

## Decisions

- **A plugin drives it, like a campaign.** `plugins/prreview/` picks PRs, prepares each one, runs the reviewer
  turns and records the results. It drives sessions through `aid.drive`, so a model outage costs no attempt.
- **One PR at a time.** Disk stays bounded: one worktree of a shared nixpkgs clone, and the outputs of one build.
  The worktree goes away after the PR, and the build outputs are deleted (`nix store delete`, only those paths).
- **Only PRs that build cheaply.** Open, every check green (`--checks success`), label `10.rebuild-linux: 1-10`.
  No public cache holds PR outputs, but for such a PR only its own derivations build here. Everything else comes
  from cache.nixos.org.
- **CI says what to build.** The PR workflow's `comparison` artifact lists `rebuildsByPlatform.<system>`. The plugin
  builds those attributes from the worktree. It does not evaluate nixpkgs itself.
- **Reviewers only read.** The model is abliterated, so its sessions get read tools and nothing that writes or runs
  a program. The plugin runs git and Nix, never the model. A reviewer reads the build's results in its prompt.
- **Angles are sessions, the judge is one more.** Each angle (security, nixpkgs conventions, a better way, and
  others in the config) is an instructions spec on fresh turns, in the PR's worktree. Its answer is stored. The judge
  gets the PR, the diff, the build results and every angle's answer. Its `submit_review` tool records one review:
  an event (approve, comment, request changes), a body, and line comments. The plugin checks that each comment
  names a line the diff shows, as GitHub would.
- **PR text is untrusted.** The title, body, diff and files are data. The reviewer tools cannot act on them. The
  worst a poisoned PR can do is a wrong review, which is recorded and not posted.

## State

`prreview.db` in the plugin directory: `prs` (number, head sha, title, state, error, times), `builds` (attribute,
outcome, log tail, closure size), `angle_reviews` (angle, session, answer, usage), `reviews` (event, body, comments,
session). A PR whose head moved since its review is a new row.

## Open

- Posting: a later operator command, after the recorded reviews are checked.
- A calibration set: merged PRs and their human reviews, to measure the judge against.
