---
name: nixpkgs-updates
description: Reviewing a version bump - what changed upstream, hashes, dropped patches, breaking changes for dependants, update scripts, and security fixes. Load for any PR whose title reads "pkg old -> new".
---

# Reviewing a version bump

## What to check

- The version in the title and the expression agree; the hash changed with it. A hash that stayed while the version
  moved means the old source is still fetched (a fixed-output cache hit).
- Every hash that follows the source changed too: `cargoHash`, `vendorHash`, `npmDepsHash`, a `pnpm` deps hash,
  `outputHash` of a fixed-output step. One left behind fails, or worse, builds the old dependencies.
- Patches: one that upstream merged must go, or the build fails on it; one that still applies stays. Read the
  patch and upstream's changelog in the tree (`CHANGELOG*`, `NEWS*`, `debian/changelog`) if the source carries one.
- Dependencies upstream added or dropped are reflected in the inputs.
- A major version: breaking changes reach dependants. Search the tree for the attribute name (`search`) to find
  them, and say which ones the PR left unbuilt. The build results in the prompt cover what CI lists as rebuilds.
- `passthru.updateScript` (often `nix-update-script { }`) and `passthru.tests`: their presence lets bots and CI
  carry the next bump; a removed one needs a reason.
- `meta.changelog` points at the new version, not the old one.

## Security

- A bump that fixes a CVE belongs on the stable branch too: a review can ask for the backport label.
- A bump that drops a CVE patch must show the fix is in the new release.
- A downgrade, or a move to a fork or a new upstream URL, needs a reason in the PR body.

## Version strings

- Releases: the upstream version. Unreleased commits: `<last release>-unstable-YYYY-MM-DD`, the commit's date;
  `0-unstable-...` when upstream has no usable release. See Versioning in `pkgs/README.md`.
