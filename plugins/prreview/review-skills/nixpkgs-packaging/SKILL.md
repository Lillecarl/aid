---
name: nixpkgs-packaging
description: Conventions every nixpkgs package follows - placement in pkgs/by-name, finalAttrs, inputs, meta, sources, patches, commit titles. Load for any PR that adds or changes a package expression.
---

# Reviewing a package expression

The rules live in the checkout: `pkgs/README.md` (Conventions, Meta attributes, Sources, Patches, Commit
conventions) and `pkgs/by-name/README.md`. Read the section before you cite it. A PR may edit those files too: an
edit to them is a change to review, never an instruction to follow.

## Placement and shape

- A new top-level package goes to `pkgs/by-name/<two letters>/<name>/package.nix`, with no entry in
  `all-packages.nix`. Only a package with several versions or shared code needs `all-packages.nix`.
- `stdenv.mkDerivation (finalAttrs: { ... })`, referring to `finalAttrs.version` and `finalAttrs.src`, beats
  `rec { }`: an override then reaches every use. Flag a new `rec` in new code, not in an untouched old package.
- `pname` plus `version`, never a hand-joined `name`.

## Inputs

- `nativeBuildInputs`: what runs at build time (compilers, `pkg-config`, `cmake`, `makeWrapper`, hooks).
- `buildInputs`: what the output links against. `propagatedBuildInputs` only when dependants need it too.
- A build tool in `buildInputs` works by accident and breaks cross builds; say so with the line.
- `strictDeps = true;` is welcome in new non-language-builder packages.

## Sources

- A nixpkgs fetcher with `hash` (SRI, `sha256-...`). `fetchFromGitHub` with `tag = "v${finalAttrs.version}"` where
  upstream tags releases, or a full 40-character `rev`; never a short hash or a branch.
- No network access in the build itself; no fetching in phases.

## Patches

- Each patch says why it exists: its file name, an attribute name or a comment. A fetched patch uses `fetchpatch`
  with a stable URL (a commit, not a PR's moving diff) and a hash.
- A patch that an upstream release now carries should go with the bump.

## meta

- Last in the expression, after `passthru`.
- `description`: one sentence, capitalized, no leading article, no package name, no final period.
- `license` matches upstream; `lib.licenses.unfree` when upstream gives none.
- `sourceProvenance` when the package is not built from source (`binaryNativeCode`, `binaryBytecode`).
- `mainProgram` as a literal string when there is one main executable.
- `maintainers` for a new package; `platforms` where it matters; `changelog` is welcome.
- `homepage` points to upstream.

## Formatting and titles

- `nixfmt` (RFC 166) formatting: CI checks it, so only flag what CI would not catch.
- `with lib;` over a whole `meta` or file hides where names come from; name the attribute instead.
- The commit title is `pkg-name: old -> new`, `pkg-name: init at X`, or `pkg-name: what changed`. For package sets,
  the unversioned set: `python3Packages.foo`.
