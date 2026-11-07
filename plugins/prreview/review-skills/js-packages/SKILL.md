---
name: js-packages
description: JavaScript and TypeScript packages in nixpkgs - buildNpmPackage and npmDepsHash, pnpm, Yarn, the project's own Node.js and lock file. Load for PRs with npmDepsHash, pnpm or yarn dependencies, or a package.json.
---

# Reviewing a JavaScript package

The rules: `doc/languages-frameworks/javascript.section.md` in the checkout (General principles first).

- Use the project's package manager and its lock file: npm with `buildNpmPackage` and `npmDepsHash`, pnpm with the
  pnpm deps fetcher and its hash, Yarn with the Yarn hooks. A different manager than upstream's needs a reason.
- Every dependency hash changes with the lock file. A bump that keeps the old one fails or ships old modules.
- A lock file copied into nixpkgs (because upstream has none) must be regenerated with each bump.
- Use the Node.js version upstream asks for (`engines` in `package.json`, `.nvmrc`), not just the default `nodejs`.
- `npmFlags`, `makeCacheWritable` and `--legacy-peer-deps` work around lock problems: each needs a reason.
- Build output only: the installed tree should not carry the whole `node_modules` of dev dependencies when a
  bundled build exists (`npm prune --omit=dev` or the builder's options).
- Electron apps: use nixpkgs' `electron`, not the downloaded binary; check the version matches upstream's major.
