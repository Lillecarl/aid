---
name: binary-packages
description: Packages built from upstream binaries - .deb, .rpm, AppImage, tarballs - with autoPatchelfHook, buildFHSEnv, appimageTools, makeWrapper, unfree licenses and sourceProvenance. Load for PRs that repackage prebuilt files or use an FHS environment.
---

# Reviewing a binary package

- `meta.sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];` (or `binaryBytecode` for jars) is required; the
  license is usually `lib.licenses.unfree` or `unfreeRedistributable`, matching upstream's terms.
- Per platform sources: one URL and hash per supported system, and `meta.platforms` listing exactly those.
- ELF binaries: `autoPatchelfHook` in `nativeBuildInputs` with the libraries in `buildInputs`;
  `runtimeDependencies` or `appendRunpaths` for what is dlopened. A missing library shows only at runtime, so
  compare the list with what the binary loads (the PR body, upstream's dependency list).
- `makeWrapper`/`wrapProgram` for runtime paths and environment; `--prefix PATH` with `lib.makeBinPath`.
- FHS environments (`buildFHSEnv`, `appimageTools.wrapType2`): `targetPkgs` are the host architecture's packages,
  `multiPkgs` are built for every architecture of a multilib env. `appimageTools.defaultFhsEnvArgs` carries a
  tested default set; an expression that replaces only one half of it drops the other half.
- Desktop integration: the `.desktop` file and icons installed (`copyDesktopItems`, `makeDesktopItem`), with
  `Exec` pointing at the wrapper, not the unwrapped binary.
- The upstream download URL is versioned and stable; a "latest" URL breaks on the next release.
- Update scripts are often custom here; a hand-maintained per-platform hash list needs one.
