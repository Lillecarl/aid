---
name: go-packages
description: Go packages in nixpkgs - buildGoModule, vendorHash, ldflags, subPackages, tags and CGO. Load for PRs whose package builds with buildGoModule or carries a vendorHash.
---

# Reviewing a Go package

The rules: `doc/languages-frameworks/go.section.md` in the checkout.

- `buildGoModule (finalAttrs: { ... })` with `vendorHash`. A bump that keeps the old `vendorHash` while `go.mod`
  changed fails or builds old modules. `vendorHash = null;` only when upstream vendors its modules in the source.
- `proxyVendor = true;` only when the vendor directory cannot be built (it needs a reason).
- `ldflags = [ "-s" "-w" "-X main.version=${finalAttrs.version}" ];`: check the `-X` path names a variable that
  exists in the source, or the version string stays empty.
- `subPackages` limits what builds; a missing entry drops a binary the package used to ship.
- `tags` for build tags; `env.CGO_ENABLED = 0;` for a static, pure-Go build where cgo is not needed.
- Tests run in `checkPhase`; skipped tests (`checkFlags = [ "-skip=..." ]`) each need a reason.
- `versionCheckHook` and `meta.mainProgram` as for any package.
