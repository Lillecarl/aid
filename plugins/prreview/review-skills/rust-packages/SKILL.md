---
name: rust-packages
description: Rust packages in nixpkgs - rustPlatform.buildRustPackage, cargoHash, cargo features, tests and checkFlags. Load for PRs whose package builds with Cargo (cargoHash, Cargo.lock, rustPlatform).
---

# Reviewing a Rust package

The rules: `doc/languages-frameworks/rust.section.md` in the checkout.

- `rustPlatform.buildRustPackage (finalAttrs: { ... })` with `cargoHash`. A bump that keeps the old `cargoHash`
  builds with stale crates or fails; the hash changes with almost every release.
- A vendored `Cargo.lock` copied into nixpkgs (`cargoLock.lockFile`) is for sources without a lock file or with git
  dependencies; it must be updated with every bump.
- Features: `buildFeatures`, `buildNoDefaultFeatures`. A disabled default feature needs a reason.
- Tests run by default. `checkFlags = [ "--skip=name" ];` for single tests, each with the reason (network,
  sandbox, timing). `doCheck = false;` wholesale needs a strong reason. `cargoTestFlags` passes flags to cargo.
- System libraries: `pkg-config` in `nativeBuildInputs`, the library (`openssl`, `sqlite`, ...) in `buildInputs`.
  Bundled C sources built by a `-sys` crate where nixpkgs has the library are worth replacing (often an env var
  or feature such as `LIBSQLITE3_SYS_USE_PKG_CONFIG`).
- A GUI that dlopens libraries at runtime (Vulkan, Wayland, X11, libGL) needs them on the runpath:
  `autoPatchelfHook`/`runtimeDependencies` or a wrapper setting `LD_LIBRARY_PATH`; check the list matches.
- `versionCheckHook` in `nativeInstallCheckInputs` with `doInstallCheck = true;` is a cheap smoke test.
