---
name: nixos-modules
description: NixOS modules and tests - options and types, mkEnableOption and mkPackageOption, systemd service hardening, settings formats, release notes, nixos/tests. Load for PRs under nixos/modules or nixos/tests.
---

# Reviewing a NixOS module

The rules: `nixos/doc/manual/development/` in the checkout (writing modules, option declarations, option types,
settings options, writing tests).

## Options

- `lib.mkEnableOption "the X service"` and `lib.mkPackageOption pkgs "x" { }` over hand-written equivalents.
- Every option has a precise `type` (`types.port`, `types.listOf types.str`, `types.enum [ ... ]`); `types.anything`
  or `types.attrs` loses checking. A `default` that depends on other options uses `defaultText`.
- Free-form configuration: a `settings` option with `format = pkgs.formats.<ini|json|yaml|toml> { };` and
  `type = format.type` (RFC 42), not a raw `extraConfig` string, in new modules.
- A renamed or removed option keeps a `lib.mkRenamedOptionModule` or `lib.mkRemovedOptionModule` entry.
- A user-visible or breaking change gets a release notes entry under `nixos/doc/manual/release-notes/`.

## Services

- `systemd.services.x` with `DynamicUser = true;` or a dedicated user, `StateDirectory`, and hardening
  (`ProtectSystem = "strict"`, `ProtectHome`, `PrivateTmp`, `NoNewPrivileges`, `CapabilityBoundingSet`,
  `SystemCallFilter`, ...). A service running as root, or with hardening removed, needs a reason.
- Secrets never go into the Nix store: options for them take a file path (`passwordFile`), read at runtime
  (`LoadCredential`), not a string interpolated into a generated config.
- `networking.firewall` ports open only behind an option such as `openFirewall` (default false).

## Tests

- A new module or service gets a test in `nixos/tests/`, registered in `nixos/tests/all-tests.nix`, and linked from
  the package with `passthru.tests`.
- The test checks the service works (a request, a command's output), not only that the unit started.
