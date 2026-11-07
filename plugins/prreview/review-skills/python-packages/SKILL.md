---
name: python-packages
description: Python packages in nixpkgs - buildPythonPackage with pyproject, build-system and dependencies, tests with pytestCheckHook, relaxed and removed requirements. Load for PRs under pkgs/development/python-modules or any python3Packages attribute.
---

# Reviewing a Python package

The rules: `doc/languages-frameworks/python.section.md` in the checkout.

- `buildPythonPackage` in `pkgs/development/python-modules/<name>/default.nix`, listed in
  `pkgs/top-level/python-packages.nix`. An application rather than a library uses `buildPythonApplication`, often
  in `pkgs/by-name`.
- `pyproject = true;` with `build-system = [ setuptools ];` (or hatchling, poetry-core, flit-core, ...). The old
  `format = "setuptools"` and `nativeBuildInputs` for the build backend are legacy; flag them in new or rewritten
  expressions.
- `dependencies = [ ... ];` for runtime requirements, `optional-dependencies = { extra = [ ... ]; };` for extras.
  A requirement missing from `dependencies` fails at import, often only at runtime.
- Tests: `nativeCheckInputs = [ pytestCheckHook ... ];`. `disabledTests` and `disabledTestPaths` each need a reason
  in a comment (network, sandbox, flaky upstream). Tests switched off wholesale (`doCheck = false;`) need a reason.
- `pythonImportsCheck = [ "module" ];`: cheap, and it catches a broken package. Ask for it when missing.
- `pythonRelaxDeps` loosens a version bound, `pythonRemoveDeps` drops a requirement. Each is a claim that the
  package works anyway: look for a comment or upstream evidence, and check it is not hiding a real incompatibility.
- `pythonOlder`/`disabled` guards for interpreter versions nixpkgs no longer ships are dead code.
- Source: prefer the upstream repository (`fetchFromGitHub`) when PyPI's sdist lacks tests; `fetchPypi` is fine
  otherwise. A wheel (`format = "wheel"`) is a binary package: it needs a reason and `sourceProvenance`.
