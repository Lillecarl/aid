# pyrun as a pyproject.nix builders package. It is a project inside the aid repository, on the pattern of pymux's
# libpymux, so it can move to a repository of its own. See ../default.nix for the set.
{
  lib,
  python,
  stdenv,
  pyprojectHook,
  resolveBuildSystem,
  mkProject,
  mkVirtualEnv,
  runCommand,
  # `setsid`, which a test daemonizes with.
  utilLinux,
}:
let
  package =
    (mkProject {
      root = ./.;
      inherit python;
      extra = rendered: {
        passthru = rendered.passthru // {
          inherit tests;
        };
        meta = rendered.meta // {
          description = "Run processes from async Python: scopes that kill what they started, a record per process";
          homepage = "https://github.com/Lillecarl/aid";
          platforms = lib.platforms.linux;
        };
      };
    })
      {
        inherit stdenv pyprojectHook resolveBuildSystem;
      };

  testSources = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./tests
      ./pyproject.toml
    ];
  };

  testEnv = mkVirtualEnv "pyrun-test-env" { pyrun = [ "test" ]; };

  tests = runCommand "pyrun-tests" { nativeBuildInputs = [ testEnv utilLinux ]; } ''
    cp -r ${testSources}/. .
    chmod -R +w .
    export HOME="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
    python -m pytest -q -p no:cacheprovider
    touch $out
  '';
in
package
