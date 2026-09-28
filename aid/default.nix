# aid as a pyproject.nix builders package, in a set that pyterm's builders assemble. What aid needs is declared
# once, in pyproject.toml; the renderer reads it. See ../default.nix for the set.
{
  lib,
  stdenv,
  python,
  pyprojectHook,
  resolveBuildSystem,
  mkVirtualEnv,
  mkProject,
  runCommand,
  # The pymux application, whose `bin/pymux` the claude-tty tests start servers with.
  pymuxApp,
  # The OIDC provider the web tests log in through.
  dex,
  # The built Svelte UI, which the web tests serve.
  webUi,
  # The speech model the speech tests transcribe with.
  speechModel,
}:
let
  root = ../.;

  package =
    (mkProject {
      inherit root python;
      extra = rendered: {
        passthru = rendered.passthru // {
          inherit tests;
        };
        meta = rendered.meta // {
          description = "AI daemon: persistent ACP, pydantic-ai and interactive Claude sessions behind an async API";
          homepage = "https://github.com/Lillecarl/aid";
          mainProgram = "aid";
          platforms = lib.platforms.linux;
        };
      };
    })
      {
        inherit stdenv pyprojectHook resolveBuildSystem;
      };

  testSources = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      (root + "/tests")
      (root + "/pyproject.toml")
    ];
  };

  testEnv = mkVirtualEnv "aid-test-env" { aid = [ "test" ]; };

  tests = runCommand "aid-tests" { nativeBuildInputs = [ testEnv pymuxApp dex ]; } ''
    cp -r ${testSources}/. .
    chmod -R +w .
    export HOME="$TMPDIR" PYTHONDONTWRITEBYTECODE=1 AID_WEB_ASSETS=${webUi} AID_TEST_SPEECH_MODEL=${speechModel}
    # The sandbox sets it to /no-cert-file.crt, and httpx loads it for every client, http:// ones too.
    unset SSL_CERT_FILE
    python -m pytest -q -p no:cacheprovider
    touch $out
  '';
in
package
