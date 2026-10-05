# The aid smoke test: one UML guest running dex, the daemon and the web UI as the
# NixOS module ships them, and phases asserting they answer. No forwards, no host
# services: everything the test touches runs inside the guest. The browser phase runs
# tests/test_browser.py in the guest unmodified: the `web` fixture boots its own
# dex, daemon and app on ephemeral ports, so the systemd units and the suite never meet.
{ lib, pkgs, vivarium, aidSrc ? ../. }:
let
  aid = import aidSrc { inherit pkgs; };
  # The dev shell's venv as a package: the suite's exact dependency versions, runnable in the guest.
  testEnv = aid.set.mkVirtualEnv "aid-vivarium-test" {
    aid = [ "test" ];
    pyrun = [ "test" ];
  };
  # Only the suite and its pytest config: the whole working copy never enters the guest closure.
  testsTree = lib.cleanSourceWith {
    src = aidSrc;
    filter =
      path: _type:
      let
        # toString, never interpolation: "${aidSrc}" would copy the tree to the store and the
        # prefix would never match the source paths the filter sees.
        base = toString aidSrc;
        rel = lib.removePrefix (base + "/") (toString path);
      in
      toString path == base || rel == "tests" || lib.hasPrefix "tests/" rel || rel == "pyproject.toml";
  };
  runBrowserTests = pkgs.writeShellScriptBin "aid-browser-tests" ''
    set -eu
    export PATH=${lib.makeBinPath [ testEnv pkgs.dex-oidc pkgs.coreutils ]}:"$PATH"
    export PLAYWRIGHT_BROWSERS_PATH=${aid.browsers}
    export AID_WEB_ASSETS=${aid.ui}
    export HOME="$(mktemp -d)"
    # Unbuffered, or a killed run's captured output shows nothing of where it stuck.
    export PYTHONUNBUFFERED=1
    # UML is slow: every wait in the suite may take minutes, not seconds.
    export AID_TEST_TIMEOUT=300
    cd ${testsTree}/tests
    # Extra arguments replace the default test selection, so a phase can probe one test at a time.
    if [ $# -eq 0 ]; then set -- test_browser.py; fi
    pytest "$@" -p no:cacheprovider --browser chromium -q --junitxml=/artifacts/junit/browser.xml
  '';
  dexConfig = pkgs.writeText "dex.json" (
    builtins.toJSON {
      issuer = "http://127.0.0.1:5556/dex";
      storage = { type = "memory"; };
      web = { http = "127.0.0.1:5556"; };
      oauth2 = { skipApprovalScreen = true; };
      staticClients = [
        {
          id = "aid";
          secret = "aid-secret";
          name = "aid";
          redirectURIs = [ "http://127.0.0.1:8080/auth/callback" ];
        }
      ];
      enablePasswordDB = true;
      staticPasswords = [
        {
          email = "admin@example.com";
          # bcrypt of "password", from dex's example configuration.
          hash = "$2a$10$2b2cU8CPhOTaGrs1HRQuAueS7JTT5ZHsHSzYiFPm1leZck7Mc8T4W";
          username = "admin";
          userID = "user-0";
        }
      ];
    }
  );
  # A test, so the secrets are fixed and public, as in tests/web_harness.py.
  envFile = pkgs.writeText "aid-web.env" ''
    AID_OIDC_CLIENT_SECRET=aid-secret
    AID_WEB_SESSION_SECRET=test-session-secret
  '';
in
vivarium.mkTest {
  name = "aid-smoke";
  nodes.aid = {
    imports = [ aid.nixosModules.default ];
    # The default 256M fits the daemon, not chromium beside it. A ceiling, not a cost, under UML.
    vivarium.memory = "2048M";
    environment.systemPackages = [ pkgs.curl runBrowserTests ];
    services.aid = {
      enable = true;
      web = {
        enable = true;
        issuer = "http://127.0.0.1:5556/dex";
        allowEmails = [ "admin@example.com" ];
        environmentFile = "${envFile}";
      };
    };
    # The web UI reads the issuer's discovery document at startup, so it starts after dex.
    systemd.services.aid-web = {
      after = [ "dex.service" ];
      wants = [ "dex.service" ];
    };
    systemd.services.dex = {
      description = "dex, the test OIDC provider";
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "exec";
        ExecStart = "${lib.getExe pkgs.dex-oidc} serve ${dexConfig}";
        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = "2s";
      };
    };
  };
  phases.smoke = {
    script = ../tests/vivarium/smoke.py;
    after = [ "boot" ];
  };
  phases.browser = {
    script = ../tests/vivarium/browser.py;
    after = [ "smoke" ];
  };
}
