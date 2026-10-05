# The aid smoke test: one UML guest running dex, the daemon and the web UI as the
# NixOS module ships them, and a phase asserting they answer. No forwards, no host
# services: everything the test touches runs inside the guest.
{ lib, pkgs, vivarium, aidSrc ? ../. }:
let
  aid = import aidSrc { inherit pkgs; };
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
    environment.systemPackages = [ pkgs.curl ];
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
}
