{ lib, pkgs }:
let
  aid = import ../. { inherit pkgs; };
in
{
  options = {
    enable = lib.mkEnableOption "aid, the AI daemon that keeps ACP, pydantic-ai and interactive Claude agents running";

    package = lib.mkOption {
      type = lib.types.package;
      default = aid.aid;
      defaultText = lib.literalExpression "(import <aid> { inherit pkgs; }).aid";
      description = "The aid package.";
    };

    pymuxPackage = lib.mkOption {
      type = lib.types.package;
      default = aid.pymux;
      defaultText = lib.literalExpression "(import <aid> { inherit pkgs; }).pymux";
      description = ''
        The pymux that starts aid's own server for claude-tty sessions. The default comes from the same pyterm
        revision as aid's libpymux, because the wire protocol between them still moves.
      '';
    };

    extraPackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ ];
      example = lib.literalExpression "[ pkgs.claude-agent-acp pkgs.opencode ]";
      description = "Packages on the daemon's PATH, so ACP sessions can start their agent commands by name.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Environment for the daemon. Workers and ACP agents inherit it unless a session opts out.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/aid.env";
      description = "systemd EnvironmentFile for secrets such as ANTHROPIC_API_KEY. Kept out of the store.";
    };

    web = {
      enable = lib.mkEnableOption "the aid web UI, behind OIDC login";

      bind = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1:8080";
        description = "host:port the UI listens on. Put a TLS proxy in front of it for anything but localhost.";
      };

      baseUrl = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "https://aid.example.com";
        description = "Where browsers reach the UI; the OIDC redirect URI is this plus /auth/callback. Default: http://BIND.";
      };

      issuer = lib.mkOption {
        type = lib.types.str;
        example = "https://dex.example.com";
        description = "The OIDC issuer URL.";
      };

      clientId = lib.mkOption {
        type = lib.types.str;
        default = "aid";
        description = "The OIDC client id.";
      };

      allowEmails = lib.mkOption {
        type = lib.types.nonEmptyListOf lib.types.str;
        description = "Verified emails that may log in. A session runs commands on this host, so keep it short.";
      };

      environmentFile = lib.mkOption {
        type = lib.types.path;
        example = "/run/secrets/aid-web.env";
        description = "systemd EnvironmentFile with AID_OIDC_CLIENT_SECRET and AID_WEB_SESSION_SECRET.";
      };
    };
  };

  /**
    systemd unit settings both modules share.

    # Inputs

    `cfg`
    : The module's evaluated `services.aid` config
  */
  service = cfg: {
    path = [ cfg.pymuxPackage ] ++ cfg.extraPackages;
    serviceConfig = {
      Type = "exec";
      ExecStart = "${lib.getExe cfg.package} daemon";
      Restart = "on-failure";
      RestartSec = "2s";
      # SIGTERM reaches only the daemon, which stops its workers; stragglers get SIGKILL after the timeout.
      KillMode = "mixed";
      TimeoutStopSec = "30s";
    }
    // lib.optionalAttrs (cfg.environmentFile != null) { EnvironmentFile = cfg.environmentFile; };
  };

  /**
    systemd unit settings for `aid web`, which both modules share.

    # Inputs

    `cfg`
    : The module's evaluated `services.aid` config
  */
  webService = cfg: {
    serviceConfig = {
      Type = "exec";
      ExecStart = lib.escapeShellArgs (
        [
          (lib.getExe cfg.package)
          "web"
          "--bind"
          cfg.web.bind
          "--issuer"
          cfg.web.issuer
          "--client-id"
          cfg.web.clientId
        ]
        ++ lib.optionals (cfg.web.baseUrl != null) [
          "--base-url"
          cfg.web.baseUrl
        ]
        ++ lib.concatMap (email: [
          "--allow-email"
          email
        ]) cfg.web.allowEmails
      );
      EnvironmentFile = cfg.web.environmentFile;
      Restart = "on-failure";
      RestartSec = "2s";
    };
  };
}
