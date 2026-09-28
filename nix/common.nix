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

    agentsPath = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "/home/me/aid-agents" ];
      description = ''
        Directories whose modules define `aid.PydanticAgent` subclasses; they become AID_AGENTS_PATH. Empty
        leaves aid's default, `$XDG_CONFIG_HOME/aid/agents`. Changes to the modules need no restart.
      '';
    };

    environmentFile = lib.mkOption {
      # A string, not a path: systemd specifiers such as %t (the runtime directory) are what a user unit
      # needs to reach home-manager agenix's secrets, and a path type rejects them.
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "%t/agenix/aid-env";
      description = ''
        systemd EnvironmentFile for secrets such as DEEPSEEK_API_KEY or ANTHROPIC_API_KEY; every agent inherits
        them. Kept out of the store. systemd specifiers work.
      '';
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
        type = lib.types.str;
        example = "/run/secrets/aid-web.env";
        description = ''
          systemd EnvironmentFile with AID_OIDC_CLIENT_SECRET and AID_WEB_SESSION_SECRET. systemd specifiers work.
        '';
      };

      speechModel = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        example = lib.literalExpression "config.services.aid.package.speechModel";
        description = ''
          A sherpa-onnx streaming transducer for speech to text in the web UI. The package's `speechModel` is an
          English one. Null turns speech to text off, and the page shows no microphone button.
        '';
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
    environment =
      lib.optionalAttrs (cfg.agentsPath != [ ]) { AID_AGENTS_PATH = lib.concatStringsSep ":" cfg.agentsPath; }
      // cfg.environment;
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
        ++ lib.optionals (cfg.web.speechModel != null) [
          "--speech-model"
          "${cfg.web.speechModel}"
        ]
      );
      EnvironmentFile = cfg.web.environmentFile;
      Restart = "on-failure";
      RestartSec = "2s";
    };
  };
}
