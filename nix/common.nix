{ lib, pkgs }:
let
  aid = import ../. { inherit pkgs; };
in
{
  options = {
    enable = lib.mkEnableOption "aid, the AI daemon that keeps ACP, native and interactive Claude agents running";

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

    pluginsPath = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "/home/me/aid-plugins" ];
      description = ''
        Directories of plugins (agents, instructions, tools, skills, daemon tasks), searched before
        `$XDG_CONFIG_HOME/aid/plugins` and the bundled plugins; they become AID_PLUGINS_PATH. Changes to agents,
        instructions and tools need no restart; a daemon task loads at daemon start.
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

    allowWorkerCommand = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Let sessions name their own worker command (`aid daemon --allow-worker-command`). A command runs as the
        daemon with the session's worker credentials: allow it only where session creators are trusted.
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

      theme = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "pygments:catppuccin-mocha";
        description = ''
          The web UI's colours, as pymux's `theme` client option names them: `pygments:<name>` or `base16:<name>`.
          Null keeps the browser's light or dark.
        '';
      };
    };

    telegram = {
      enable = lib.mkEnableOption "the aid Telegram plugin: settled turn ends as messages, basic commands back";

      environmentFile = lib.mkOption {
        type = lib.types.str;
        example = "%t/agenix/aid-env";
        description = ''
          systemd EnvironmentFile with AID_TELEGRAM_TOKEN and AID_TELEGRAM_CHATS. Kept out of the store.
          systemd specifiers work.
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
      lib.optionalAttrs (cfg.pluginsPath != [ ]) { AID_PLUGINS_PATH = lib.concatStringsSep ":" cfg.pluginsPath; }
      // cfg.environment;
    serviceConfig = {
      Type = "exec";
      ExecStart = "${lib.getExe cfg.package} daemon" + lib.optionalString cfg.allowWorkerCommand " --allow-worker-command";
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
        ++ lib.optionals (cfg.web.theme != null) [
          "--theme"
          cfg.web.theme
        ]
      );
      EnvironmentFile = cfg.web.environmentFile;
      Restart = "on-failure";
      RestartSec = "2s";
    };
  };

  /**
    systemd unit settings for `aid plugin run telegram`, which both modules share.

    # Inputs

    `cfg`
    : The module's evaluated `services.aid` config
  */
  telegramService = cfg: {
    serviceConfig = {
      Type = "exec";
      ExecStart = "${lib.getExe cfg.package} plugin run telegram";
      EnvironmentFile = cfg.telegram.environmentFile;
      # The plugin exits loudly on a dead daemon, a refused token and a silent one alike: always come back,
      # and pick up a provisioned token without anyone restarting the unit.
      Restart = "always";
      RestartSec = "10s";
    };
  };
}
