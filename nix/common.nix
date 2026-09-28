{ lib, pkgs }:
{
  options = {
    enable = lib.mkEnableOption "aid, the AI daemon that keeps ACP and pydantic-ai agents running";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.python3Packages.callPackage ../aid { };
      defaultText = lib.literalExpression "pkgs.python3Packages.callPackage <aid/aid> { }";
      description = "The aid package.";
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
  };

  /**
    systemd unit settings both modules share.

    # Inputs

    `cfg`
    : The module's evaluated `services.aid` config
  */
  service = cfg: {
    path = cfg.extraPackages;
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
}
