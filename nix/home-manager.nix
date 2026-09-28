{
  config,
  lib,
  pkgs,
  ...
}:
let
  common = import ./common.nix { inherit lib pkgs; };
  cfg = config.services.aid;
  unit = common.service cfg;
in
{
  options.services.aid = common.options;

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];

    systemd.user.services.aid = {
      Unit.Description = "aid, the AI daemon";
      Install.WantedBy = [ "default.target" ];
      Service = unit.serviceConfig // {
        Environment = lib.mapAttrsToList (name: value: "${name}=${value}") (
          {
            PATH = lib.concatStringsSep ":" [
              (lib.makeBinPath unit.path)
              "${config.home.profileDirectory}/bin"
              "/run/current-system/sw/bin"
            ];
          }
          // unit.environment
        );
      };
    };

    systemd.user.services.aid-web = lib.mkIf cfg.web.enable {
      Unit = {
        Description = "aid web UI";
        After = [ "aid.service" ];
        Wants = [ "aid.service" ];
      };
      Install.WantedBy = [ "default.target" ];
      Service = (common.webService cfg).serviceConfig;
    };
  };
}
