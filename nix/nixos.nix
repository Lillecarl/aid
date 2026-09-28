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
  runtimeDir = "/run/aid";
in
{
  options.services.aid = common.options // {
    user = lib.mkOption {
      type = lib.types.str;
      default = "aid";
      description = "User the daemon and its agents run as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "aid";
      description = "Group whose members may talk to the daemon.";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];
    environment.variables.AID_RUNTIME_DIR = runtimeDir;

    users.users = lib.mkIf (cfg.user == "aid") {
      aid = {
        isSystemUser = true;
        inherit (cfg) group;
        home = "/var/lib/aid";
      };
    };
    users.groups = lib.mkIf (cfg.group == "aid") { aid = { }; };

    systemd.services.aid = {
      description = "aid, the AI daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      inherit (unit) path;
      environment = {
        AID_RUNTIME_DIR = runtimeDir;
        AID_STATE_DIR = "/var/lib/aid";
        HOME = "/var/lib/aid";
      }
      // cfg.environment;
      serviceConfig = unit.serviceConfig // {
        User = cfg.user;
        Group = cfg.group;
        StateDirectory = "aid";
        StateDirectoryMode = "0750";
        RuntimeDirectory = "aid";
        RuntimeDirectoryMode = "0750";
        # The zmq ipc sockets get their mode from the umask; the group must be able to connect.
        UMask = "0007";
        NoNewPrivileges = true;
      };
    };

    systemd.services.aid-web = lib.mkIf cfg.web.enable {
      description = "aid web UI";
      wantedBy = [ "multi-user.target" ];
      after = [
        "aid.service"
        "network-online.target"
      ];
      wants = [
        "aid.service"
        "network-online.target"
      ];
      environment.AID_RUNTIME_DIR = runtimeDir;
      serviceConfig = (common.webService cfg).serviceConfig // {
        User = cfg.user;
        Group = cfg.group;
        NoNewPrivileges = true;
      };
    };
  };
}
