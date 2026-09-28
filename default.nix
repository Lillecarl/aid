{
  pkgs ? import <nixpkgs> { },
  # pyterm supplies the pyproject.nix builders and libpymux. Its default.nix takes our `pkgs` and imports no
  # nixpkgs of its own; do not set UMBRELLA_DEV, which points it at working copies this build does not have.
  pyterm ? builtins.fetchTree {
    type = "github";
    owner = "Lillecarl";
    repo = "pyterm";
    rev = "c44c100de8b294b50d9df62e6b5001633c66e579";
    narHash = "sha256-spzzALWz+FRnIGKEnBGBzYu4Q5QVAtD5MrxzBKq70Q8=";
  },
}:
let
  inherit (pkgs) lib;
  p = import pyterm { inherit pkgs; };
  venv = set.mkVirtualEnv "aid-env" { aid = [ ]; };
  ui = pkgs.callPackage ./web { };
  python = pkgs.python3;

  set = p.mkPythonSet {
    inherit python;
    nixpkgsRoots = p.nixpkgsRootsFor {
      inherit python;
      projectRoots = p.projectRoots ++ [ ./. ];
      exclude = p.suppliedNames ++ [ "aid" ];
    };
    overlay = lib.composeExtensions p.overlay (
      final: _prev: {
        aid = final.callPackage ./aid {
          inherit (p) mkProject;
          pymuxApp = p.pymux;
          dex = pkgs.dex-oidc;
          webUi = ui;
        };
      }
    );
  };
in
{
  inherit set ui;

  # Only `bin/aid`: a profile that installs this next to another virtualenv, such as pymux's, would otherwise
  # get two `bin/python` and `bin/activate` and refuse to build. The wrapper points `aid web` at the built UI.
  aid =
    pkgs.runCommand "aid-${set.aid.version}"
      {
        inherit (set.aid) meta;
        passthru = { inherit venv ui; };
        nativeBuildInputs = [ pkgs.makeWrapper ];
      }
      ''
        makeWrapper ${venv}/bin/aid $out/bin/aid --set-default AID_WEB_ASSETS ${ui}
      '';

  inherit (set.aid) tests;

  # From the same pyterm pin as libpymux: aid starts its own server with it, and the wire protocol still moves.
  inherit (p) pymux;

  nixosModules.default = ./nix/nixos.nix;
  homeModules.default = ./nix/home-manager.nix;

  shell = pkgs.mkShell {
    packages = [
      (set.mkVirtualEnv "aid-dev" { aid = [ "test" ]; })
      p.pymux
      pkgs.dex-oidc
      pkgs.nodejs_24
      pkgs.pyright
      pkgs.ruff
    ];
    # The working copy, ahead of the aid the venv carries. The UI is the built one; `npm run dev` in web/
    # serves the working copy instead.
    shellHook = ''
      export PYTHONPATH=${lib.escapeShellArg (toString ./src)}''${PYTHONPATH:+:$PYTHONPATH}
      export AID_WEB_ASSETS=${ui}
    '';
  };
}
