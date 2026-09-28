{
  pkgs ? import <nixpkgs> { },
  # pyterm supplies the pyproject.nix builders and libpymux. Its default.nix takes our `pkgs` and imports no
  # nixpkgs of its own; do not set UMBRELLA_DEV, which points it at working copies this build does not have.
  pyterm ? builtins.fetchTree {
    type = "github";
    owner = "Lillecarl";
    repo = "pyterm";
    rev = "0b4f4361069327add97d0ffcf8445eacb8640170";
    narHash = "sha256-SgK3W1nAUJY9Cvt3rcVqkV4ArE44zxm0/yzdnZo8Nio=";
  },
}:
let
  inherit (pkgs) lib;
  p = import pyterm { inherit pkgs; };
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
        };
      }
    );
  };
in
{
  inherit set;

  aid = set.mkVirtualEnv "aid" { aid = [ ]; } // {
    inherit (set.aid) meta;
  };

  inherit (set.aid) tests;

  # From the same pyterm pin as libpymux: aid starts its own server with it, and the wire protocol still moves.
  inherit (p) pymux;

  nixosModules.default = ./nix/nixos.nix;
  homeModules.default = ./nix/home-manager.nix;

  shell = pkgs.mkShell {
    packages = [
      (set.mkVirtualEnv "aid-dev" { aid = [ "test" ]; })
      p.pymux
      pkgs.pyright
      pkgs.ruff
    ];
    # The working copy, ahead of the aid the venv carries.
    shellHook = ''
      export PYTHONPATH=${lib.escapeShellArg (toString ./src)}''${PYTHONPATH:+:$PYTHONPATH}
    '';
  };
}
