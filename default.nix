{
  pkgs ? import <nixpkgs> { },
}:
let
  aid = pkgs.python3Packages.callPackage ./aid { };
in
{
  inherit aid;

  nixosModules.default = ./nix/nixos.nix;
  homeModules.default = ./nix/home-manager.nix;

  shell = pkgs.mkShell {
    packages = [
      pkgs.pyright
      pkgs.ruff
      (pkgs.python3.withPackages (
        p:
        aid.dependencies
        ++ [
          p.pytest
          (p.mkPythonEditablePackage {
            pname = "aid";
            inherit (aid) version;
            root = toString ./src;
            scripts.aid = "aid.cli:main";
          })
        ]
      ))
    ];
  };
}
