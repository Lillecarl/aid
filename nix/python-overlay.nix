# aid's libraries for a nixpkgs Python package set, as a packageOverrides overlay:
#   python3.override { packageOverrides = import (aid + "/nix/python-overlay.nix"); }
# aid itself builds them with ../models/default.nix and ../agent/default.nix.
final: _prev: {
  aid-models = final.callPackage ../models/package.nix { };
  aid-agent = final.callPackage ../agent/package.nix { };
}
