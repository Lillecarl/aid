# aid-models as a pyproject.nix builders package, in aid's set (../default.nix), on the pattern of ../pyrun.
# package.nix builds the same project for a nixpkgs Python package set, for consumers outside aid.
{
  lib,
  python,
  stdenv,
  pyprojectHook,
  resolveBuildSystem,
  mkProject,
}:
(mkProject {
  root = ./.;
  inherit python;
  extra = rendered: {
    meta = rendered.meta // {
      description = "aid's model clients as a library: embeddings, reranking, prices";
      homepage = "https://github.com/Lillecarl/aid";
      platforms = lib.platforms.unix;
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
