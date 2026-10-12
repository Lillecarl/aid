{
  lib,
  python,
  stdenv,
  pyprojectHook,
  resolveBuildSystem,
  mkProject,
  src,
}:
let
  root = /. + builtins.unsafeDiscardStringContext src.outPath;
in
(mkProject {
  inherit root python;
  extra = _rendered: {
    src = lib.fileset.toSource {
      inherit root;
      fileset = lib.fileset.unions [
        (root + "/src")
        (root + "/pyproject.toml")
        (root + "/README.md")
        (root + "/LICENSE")
      ];
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
