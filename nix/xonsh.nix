# xonsh (xonsh/xonsh) as a package of aid's set, from its own pyproject.toml, for the `xonsh` mode of
# aid's python tool. Its checks run in its own repository, not here.
{
  lib,
  python,
  stdenv,
  pyprojectHook,
  resolveBuildSystem,
  mkProject,
  src,
  ply,
  prompt_toolkit,
  pygments,
}:
(mkProject {
  # A path, as lib.fileset requires. A fetched tree is a string with store context, which a path cannot take;
  # the tree is already in the store when this evaluates, so the context carries nothing.
  root = /. + builtins.unsafeDiscardStringContext src.outPath;
  inherit python;
  # setuptools, so none of hatchling's `[tool.hatch.build.targets.wheel]` to read: the package directories
  # themselves, as `[tool.setuptools] packages` names them.
  packages = [
    "xonsh"
    "xontrib"
    "xompletions"
  ];
  extra = rendered: {
    # Runtime dependencies the project leaves to extras (nixpkgs promotes the same three).
    propagatedBuildInputs = (rendered.propagatedBuildInputs or [ ]) ++ [
      ply
      prompt_toolkit
      pygments
    ];
    meta = rendered.meta // {
      description = "Python-powered shell";
      homepage = "https://github.com/xonsh/xonsh";
      license = lib.licenses.bsd2;
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
