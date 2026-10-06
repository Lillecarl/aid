# shellous (byllyfish/shellous) as a package of aid's set, from its own pyproject.toml, for `import
# shellous` wherever aid runs subprocesses. Its checks run in its own repository, not here.
{
  lib,
  python,
  stdenv,
  pyprojectHook,
  resolveBuildSystem,
  mkProject,
  src,
}:
(mkProject {
  # A path, as lib.fileset requires. A fetched tree is a string with store context, which a path cannot take;
  # the tree is already in the store when this evaluates, so the context carries nothing.
  root = /. + builtins.unsafeDiscardStringContext src.outPath;
  inherit python;
  # uv_build, so none of hatchling's `[tool.hatch.build.targets.wheel]` to read: the package directory itself.
  packages = [ "shellous" ];
  extra = rendered: {
    meta = rendered.meta // {
      description = "Async Processes and Pipelines";
      homepage = "https://github.com/byllyfish/shellous";
      license = lib.licenses.asl20;
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
