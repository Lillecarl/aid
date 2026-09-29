# pyedit (Lillecarl/pyedit) as a package of aid's set, from its own pyproject.toml, for `import pyedit` in workers.
# Its checks run in its own repository, not here.
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
  extra = rendered: {
    meta = rendered.meta // {
      description = "Scripted multi-file edits with dry-run diffs";
      homepage = "https://github.com/Lillecarl/pyedit";
      license = lib.licenses.asl20;
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
