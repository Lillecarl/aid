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
  root = /. + builtins.unsafeDiscardStringContext src.outPath;
  inherit python;
  packages = [ "litelm" ];
  extra = rendered: {
    patches = [ ./litelm-anthropic-stream.patch ];
    meta = rendered.meta // {
      description = "Lightweight model transport and provider translation";
      homepage = "https://github.com/kennethwolters/litelm";
      license = lib.licenses.mit;
    };
  };
})
  {
    inherit stdenv pyprojectHook resolveBuildSystem;
  }
