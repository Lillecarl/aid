# The Svelte UI, built to the directory `aid web` serves (AID_WEB_ASSETS). svelte-check runs as the check phase.
{
  lib,
  buildNpmPackage,
  nodejs_24,
}:
buildNpmPackage {
  pname = "aid-web";
  version = (lib.importJSON ./package.json).version;
  nodejs = nodejs_24;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./package.json
      ./package-lock.json
      ./index.html
      ./svelte.config.js
      ./tsconfig.json
      ./vite.config.ts
      ./src
    ];
  };

  # Changes with package-lock.json: set lib.fakeHash, build, and take the hash the error prints.
  npmDepsHash = "sha256-yJn1JVqNiZuR5IXDHTZNtjqdF1sYoiHKzQRgWxNzmho=";

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    npm run check
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    cp -r dist $out
    runHook postInstall
  '';
}
