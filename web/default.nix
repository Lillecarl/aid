# The Svelte UI, built to the directory `aid web` serves (AID_WEB_ASSETS). svelte-check runs as the check phase.
{
  lib,
  buildNpmPackage,
  nodejs_24,
  # `<pymux-pane>`, npm-shaped, from the pyterm pin: linked in as node_modules/pymux-pane.
  pymuxElement,
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
      ./public
    ];
  };

  # Changes with package-lock.json: set lib.fakeHash, build, and take the hash the error prints.
  npmDepsHash = "sha256-N0OIbXPs3lzYBe+uFTZBWcB8vvrtP0aXMrNNQRiFwAg=";

  # After npm's own install, which would drop a package the lock does not name.
  preBuild = ''
    ln -s ${pymuxElement} node_modules/pymux-pane
  '';

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    # protocol.ts is generated from protocol.schema.json: a hand edit, or a schema not regenerated from, fails here.
    npm run types:check
    npm run check
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    cp -r dist $out
    runHook postInstall
  '';
}
