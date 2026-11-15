# aid-agent for a nixpkgs Python package set, beside aid-models: ../nix/python-overlay.nix names both.
# aid itself builds it with ./default.nix. Its httpx2 is whatever the set carries.
{
  lib,
  buildPythonPackage,
  hatchling,
  aid-models,
  anyio,
  httpx2,
  pydantic,
  structlog,
}:
buildPythonPackage {
  pname = "aid-agent";
  version = (lib.importTOML ./pyproject.toml).project.version;
  pyproject = true;
  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./README.md
      ./src
    ];
  };
  build-system = [ hatchling ];
  dependencies = [
    aid-models
    anyio
    httpx2
    pydantic
    structlog
  ];
  pythonImportsCheck = [ "aid_agent.model" ];
  meta = {
    description = "aid's agent loop as a library";
    homepage = "https://github.com/Lillecarl/aid";
    platforms = lib.platforms.unix;
  };
}
