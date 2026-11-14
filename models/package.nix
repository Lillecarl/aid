# aid-models for a nixpkgs Python package set:
#   python3Packages.callPackage (aid + "/models/package.nix") { }
# aid itself builds it with ./default.nix. Its httpx2 is whatever the set carries.
{
  lib,
  buildPythonPackage,
  hatchling,
  anyio,
  httpx2,
  pydantic,
  structlog,
}:
buildPythonPackage {
  pname = "aid-models";
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
    anyio
    httpx2
    pydantic
    structlog
  ];
  pythonImportsCheck = [ "aid_models.embedding" ];
  meta = {
    description = "aid's model clients as a library: embeddings, reranking, prices";
    homepage = "https://github.com/Lillecarl/aid";
    platforms = lib.platforms.unix;
  };
}
