{
  lib,
  buildPythonPackage,
  hatchling,
  agent-client-protocol,
  anyio,
  pydantic,
  pydantic-ai-slim,
  pyzmq,
  pytestCheckHook,
}:
let
  root = ../.;
in
buildPythonPackage {
  pname = "aid";
  inherit ((lib.importTOML (root + "/pyproject.toml")).project) version;
  pyproject = true;

  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      (root + "/pyproject.toml")
      (root + "/src")
      (root + "/tests")
    ];
  };

  build-system = [ hatchling ];

  dependencies = [
    agent-client-protocol
    anyio
    pydantic
    pydantic-ai-slim
    pyzmq
  ];

  nativeCheckInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "aid" ];

  meta = {
    description = "AI daemon: persistent ACP and pydantic-ai agents behind a Pythonic async API";
    mainProgram = "aid";
    platforms = lib.platforms.linux;
  };
}
