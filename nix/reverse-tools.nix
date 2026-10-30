/**
  The programs reverse-engineering investigations run, for `AID_REVERSE_TOOLS` (docs/reverse-engineering.md):
  `$out/bin` leads each lead session's path. `aid-ghidra-python` is a Python with pyghidra, pointed at this Ghidra
  and the JDK Ghidra is built with (measured with Ghidra 12.1.4: pyghidra finds no Java on its own); it runs the
  plugin's Ghidra helper. No bare `python` here: it would shadow the session's own.
*/
{
  buildEnv,
  writeShellApplication,
  python3,
  ghidra,
  openjdk21,
  binwalk,
  unblob,
  acpica-tools,
  uefitool,
  rizin,
}:
let
  pyghidra = python3.withPackages (ps: [ ps.pyghidra ]);
  ghidraPython = writeShellApplication {
    name = "aid-ghidra-python";
    text = ''
      export JAVA_HOME=${openjdk21}/lib/openjdk GHIDRA_INSTALL_DIR=${ghidra}/lib/ghidra
      exec ${pyghidra}/bin/python "$@"
    '';
  };
in
buildEnv {
  name = "aid-reverse-tools";
  paths = [
    ghidraPython
    ghidra
    binwalk
    unblob
    acpica-tools
    uefitool
    rizin
  ];
}
