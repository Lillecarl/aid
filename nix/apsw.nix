{
  fetchurl,
  unzip,
}:

/**
  apsw newer than nixpkgs has, from pyterm's `nix/python-pins.nix`. A
  `packageOverrides` function.

  # Type

  ```
  apsw :: PythonPackages -> PythonPackages -> AttrSet
  ```
*/
_pyFinal: pyPrev: {
  # Async connections arrived in 3.52; 3.54 fixes leaks and crashes in error handling.
  #
  # SQLite comes from apsw's own release, compiled into the extension: an apsw release is made for the
  # SQLite of the same number, and nixpkgs' `sqlite` is older. The amalgamation build declares every SQLite
  # symbol `static`, so the `libsqlite3.so.0` Python's own `sqlite3` loads cannot answer for it. The check
  # holds the two version numbers equal.
  apsw = pyPrev.apsw.overridePythonAttrs (old: rec {
    version = "3.54.1.0";
    src = fetchurl {
      url = "https://github.com/rogerbinns/apsw/releases/download/${version}/apsw-${version}.tar.gz";
      hash = "sha256-BDdfCM8d+AlJ7reZIK0c3eCOYXp/0+VEKkUk7xw/Lsg=";
    };
    amalgamation = fetchurl {
      url = "https://sqlite.org/2026/sqlite-amalgamation-3540100.zip";
      hash = "sha256-CpyONDUtrUYAaqgoNtU4O/+dxryc5aYFU8Baow5q0fk=";
    };
    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ unzip ];
    buildInputs = [ ];
    # `setup.py` builds against `sqlite3/sqlite3.c` when it is there.
    postPatch = (old.postPatch or "") + ''
      unzip -q ${amalgamation}
      mv sqlite-amalgamation-3540100 sqlite3
    '';
    checkPhase = ''
      runHook preCheck
      python -c '
      import apsw
      assert apsw.using_amalgamation, "apsw links a shared SQLite"
      assert apsw.apsw_version().startswith(apsw.sqlite_lib_version() + "."), (apsw.apsw_version(), apsw.sqlite_lib_version())
      '
      python -m apsw.tests
      runHook postCheck
    '';
  });
}
