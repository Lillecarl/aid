{
  pkgs ? import <nixpkgs> { },
  # pyterm supplies the pyproject.nix builders and libpymux. Its default.nix takes our `pkgs` and imports no
  # nixpkgs of its own; do not set UMBRELLA_DEV, which points it at working copies this build does not have.
  pyterm ? builtins.fetchTree {
    type = "github";
    owner = "Lillecarl";
    repo = "pyterm";
    rev = "096294d637c8781c65c17cbdbbba84a71e3685d2";
    narHash = "sha256-nyTfy4DQyg7wMGW0M6DNwKc0PmwZV6LB/Vh/KeBPKmk=";
  },
  # aid's agents edit through pyedit's library (`aid.coding`); the same pin as croshome's.
  pyedit ? builtins.fetchTree {
    type = "github";
    owner = "Lillecarl";
    repo = "pyedit";
    rev = "0910a2436bcbbbef6889be7c10d96520593e9578";
    narHash = "sha256-AO0LvZeEB/kUNxym2tgpXOU2FpEq3kUWNTnUP8rHBus=";
  },
}:
let
  inherit (pkgs) lib;
  p = import pyterm { inherit pkgs; };
  venv = set.mkVirtualEnv "aid-env" { aid = [ ]; };
  ui = pkgs.callPackage ./web { pymuxElement = p.pymux-element; };
  speechModel = pkgs.callPackage ./nix/speech-model.nix { };
  grammars = pkgs.callPackage ./nix/tree-sitter-grammars.nix { };
  # Drafts turn on libzmq's ws:// transport (Lillecarl/aid#1): remote workers reach the daemon through an HTTP
  # reverse proxy. GnuTLS adds wss://, for a worker whose proxy speaks only HTTPS.
  # Propagated: libzmq.pc names gnutls in Requires.private, and pyzmq links what pkg-config reports.
  zeromq = (pkgs.zeromq.override { enableDrafts = true; }).overrideAttrs (old: {
    propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pkgs.gnutls ];
  });
  # pyedit's grammar bindings, taken where its own build takes them: some are missing from nixpkgs' top level.
  grammarsByName = map (lang: "tree-sitter-${lang}") [
    "bash"
    "c"
    "cpp"
    "go"
    "java"
    "javascript"
    "json"
    "lua"
    "nix"
    "python"
    "ruby"
    "rust"
    "toml"
    "tsx"
    "typescript"
    "yaml"
    "zig"
  ];
  python = pkgs.python3.override {
    self = python;
    packageOverrides = _final: prev: { pyzmq = prev.pyzmq.override { inherit zeromq; }; };
  };

  set = p.mkPythonSet {
    inherit python;
    nixpkgsRoots = p.nixpkgsRootsFor {
      inherit python;
      projectRoots = p.projectRoots ++ [
        ./.
        ./pyrun
        pyedit
      ];
      exclude = p.suppliedNames ++ [
        "aid"
        "pyrun"
        "pyedit"
      ]
      ++ grammarsByName;
    }
    ++ map (name: python.pkgs.tree-sitter-grammars.${name}) grammarsByName;
    overlay = lib.composeExtensions p.overlay (
      final: _prev: {
        pyedit = final.callPackage ./nix/pyedit.nix {
          inherit (p) mkProject;
          src = pyedit;
        };
        pyrun = final.callPackage ./pyrun {
          inherit (p) mkProject;
          utilLinux = pkgs.util-linux;
        };
        aid = final.callPackage ./aid {
          inherit (p) mkProject;
          pymuxApp = p.pymux;
          dex = pkgs.dex-oidc;
          webUi = ui;
          inherit speechModel grammars;
          inherit (pkgs) cacert;
        };
      }
    );
  };
in
{
  inherit set ui speechModel grammars;

  # Only `bin/aid`: a profile that installs this next to another virtualenv, such as pymux's, would otherwise
  # get two `bin/python` and `bin/activate` and refuse to build. The wrapper points `aid web` at the built UI
  # and the highlighter's grammars.
  aid =
    pkgs.runCommand "aid-${set.aid.version}"
      {
        inherit (set.aid) meta;
        passthru = { inherit venv ui speechModel grammars; };
        nativeBuildInputs = [ pkgs.makeWrapper ];
      }
      ''
        makeWrapper ${venv}/bin/aid $out/bin/aid --set-default AID_WEB_ASSETS ${ui} \
          --set-default AID_TREE_SITTER_GRAMMARS ${grammars}
      '';

  inherit (set.aid) tests;
  pyrun-tests = set.pyrun.tests;

  # From the same pyterm pin as libpymux: aid starts its own server with it, and the wire protocol still moves.
  inherit (p) pymux;

  nixosModules.default = ./nix/nixos.nix;
  homeModules.default = ./nix/home-manager.nix;

  shell = pkgs.mkShell {
    packages = [
      (set.mkVirtualEnv "aid-dev" {
        aid = [ "test" ];
        pyrun = [ "test" ];
      })
      p.pymux
      pkgs.dex-oidc
      pkgs.nodejs_24
      pkgs.pyright
      pkgs.ruff
    ];
    # The working copy, ahead of the aid the venv carries. The UI is the built one; `npm run dev` in web/
    # serves the working copy instead.
    shellHook = ''
      export PYTHONPATH=${lib.escapeShellArg (toString ./src)}:${lib.escapeShellArg (toString ./pyrun/src)}''${PYTHONPATH:+:$PYTHONPATH}
      export AID_WEB_ASSETS=${ui}
      export AID_TEST_SPEECH_MODEL=${speechModel}
      export AID_TREE_SITTER_GRAMMARS=${grammars}
      # The element the web build links in; `npm ci` in web/ removes it, and the next shell puts it back.
      mkdir -p ${lib.escapeShellArg (toString ./web)}/node_modules
      ln -sfn ${p.pymux-element} ${lib.escapeShellArg (toString ./web)}/node_modules/pymux-pane
    '';
  };
}
