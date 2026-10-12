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
  # shellous, the asyncio subprocess library: v0.42.0, the latest release.
  shellous ? builtins.fetchTree {
    type = "github";
    owner = "byllyfish";
    repo = "shellous";
    rev = "fc6f1b51f1cc1ef35d492787535ae120296d3dd5";
    narHash = "sha256-Sr7o0oXreN5ax+HDHaE3t8oXhc7pH/uj1esw5Q0/8iw=";
  },
  # xonsh, the Python-powered shell: 0.24.2, the latest release.
  xonsh ? builtins.fetchTree {
    type = "github";
    owner = "xonsh";
    repo = "xonsh";
    rev = "d6a344b758ff1bf4fca8e9d32a61dd6b3b4b6254";
    narHash = "sha256-6dLl2VDUyfoFVbnSpDxXgEhp+GpYab3yewMuq6Nd6oQ=";
  },
  # Vivarium, the NixOS integration tests aid boots in UML guests. A local path, not a pin: one-man
  # project, and the checkout beside this one is what the tests run against.
  vivariumSrc ? ../nixidae/vivarium,
  litelm ? builtins.fetchTree {
    type = "github";
    owner = "kennethwolters";
    repo = "litelm";
    rev = "3cae20176bbda16633da7238a75d33871abea189";
    narHash = "sha256-1BGppa4gN3iVA8lDpCV5A8+bXv1kq8oG2U4GVFkqqrY=";
  },
  openai ? builtins.fetchTree {
    type = "github";
    owner = "openai";
    repo = "openai-python";
    rev = "03b3ec474241fe3a73c4818dc6b886da28faaf91";
    narHash = "sha256-SdoZqL65ptS+VTo18bETJWmTn4/sPWqtFVLoaFf4AYo=";
  },
  anthropic ? builtins.fetchTree {
    type = "github";
    owner = "anthropics";
    repo = "anthropic-sdk-python";
    rev = "eb21a4352015686c30f5759e8c2f02d70f5371e2";
    narHash = "sha256-DdXmpunIxLL3kCAQ1RrgLskfkyDtzfBqjMMPwWtX2wQ=";
  },
  httpx2 ? builtins.fetchTree {
    type = "github";
    owner = "pydantic";
    repo = "httpx2";
    rev = "71ae23be5448f859c2b4e21d9972ddfa7b8d759d";
    narHash = "sha256-mEJHlG6skKkV90SfjTg+MHGkL7gnf3QLKLbWlTtEb9Q=";
  },
}:
let
  inherit (pkgs) lib;
  p = import pyterm { inherit pkgs; };
  venv = set.mkVirtualEnv "aid-env" { aid = [ ]; };
  # One environment for the dev shell and the pyrefly wrapper: its python carries every dependency,
  # so both resolve the same third-party imports.
  devEnv = set.mkVirtualEnv "aid-dev" {
    aid = [ "test" ];
    pyrun = [ "test" ];
  };
  ui = pkgs.callPackage ./web { pymuxElement = p.pymux-element; };
  speechModel = pkgs.callPackage ./nix/speech-model.nix { };
  grammars = pkgs.callPackage ./nix/tree-sitter-grammars.nix { };
  # The browsers the browser tests drive: chromium's headless shell for the sandbox run, firefox for a
  # developer's `--browser firefox` run. Fixed-output fetches, so the sandbox may use them.
  browsers = pkgs.playwright-driver.browsers.override {
    withChromium = false;
    withWebkit = false;
    withFfmpeg = false;
  };
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
    packageOverrides = final: prev: {
      pyzmq = prev.pyzmq.override { inherit zeromq; };
      dunamai = prev.dunamai.overridePythonAttrs {
        version = "1.26.0";
        src = builtins.fetchTree {
          type = "github";
          owner = "mtkennerly";
          repo = "dunamai";
          rev = "70939d32b2da9816f70fcddef544e54358135c30";
          narHash = "sha256-YzLE0+/ZOROhWDVSBjWGdGmmMuMHOKP69zMnT8MKSvc=";
        };
      };
      uv-dynamic-versioning = prev.uv-dynamic-versioning.overridePythonAttrs {
        version = "0.14.0";
        src = builtins.fetchTree {
          type = "github";
          owner = "ninoseki";
          repo = "uv-dynamic-versioning";
          rev = "46cb92b53a9b822cee62572a0a14aa0be7cd045d";
          narHash = "sha256-4hfk9A4ZuiO8gevb5U6pTWkD5YKqk7J9I5eZ+KkfYX0=";
        };
        UV_DYNAMIC_VERSIONING_BYPASS = "0.14.0";
        preCheck = prev.uv-dynamic-versioning.preCheck + ''
          git init
          git add .
          git commit -m 'Source fixture for version tests'
          unset UV_DYNAMIC_VERSIONING_BYPASS
        '';
      };
      httpcore2 = prev.httpcore2.overridePythonAttrs {
        version = "2.12.0";
        src = httpx2;
        dependencies = [
          final.truststore
          final.h11
        ];
        UV_DYNAMIC_VERSIONING_BYPASS = "2.12.0";
      };
      httpx2 = prev.httpx2.overridePythonAttrs {
        version = "2.12.0";
        src = httpx2;
        dependencies = [
          final.truststore
          final.httpcore2
          final.anyio
          final.idna
        ];
        UV_DYNAMIC_VERSIONING_BYPASS = "2.12.0";
      };
    };
  };

  set = p.mkPythonSet {
    inherit python;
    nixpkgsRoots =
      p.nixpkgsRootsFor {
        inherit python;
        # xonsh declares no runtime dependencies, but its dev and doc extras name three packages this
        # nixpkgs does not ship: nothing installs them (the renderer enables no extras), so they join exclude
        # instead of failing the closure computation.
        projectRoots = p.projectRoots ++ [
          ./.
          ./pyrun
          pyedit
          shellous
          xonsh
          litelm
          openai
          anthropic
        ];
        exclude =
          p.suppliedNames
          ++ [
            "aid"
            "pyrun"
            "pyedit"
            "shellous"
            "litelm"
            "openai"
            "anthropic"
            "pre-commit"
            "re-ver"
            "runthis-sphinxext"
          ]
          ++ grammarsByName;
      }
      ++ map (name: python.pkgs.tree-sitter-grammars.${name}) grammarsByName;
    overlay = lib.composeExtensions p.overlay (
      final: _prev: {
        openai = final.callPackage ./nix/model-sdk.nix {
          inherit (p) mkProject;
          src = openai;
        };
        anthropic = final.callPackage ./nix/model-sdk.nix {
          inherit (p) mkProject;
          src = anthropic;
        };
        litelm = final.callPackage ./nix/litelm.nix {
          inherit (p) mkProject;
          src = litelm;
        };
        pyedit = final.callPackage ./nix/pyedit.nix {
          inherit (p) mkProject;
          src = pyedit;
        };
        shellous = final.callPackage ./nix/shellous.nix {
          inherit (p) mkProject;
          src = shellous;
        };
        xonsh = final.callPackage ./nix/xonsh.nix {
          inherit (p) mkProject;
          src = xonsh;
          ply = final.ply;
          prompt_toolkit = final."prompt-toolkit";
          pygments = final.pygments;
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
          inherit speechModel grammars browsers;
          inherit (pkgs) cacert;
          testArgs = lib.filter (arg: arg != "") (lib.splitString " " (builtins.getEnv "AID_TEST_ARGS"));
        };
      }
    );
  };
in
{
  inherit
    set
    ui
    speechModel
    grammars
    ;

  schema = pkgs.writeShellApplication {
    name = "aid-schema";
    runtimeInputs = [ devEnv ];
    text = ''
      export PYTHONPATH=${lib.escapeShellArg (toString ./src)}:${lib.escapeShellArg (toString ./pyrun/src)}
      python -m aid.schema
    '';
  };

  # The browsers the browser tests drive, here so the vivarium guest takes the same ones as the shell.
  inherit browsers;

  # Only `bin/aid`: a profile that installs this next to another virtualenv, such as pymux's, would otherwise
  # get two `bin/python` and `bin/activate` and refuse to build. The wrapper points `aid web` at the built UI
  # and the highlighter's grammars. In-tree plugins ride along as the plugins path's last tier, after the
  # caller's AID_PLUGINS_PATH and their XDG directory.
  aid =
    pkgs.runCommand "aid-${set.aid.version}"
      {
        inherit (set.aid) meta;
        passthru = {
          inherit
            venv
            ui
            speechModel
            grammars
            ;
        };
        nativeBuildInputs = [ pkgs.makeWrapper ];
      }
      ''
        mkdir -p $out/share/aid
        cp -r ${./plugins} $out/share/aid/plugins
        makeWrapper ${venv}/bin/aid $out/bin/aid --set-default AID_WEB_ASSETS ${ui} \
          --set-default AID_TREE_SITTER_GRAMMARS ${grammars} \
          --set-default AID_PLUGINS_DIST $out/share/aid/plugins
      '';

  inherit (set.aid) tests;
  pyrun-tests = set.pyrun.tests;

  ruff = pkgs.writeShellApplication {
    name = "ruff";
    runtimeInputs = [
      pkgs.ruff
    ];
    text = # bash
      ''
        # SET ENV VARS AND SHIT HERE
        ruff "$@"
      '';
  };
  pyrefly = pkgs.writeShellApplication {
    name = "pyrefly";
    runtimeInputs = [
      pkgs.pyrefly
      devEnv
    ];
    text = # bash
      ''
        # pyrefly resolves imports through PYTHONPATH: point it at the working copy, as the dev shell does.
        export PYTHONPATH=${lib.escapeShellArg (toString ./src)}:${lib.escapeShellArg (toString ./pyrun/src)}''${PYTHONPATH:+:$PYTHONPATH}
        pyrefly "$@"
      '';
  };
  nixpkgs-fmt = pkgs.writeShellApplication {
    name = "nixpkgs-fmt";
    runtimeInputs = [
      pkgs.nixpkgs-fmt
    ];
    text = # bash
      ''
        nixpkgs-fmt **/*.nix
      '';
  };

  # One UML guest running dex, the daemon and the web UI as the NixOS module ships them. The UML
  # backend needs no KVM and no root, so this runs in the Nix sandbox like `tests` does.
  vivarium-smoke = pkgs.callPackage ./nix/vivarium-smoke.nix {
    vivarium = import (vivariumSrc + "/lib.nix") { inherit pkgs; };
  };

  # From the same pyterm pin as libpymux: aid starts its own server with it, and the wire protocol still moves.
  inherit (p) pymux;

  nixosModules.default = ./nix/nixos.nix;
  homeModules.default = ./nix/home-manager.nix;

  shell = pkgs.mkShell {
    packages = [
      devEnv
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
      export PLAYWRIGHT_BROWSERS_PATH=${browsers}
      # The element the web build links in; `npm ci` in web/ removes it, and the next shell puts it back.
      mkdir -p ${lib.escapeShellArg (toString ./web)}/node_modules
      ln -sfn ${p.pymux-element} ${lib.escapeShellArg (toString ./web)}/node_modules/pymux-pane
    '';
  };
}
