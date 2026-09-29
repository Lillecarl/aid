/**
  The tree-sitter grammars `aid web` highlights files with, for `AID_TREE_SITTER_GRAMMARS`.

  `$out/<name>/parser` is the grammar, `$out/<name>/highlights.scm` its queries joined in the order its
  tree-sitter.json lists them, and `$out/languages.json` maps each name to its C symbol and file types. A file
  type is an extension or a whole file name, as in tree-sitter.json.
*/
{
  lib,
  runCommand,
  tree-sitter-grammars,
}:
let
  g = tree-sitter-grammars;
  queries = grammar: file: "${grammar}/queries/${file}";
  # typescript and tsx install no queries; their source has them, and they build on javascript's.
  typescriptSrc = g.tree-sitter-typescript.src;
  js = queries g.tree-sitter-javascript;

  languages = {
    bash = {
      grammar = g.tree-sitter-bash;
      types = [ "sh" "bash" ".bashrc" ".bash_profile" ".envrc" ];
    };
    c = {
      grammar = g.tree-sitter-c;
      types = [ "c" "h" ];
    };
    cpp = {
      grammar = g.tree-sitter-cpp;
      types = [ "cc" "cpp" "cxx" "hpp" "hxx" "hh" ];
      highlights = [
        (queries g.tree-sitter-c "highlights.scm")
        (queries g.tree-sitter-cpp "highlights.scm")
      ];
    };
    css = {
      grammar = g.tree-sitter-css;
      types = [ "css" ];
    };
    diff = {
      grammar = g.tree-sitter-diff;
      types = [ "diff" "patch" ];
    };
    dockerfile = {
      grammar = g.tree-sitter-dockerfile;
      types = [ "Dockerfile" "Containerfile" "dockerfile" ];
    };
    go = {
      grammar = g.tree-sitter-go;
      types = [ "go" ];
    };
    haskell = {
      grammar = g.tree-sitter-haskell;
      types = [ "hs" ];
    };
    html = {
      grammar = g.tree-sitter-html;
      types = [ "html" "htm" ];
    };
    java = {
      grammar = g.tree-sitter-java;
      types = [ "java" ];
    };
    javascript = {
      grammar = g.tree-sitter-javascript;
      types = [ "js" "mjs" "cjs" "jsx" ];
      highlights = [
        (js "highlights.scm")
        (js "highlights-jsx.scm")
        (js "highlights-params.scm")
      ];
    };
    json = {
      grammar = g.tree-sitter-json;
      types = [ "json" "jsonc" "lock" ];
    };
    lua = {
      grammar = g.tree-sitter-lua;
      types = [ "lua" ];
    };
    make = {
      grammar = g.tree-sitter-make;
      types = [ "Makefile" "makefile" "GNUmakefile" "mk" "mak" ];
    };
    nix = {
      grammar = g.tree-sitter-nix;
      types = [ "nix" ];
    };
    python = {
      grammar = g.tree-sitter-python;
      types = [ "py" "pyi" ];
    };
    ruby = {
      grammar = g.tree-sitter-ruby;
      types = [ "rb" ];
    };
    rust = {
      grammar = g.tree-sitter-rust;
      types = [ "rs" ];
    };
    sql = {
      grammar = g.tree-sitter-sql;
      types = [ "sql" ];
    };
    toml = {
      grammar = g.tree-sitter-toml;
      types = [ "toml" ];
    };
    tsx = {
      grammar = g.tree-sitter-tsx;
      types = [ "tsx" ];
      highlights = [
        "${typescriptSrc}/queries/highlights.scm"
        (js "highlights-jsx.scm")
        (js "highlights.scm")
      ];
    };
    typescript = {
      grammar = g.tree-sitter-typescript;
      types = [ "ts" "mts" "cts" ];
      highlights = [
        "${typescriptSrc}/queries/highlights.scm"
        (js "highlights.scm")
      ];
    };
    yaml = {
      grammar = g.tree-sitter-yaml;
      types = [ "yml" "yaml" ];
    };
  };

  manifest = builtins.toJSON (
    lib.mapAttrs (name: l: {
      symbol = "tree_sitter_${name}";
      file_types = l.types;
    }) languages
  );

  install = name: l: ''
    mkdir -p $out/${name}
    ln -s ${l.grammar}/parser $out/${name}/parser
    cat ${lib.escapeShellArgs (l.highlights or [ (queries l.grammar "highlights.scm") ])} > $out/${name}/highlights.scm
  '';
in
runCommand "aid-tree-sitter-grammars" { passAsFile = [ "manifest" ]; inherit manifest; } ''
  ${lib.concatStrings (lib.mapAttrsToList install languages)}
  cp $manifestPath $out/languages.json
''
