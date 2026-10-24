/**
  The English tldr pages the supervisor's `tldr_page` tool reads, for `AID_TLDR_PAGES`: `$out/<platform>/<command>.md`.
  Only `pages/` of the repository: its translations and tooling stay out of the closure.
*/
{
  runCommand,
  src,
}:
runCommand "tldr-pages" { } ''
  cp -r ${src}/pages $out
''
