/**
  The default speech model for `aid web --speech-model`: a sherpa-onnx streaming transducer for English, from
  k2-fsa's `asr-models` release. Cased, punctuated text; about 60 MB.
*/
{
  fetchurl,
  runCommand,
}:
let
  name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06";
  tarball = fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${name}.tar.bz2";
    hash = "sha256-yGduX/msKoUpblPuD9TV+x22dw56dkcWbur+NJreaDQ=";
  };
in
runCommand name { } ''
  mkdir $out
  tar -xjf ${tarball} -C $out --strip-components=1
''
