#!/usr/bin/env bash
# Fetch Path D's speaker-segmentation checkpoint.
#
# pyannote/segmentation-3.0 - MIT licence, Copyright (c) 2022 CNRS - as an ONNX
# export maintained by k2-fsa. 5.7 MB, CPU-only, no torch and no Hugging Face
# token required (the upstream HF model card is gated; this mirror is not).
#
# The Docker image runs the equivalent step at build time, so this is only
# needed for a local checkout. Without it Path D reports available=false and the
# system falls back to Path B's weaker dual-pitch cue rather than failing.
set -euo pipefail

URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
TARGET="$DEST/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

if [ -f "$TARGET" ]; then
  echo "already present: $TARGET"
  exit 0
fi

mkdir -p "$DEST"
tmp="$(mktemp -t seg.XXXXXX).tar.bz2"
trap 'rm -f "$tmp"' EXIT

echo "downloading segmentation model..."
curl -fsSL "$URL" -o "$tmp"
tar xjf "$tmp" -C "$DEST"
test -f "$TARGET"
echo "installed: $TARGET"
