#!/usr/bin/env bash
# Restore training data after a sandbox rollback.
# Downloads the published corpus + checkpoints from the release first (fast),
# falls back to regenerating locally (slow but guaranteed).
set -uo pipefail
cd "$(dirname "$0")/.."

REL=${REL:-https://github.com/dreamer2664/neurobrain/releases/download/ckpt-1022}
DL=${DL:-curl -sfL --retry 2 --max-time 300}

mkdir -p data checkpoints

need() { [ -s "$1" ] || return 0; return 1; }

if [ ! -s data/tokenizer.json ]; then
  echo "[restore] tokenizer"
  $DL -o data/tokenizer.json "$REL/tokenizer.json" || echo "[restore] tokenizer download failed"
fi

for f in code_train.bin drive_train.bin ui_train.bin; do
  if [ ! -s "data/$f" ]; then
    echo "[restore] downloading $f"
    $DL -o "data/$f" "$REL/$f" || echo "[restore] $f download failed, will regenerate"
  fi
done

if [ ! -s data/code_train.bin ] || [ ! -s data/drive_train.bin ]; then
  echo "[restore] regenerating missing corpora locally"
  [ -s data/code_train.bin ] || python3 scripts/retokenize.py
  [ -s data/drive_train.bin ] || python3 scripts/build_drive_corpus.py --episodes 400000
fi

for f in lm_best.pt grip_best.pt; do
  if [ ! -s "checkpoints/$f" ]; then
    echo "[restore] downloading $f"
    $DL -o "checkpoints/$f" "$REL/$f" || echo "[restore] $f unavailable"
  fi
done

echo "[restore] data:"
ls -la data/*.bin 2>/dev/null | awk '{printf "  %6.1f MB  %s\n", $5/1e6, $9}'
echo "[restore] checkpoints:"
ls -la checkpoints/*.pt 2>/dev/null | awk '{printf "  %6.1f MB  %s\n", $5/1e6, $9}'
