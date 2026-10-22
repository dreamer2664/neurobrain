#!/usr/bin/env bash
# Continuous training driver: train -> sample -> push, forever.
set -uo pipefail
cd "$(dirname "$0")/.."

SEGMENT=${SEGMENT:-1500}     # seconds of training per push cycle
TOTAL=${TOTAL:-40000}        # total optimisation steps (resumable)
i=0

while true; do
  i=$((i+1))
  echo "=== segment $i  $(date -u +%FT%TZ) ==="
  python3 -m neurobrain.train_lm --resume --steps "$TOTAL" --time_budget "$SEGMENT" \
      --batch 8 --accum 3 --ctx 256 2>&1 | tee -a reports/train_stdout.log
  python3 -m neurobrain.sample --ckpt checkpoints/lm_best.pt --tokens 400 \
      --out samples/lm_sample.txt 2>&1 | tail -30
  bash scripts/push.sh "train: segment $i @ $(date -u +%FT%TZ) [$(python3 - <<'PY'
import json
try:
    print(json.load(open("reports/lm_summary.json"))["step"])
except Exception:
    print("?")
PY
) steps]"
  sleep 5
done
