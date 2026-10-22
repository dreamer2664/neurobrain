#!/usr/bin/env bash
# Continuous GripNet training driver: train -> render predictions -> push, forever.
set -uo pipefail
cd "$(dirname "$0")/.."

SEGMENT=${SEGMENT:-900}
TOTAL=${TOTAL:-12000}
i=0

while true; do
  i=$((i+1))
  echo "=== grip segment $i  $(date -u +%FT%TZ) ==="
  python3 -m neurobrain.train_grip --resume --steps "$TOTAL" --time_budget "$SEGMENT" \
      --batch 16 --threads 1 2>&1 | tee -a reports/grip_stdout.log
  python3 -m neurobrain.demo_grip --ckpt checkpoints/grip_latest.pt --out samples/grip_predictions.png 2>&1 | tail -5
  bash scripts/push.sh "grip: segment $i @ $(date -u +%FT%TZ)"
  sleep 5
done
