#!/usr/bin/env bash
# Periodic evaluation + artifact refresh while training runs.
# Keeps README-visible artifacts (plots, samples, metrics) current and pushes.
#
# Every job here is wrapped in lowmem.sh and gated on free memory: the box has
# 2 GB and one trainer in it, and an ungated eval is what OOM-killed segments.
set -uo pipefail
cd "$(dirname "$0")/.."

EVERY=${EVERY:-720}
MIN_AVAIL=${MIN_AVAIL:-800}   # MB; below this, skip the pass instead of killing training
REPORTS=0                     # set to 1 to push artifacts from here

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
low() { LOWMEM_MB=${LOWMEM_MB:-700} bash scripts/lowmem.sh "$@"; }

while true; do
  if [ "$(avail_mb)" -lt "$MIN_AVAIL" ]; then
    echo "=== [$(date -u +%FT%TZ)] report skipped: $(avail_mb)MB available, trainer needs the room"
    sleep "$EVERY"
    continue
  fi
  echo "=== [$(date -u +%FT%TZ)] report ($(avail_mb)MB available) ==="
  low python3 scripts/plot_training.py 2>&1 | tail -2
  low python3 -m neurobrain.sample --ckpt checkpoints/lm_best.pt --val-prompt \
      --seed "$RANDOM" --tokens 280 --temperature 0.7 --out samples/lm_sample.txt 2>&1 | tail -30
  low python3 -m neurobrain.eval_ui --n 48 --out reports/ui_eval.json 2>&1 | tail -8
  low python3 -m neurobrain.eval_drive --n 24 --impossible 24 --out reports/drive_eval.json 2>&1 | tail -14
  low python3 -m neurobrain.eval_code --n 10 --out reports/code_eval.json 2>&1 | tail -10
  low python3 scripts/e2e_lm_drive.py --ckpt checkpoints/lm_best.pt --n 12 --impossible 6 \
      --out reports/e2e_lm_drive.json 2>&1 | tail -12
  low python3 -m neurobrain.demo_grip --ckpt checkpoints/grip_latest.pt \
      --out samples/grip_predictions.png 2>&1 | tail -2
  [ "$REPORTS" = "1" ] && bash scripts/push.sh "report: $(date -u +%FT%TZ)"
  sleep "$EVERY"
done
