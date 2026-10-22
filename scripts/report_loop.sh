#!/usr/bin/env bash
# Periodic evaluation + artifact refresh while training runs.
# Keeps README-visible artifacts (plots, samples, metrics) current and pushes.
set -uo pipefail
cd "$(dirname "$0")/.."

EVERY=${EVERY:-720}

while true; do
  echo "=== [$(date -u +%FT%TZ)] report ==="
  python3 scripts/plot_training.py 2>&1 | tail -2
  python3 -m neurobrain.sample --ckpt checkpoints/lm_best.pt --val-prompt \
      --seed "$RANDOM" --tokens 280 --temperature 0.7 --out samples/lm_sample.txt 2>&1 | tail -30
  python3 -m neurobrain.eval_ui --n 48 --out reports/ui_eval.json 2>&1 | tail -8
  python3 -m neurobrain.eval_code --n 10 --out reports/code_eval.json 2>&1 | tail -10
  python3 -m neurobrain.demo_grip --ckpt checkpoints/grip_latest.pt \
      --out samples/grip_predictions.png 2>&1 | tail -2
  bash scripts/push.sh "report: $(date -u +%FT%TZ)"
  sleep "$EVERY"
done
