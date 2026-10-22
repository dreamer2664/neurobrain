#!/usr/bin/env bash
# Overnight evaluation loop.
#
# The box has 2 GB and one trainer. Evaluations are therefore run *between*
# training segments, never beside them: the loop waits until no trainer process
# exists and there is memory to spare, then measures the two bars and pushes.
set -uo pipefail
cd "$(dirname "$0")/.."

EVERY=${EVERY:-900}
GIT_TOKEN=${GIT_TOKEN:-}
export GIT_TOKEN
export REPO_SLUG=${REPO_SLUG:-dreamer2664/neurobrain}

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
trainer_running() { pgrep -f "neurobrain.train_lm|neurobrain.train_grip" >/dev/null; }

stamp() { date -u +%FT%TZ; }

while true; do
  if trainer_running; then
    sleep 45
    continue
  fi
  if [ "$(avail_mb)" -lt 700 ]; then
    echo "[night $(stamp)] waiting for memory ($(avail_mb)MB)"
    sleep 60
    continue
  fi

  echo "=== [night $(stamp)] evaluating ($(avail_mb)MB free) ==="
  LOWMEM_MB=1100 bash scripts/lowmem.sh python3 scripts/qa_eval.py \
      --ckpt checkpoints/lm_best.pt --n 50 --out reports/qa_eval.json 2>&1 | tail -16
  LOWMEM_MB=1100 bash scripts/lowmem.sh python3 scripts/e2e_web_task.py \
      --ckpt checkpoints/lm_best.pt --n 20 --out reports/web_eval.json 2>&1 | tail -10
  LOWMEM_MB=1100 bash scripts/lowmem.sh python3 scripts/e2e_lm_drive.py \
      --ckpt checkpoints/lm_best.pt --n 12 --impossible 6 --out reports/e2e_lm_drive.json 2>&1 | tail -8
  bash scripts/push.sh "night: qa + web eval @ $(stamp)" || true
  echo "=== [night $(stamp)] done, sleeping ${EVERY}s ==="
  sleep "$EVERY"
done
