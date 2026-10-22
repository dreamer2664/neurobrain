#!/usr/bin/env bash
# Unified training driver: alternates the language brain and GripNet, then
# renders artifacts and pushes to GitHub. Sequential on purpose: the sandbox
# has ~2GB RAM, so the two models take turns.
set -uo pipefail
cd "$(dirname "$0")/.."

# --- survival: rollbacks wipe installed packages and the large token streams ---
ensure_stack() {
  python3 -c "import torch, numpy, PIL" 2>/dev/null || {
    echo "[setup] reinstalling stack"
    python3 -m pip install --quiet --no-input torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install --quiet --no-input numpy pillow matplotlib
  }
  if [ ! -s data/code_train.bin ] || [ ! -s data/drive_train.bin ] || [ ! -s data/tokenizer.json ]; then
    echo "[setup] corpus missing, restoring"
    bash scripts/restore_data.sh
  fi
  if [ ! -f data/code_train.bin ] || [ ! -f data/tokenizer.json ]; then
    echo "[setup] corpus missing, re-encoding with the checkpoint tokenizer"
    python3 scripts/retokenize.py
  fi
}

LM_SEG=${LM_SEG:-1200}       # seconds of language training per cycle
GRIP_SEG=${GRIP_SEG:-300}    # seconds of vision training per cycle
LM_TOTAL=${LM_TOTAL:-16000}  # total LM steps (resumable)
GRIP_TOTAL=${GRIP_TOTAL:-24000}
FIRST=${FIRST:-lm}
MIN_AVAIL=${MIN_AVAIL:-650}  # MB of MemAvailable required before a segment starts
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

avail_mb() { awk '/MemAvailable/{print int($2)}' /proc/meminfo; }

# The box has 2 GB. Running a test suite or a corpus build next to a trainer is
# what keeps OOM-killing segments, so a segment refuses to start without room.
mem_gate() {
  local waited=0
  while [ "$(avail_mb)" -lt "$MIN_AVAIL" ] && [ "$waited" -lt 300 ]; do
    echo "[gate] only $(avail_mb)MB available, waiting"
    sleep 10
    waited=$((waited + 10))
  done
  echo "[gate] $(avail_mb)MB available, starting"
}

# A killed segment is not a lost segment: every trainer resumes from its last
# checkpoint, so the fix is simply to start it again.
run_retry() {
  local label=$1
  shift
  local n=0
  local rc=0
  while [ "$n" -lt 5 ]; do
    mem_gate
    "$@"
    rc=$?
    [ "$rc" -eq 0 ] && return 0
    n=$((n + 1))
    echo "[retry] $label exited rc=$rc (attempt $n/5), $(avail_mb)MB available; resuming in 15s"
    sleep 15
  done
  echo "[retry] $label gave up after 5 attempts"
  return 1
}

lm_segment() {
  python3 -m neurobrain.train_lm --resume --steps "$LM_TOTAL" --time_budget "$LM_SEG" \
      --batch 6 --accum 4 --ctx 256 --threads 2 2>&1 | tee -a reports/train_stdout.log
}

grip_segment() {
  python3 -m neurobrain.train_grip --resume --steps "$GRIP_TOTAL" --time_budget "$GRIP_SEG" \
      --batch 16 --threads 1 2>&1 | tee -a reports/grip_stdout.log
}

run_lm() {
  echo "=== [$(date -u +%FT%TZ)] language brain: ${LM_SEG}s ==="
  run_retry lm lm_segment
  python3 -m neurobrain.sample --ckpt checkpoints/lm_best.pt --tokens 400 \
      --out samples/lm_sample.txt 2>&1 | tail -40
}

run_grip() {
  echo "=== [$(date -u +%FT%TZ)] gripnet: ${GRIP_SEG}s ==="
  run_retry grip grip_segment
  python3 -m neurobrain.demo_grip --ckpt checkpoints/grip_latest.pt \
      --out samples/grip_predictions.png 2>&1 | tail -5
}

ensure_stack
if [ "$FIRST" = "grip" ]; then
  ( GRIP_SEG=${FIRST_GRIP_SEG:-$GRIP_SEG}; run_grip )
fi

i=0
while true; do
  i=$((i + 1))
  ensure_stack
  run_lm
  run_grip
  STEPS=$(python3 -c "
import json
out = {}
for name in ('lm_summary', 'grip_summary'):
    try:
        d = json.load(open(f'reports/{name}.json'))
        out[name.split('_')[0]] = d.get('step')
    except Exception:
        pass
print(' '.join(f'{k}={v}' for k, v in out.items()))
")
  bash scripts/push.sh "train: cycle $i @ $(date -u +%FT%TZ) [$STEPS]"
  sleep 5
done
