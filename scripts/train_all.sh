#!/usr/bin/env bash
# Unified training driver: alternates the language brain and GripNet, then
# renders artifacts and pushes to GitHub. Sequential on purpose: the sandbox
# has ~2GB RAM, so the two models take turns.
set -uo pipefail
cd "$(dirname "$0")/.."

# survival: rollbacks wipe installed packages and the large token streams
python3 -c "import torch, numpy, PIL" 2>/dev/null || {
  echo "[setup] reinstalling stack"
  python3 -m pip install --quiet --no-input torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install --quiet --no-input numpy pillow matplotlib
}
if [ ! -s data/code_train.bin ] || [ ! -s data/drive_train.bin ] || [ ! -s data/tokenizer.json ]; then
  echo "[setup] corpus missing, restoring"
  bash scripts/restore_data.sh
fi

LM_SEG=${LM_SEG:-1200}       # seconds of language training per cycle
GRIP_SEG=${GRIP_SEG:-300}    # seconds of vision training per cycle
LM_TOTAL=${LM_TOTAL:-16000}  # total LM steps (resumable)
GRIP_TOTAL=${GRIP_TOTAL:-24000}
FIRST=${FIRST:-lm}
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# survival: rollbacks wipe installed packages and large token streams
python3 -c "import torch, numpy, PIL" 2>/dev/null || {
  echo "[setup] reinstalling stack"
  python3 -m pip install --quiet --no-input torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install --quiet --no-input numpy pillow matplotlib
}
if [ ! -s data/code_train.bin ] || [ ! -s data/drive_train.bin ] || [ ! -s data/tokenizer.json ]; then
  echo "[setup] corpus missing, restoring"
  bash scripts/restore_data.sh
fi

export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# survival: large token streams can vanish in a rollback; rebuild if so
if [ ! -f data/drive_train.bin ]; then
  echo "[setup] drive corpus missing, building"
  python3 scripts/build_drive_corpus.py --episodes 400000
fi
if [ ! -f data/code_train.bin ] || [ ! -f data/tokenizer.json ]; then
  echo "[setup] corpus missing, re-encoding with the checkpoint tokenizer"
  python3 scripts/retokenize.py
fi

run_lm() {
  echo "=== [$(date -u +%FT%TZ)] language brain: ${LM_SEG}s ==="
  python3 -m neurobrain.train_lm --resume --steps "$LM_TOTAL" --time_budget "$LM_SEG" \
      --batch 6 --accum 4 --ctx 256 --threads 2 2>&1 | tee -a reports/train_stdout.log
  python3 -m neurobrain.sample --ckpt checkpoints/lm_best.pt --tokens 400 \
      --out samples/lm_sample.txt 2>&1 | tail -40
}

run_grip() {
  echo "=== [$(date -u +%FT%TZ)] gripnet: ${GRIP_SEG}s ==="
  python3 -m neurobrain.train_grip --resume --steps "$GRIP_TOTAL" --time_budget "$GRIP_SEG" \
      --batch 16 --threads 1 2>&1 | tee -a reports/grip_stdout.log
  python3 -m neurobrain.demo_grip --ckpt checkpoints/grip_latest.pt \
      --out samples/grip_predictions.png 2>&1 | tail -5
}

if [ "$FIRST" = "grip" ]; then
  ( GRIP_SEG=${FIRST_GRIP_SEG:-$GRIP_SEG}; run_grip )
fi

i=0
while true; do
  i=$((i+1))
  run_lm
  run_grip
  STEPS=$(python3 - <<'PY'
import json, os
out = {}
for name in ("lm_summary", "grip_summary"):
    try:
        d = json.load(open(f"reports/{name}.json"))
        out[name.split("_")[0]] = d.get("step")
    except Exception:
        pass
print(" ".join(f"{k}={v}" for k, v in out.items()))
PY
)
  bash scripts/push.sh "train: cycle $i @ $(date -u +%FT%TZ) [$STEPS]"
  sleep 5
done
