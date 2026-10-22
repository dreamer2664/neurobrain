#!/usr/bin/env bash
# M1 experiment: ONE variable = code corpus v1 -> v2 (AST-filtered, deduped,
# category-balanced, +20% instruction-mix stream). drive/ui bins unchanged
# (v1). --code-val stays on the v1 val bin so best-checkpoint selection and
# the headline val-loss number remain comparable with the published lineage
# (2.09-2.12 @ ~4.6k steps) and with run12k. v2-stream val losses come from
# eval_streams afterwards (code_val_v2_plain / code_val_v2_instr).
#
# Launch ONLY when the box is otherwise idle (2 cores/2GB rule):
#   nohup-free: run via start_process, logs captured by the fleet sandbox.
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m neurobrain.train_lm \
  --steps 12000 --batch 8 --accum 3 --ctx 256 \
  --lr 2e-3 --warmup 150 --time_budget 10800 \
  --code-train data/code_train_v2.bin \
  --code-val data/code_val.bin \
  --out checkpoints/runv2 --threads 2
