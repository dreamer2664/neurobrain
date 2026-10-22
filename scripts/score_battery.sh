#!/usr/bin/env bash
# Full evaluation battery for one checkpoint (deliverable 7 round table).
# Usage: bash scripts/score_battery.sh CKPT NAME [--limit N]
# Runs sequentially (this box: 2 cores/2GB): codebench (greedy+sampled),
# eval_code, eval_drive, eval_streams, self_repair, then packs the card dir.
# Each step's failure is tolerated (rc recorded) so the rest still runs.
set -u
CKPT=${1:?usage: score_battery.sh CKPT NAME [--limit N]}
NAME=${2:?missing NAME}
shift 2
LIMIT=""
if [ "${1:-}" = "--limit" ]; then LIMIT="--limit ${2:?}"; fi
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1
R=reports
declare -A RC

run() { # run <key> <cmd...>
  local key=$1; shift
  echo "=== [$NAME] $key: $*"
  local t0=$SECONDS
  "$@" > "$R/battery_${NAME}_${key}.log" 2>&1
  RC[$key]=$?
  echo "--- [$NAME] $key rc=${RC[$key]} in $((SECONDS-t0))s"
  tail -4 "$R/battery_${NAME}_${key}.log"
}

run codebench  python3 -m neurobrain.codebench score --ckpt "$CKPT" --out "$R/codebench_${NAME}.json" $LIMIT
run eval_code  python3 -m neurobrain.eval_code  --ckpt "$CKPT" --out "$R/code_eval_${NAME}.json"
run eval_drive python3 -m neurobrain.eval_drive --ckpt "$CKPT" --n 48 --tokens 220 --out "$R/drive_eval_${NAME}.json"
run streams    python3 -m neurobrain.eval_streams --ckpt "$CKPT" --out "$R/streams_${NAME}.json"
run repair     python3 -m neurobrain.self_repair --ckpt "$CKPT" --out "$R/self_repair_${NAME}.json" --seed 0

# pack the card only from reports that actually landed
PACK=(python3 scripts/pack_checkpoint.py --ckpt "$CKPT" --name "$NAME" --samples 2)
[ -f "$R/codebench_${NAME}.json" ]    && PACK+=(--codebench   "$R/codebench_${NAME}.json")
[ -f "$R/drive_eval_${NAME}.json" ]   && PACK+=(--drive-eval  "$R/drive_eval_${NAME}.json")
[ -f "$R/code_eval_${NAME}.json" ]    && PACK+=(--code-eval   "$R/code_eval_${NAME}.json")
run pack "${PACK[@]}"

echo "=== [$NAME] summary"
python3 - "$NAME" "$R" <<'PY'
import json, os, sys
name, R = sys.argv[1], sys.argv[2]
def get(f):
    p = os.path.join(R, f)
    try:
        return json.load(open(p))
    except Exception:
        return None
cb = get(f"codebench_{name}.json")
if cb:
    for m, d in cb["modes"].items():
        print(f"codebench {m:8s} pass@1 {d['pass_at_1']:.3f} syntax {d['syntax_valid_rate']:.3f} body {d['body_rate']:.3f} ({d['gen_tok_s']} tok/s)")
ce = get(f"code_eval_{name}.json")
if ce:
    print(f"eval_code val_loss {ce.get('val_loss', -1):.4f} ppl {ce.get('val_ppl', -1):.3f} "
          f"syntax_rate {ce.get('syntax_rate', -1):.3f} coherent {ce.get('coherent_rate', -1):.3f}")
de = get(f"drive_eval_{name}.json")
if de:
    keys = {k: v for k, v in de.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    print("eval_drive", {k: round(v, 4) for k, v in keys.items()})
st = get(f"streams_{name}.json")
if st:
    print("streams", {k: round(v, 4) for k, v in st["val_loss_by_stream"].items()})
sr = get(f"self_repair_{name}.json")
if sr:
    print(f"self_repair rate {sr['repair_rate']} final_pass {sr['final_pass']} ({sr['seconds']}s)")
PY
echo "=== [$NAME] battery done; rcs: $(for k in "${!RC[@]}"; do echo -n "$k=${RC[$k]} "; done)"
