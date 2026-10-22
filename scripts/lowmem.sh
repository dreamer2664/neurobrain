#!/usr/bin/env bash
# Run a heavy job with a virtual-memory ceiling so that, if it is too big, it
# dies alone instead of taking the trainer down with it.
#   scripts/lowmem.sh 900 python3 -m neurobrain.eval_drive
set -uo pipefail
CAP=${LOWMEM_MB:-900}
ulimit -v $((CAP * 1024))
exec "$@"
