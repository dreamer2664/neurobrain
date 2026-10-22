"""Per-stream validation loss table for a checkpoint.

Corpus v2 splits the code val stream into plain vs instructed
(code_val_v2_plain.bin / code_val_v2_instr.bin) so deliverable 6 can be
evaluated separately; this tool reports every *_val*.bin stream in data/
in one table, for any checkpoint.

    python3 -m neurobrain.eval_streams --ckpt checkpoints/lm_best.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch

from .data import TokenSource
from .sample import load


def main() -> None:
    ap = argparse.ArgumentParser(description="val loss per stream")
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "lm_best.pt"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--pattern", default="*_val*.bin")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    model, tok, step = load(args.ckpt, os.path.join(args.data, "tokenizer.json"))
    rng = random.Random(0)
    rows: dict[str, float] = {}
    for p in sorted(glob.glob(os.path.join(args.data, args.pattern))):
        name = os.path.basename(p)[: -len(".bin")]
        try:
            src = TokenSource(p, model.cfg.ctx)
        except ValueError:
            continue
        with torch.no_grad():
            losses = [model(*src.sample(args.batch, rng))[1].item() for _ in range(args.batches)]
        rows[name] = round(sum(losses) / len(losses), 4)
        print(f"[streams] {name:24s} {rows[name]:.4f}", flush=True)
    out = {"ckpt": args.ckpt, "step": step, "val_loss_by_stream": rows}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print("[streams] wrote", args.out)
    else:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
