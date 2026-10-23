#!/usr/bin/env python3
"""Export the EMA shadow from a training checkpoint as a standard ckpt file.

The result has "model" = ema weights, so every existing tool (sample.load,
codebench, eval_*, pack_checkpoint, score_battery) consumes it unchanged.

    python3 scripts/export_ema.py --ckpt checkpoints/runE/lm_latest.pt \
        --out checkpoints/runE/lm_ema.pt
"""
import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck.get("ema"), dict):
        raise SystemExit(f"no EMA shadow in {args.ckpt} (trained without --ema-decay?)")
    out = {"model": {k: v.clone() for k, v in ck["ema"].items()},
           "step": ck.get("step", -1), "best_val": ck.get("best_val"),
           "cfg": ck.get("cfg")}
    torch.save(out, args.out)
    print(f"[ema] exported EMA weights from step {out['step']} -> {args.out}")


if __name__ == "__main__":
    main()
