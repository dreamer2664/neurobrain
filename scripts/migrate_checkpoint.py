"""Migrate a legacy bare .pt (trainer format) to a checkpoint directory.

    python3 scripts/migrate_checkpoint.py checkpoints/lm_best.pt \
        --tokenizer data/tokenizer.json --out checkpoints/lm

Also `--make-stub checkpoints/lm-stub` writes a seeded, clearly-flagged stub
(real architecture + real tokenizer, untrained weights) for runtime work
before a trained checkpoint is published. (M2 tooling, landed early with M0
because the checkpoint writer is shared.)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainbox import checkpoints as ckmod  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", nargs="?", help="legacy .pt to migrate")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None,
                    help="output dir (default: checkpoints/lm or checkpoints/grip)")
    ap.add_argument("--kind", choices=("auto", "lm", "grip"), default="auto",
                    help="auto: filenames containing 'grip' migrate as grounder")
    ap.add_argument("--make-stub", default=None, metavar="DIR",
                    help="instead of migrating, write a stub checkpoint dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kind = args.kind
    if kind == "auto":
        kind = "grip" if args.pt and "grip" in os.path.basename(args.pt) else "lm"
    tok = args.tokenizer or ckmod.default_tokenizer_path()
    if kind == "lm" and not tok:
        print("no tokenizer.json found", file=sys.stderr)
        return 1
    if args.make_stub:
        info = ckmod.make_stub(args.make_stub, tok, seed=args.seed)
        print(f"stub checkpoint -> {info.path} (params={info.params:,}, step=0)")
        return 0
    if not args.pt:
        ap.print_usage()
        return 2
    out = args.out or f"checkpoints/{kind}"
    if kind == "grip":
        info = ckmod.migrate_grip_legacy(args.pt, out)
    else:
        info = ckmod.migrate_legacy(args.pt, tok, out)
    print(f"migrated {args.pt} ({kind}, step {info.step}) -> {info.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
