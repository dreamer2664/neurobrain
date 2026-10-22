#!/usr/bin/env python3
"""Build the drive corpus: the stream that trains persistence.

Writes data/drive_train.bin and data/drive_val.bin with the existing tokenizer
(no retraining), so checkpoints stay compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.data import write_tokens  # noqa: E402
from neurobrain.drive import drive_corpus, drive_stats  # noqa: E402
from neurobrain.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--episodes", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]

    n_val = max(200, args.episodes // 40)
    print("[drive] stats:", json.dumps(drive_stats(3000, args.seed)))

    def build(n: int, seed: int, name: str) -> None:
        ids: list[int] = []
        for i, ep in enumerate(drive_corpus(n, seed)):
            ids.extend(tok.encode(ep))
            ids.append(eot)
            if (i + 1) % 50_000 == 0:
                print(f"[drive] {name}: {i+1} episodes, {len(ids)/1e6:.2f}M tokens", flush=True)
        write_tokens(os.path.join(args.data, f"{name}.bin"), ids)
        print(f"[drive] {name}.bin {len(ids)/1e6:.2f}M tokens")

    build(args.episodes - n_val, args.seed, "drive_train")
    build(n_val, args.seed + 1, "drive_val")


if __name__ == "__main__":
    main()
