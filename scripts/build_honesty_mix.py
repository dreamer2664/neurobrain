#!/usr/bin/env python3
"""Build the honesty mix: a short second-stage stream biased toward the cases
the main corpus only shows rarely.

The main drive corpus is 6.5% impossible, 3% blocked and 14% failed. That is
enough to define the behaviour but not necessarily enough to move a model that
has already learned "every objective ends in success". This stream is the
targeted correction: mostly normal UI episodes so the task skill does not
decay, plus heavily oversampled impossible and failed terminals.

    python3 scripts/build_honesty_mix.py --tokens 15000000 --impossible 0.30

Writes data/hon_train.bin / data/hon_val.bin in the same format as the main
streams, so the trainer needs no changes and the invariants still apply.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from array import array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.drive import (  # noqa: E402
    _exhausted_episode, drive_stats, impossible_episode, ui_episode,
)
from neurobrain.tokenizer import BPETokenizer  # noqa: E402
from neurobrain.ui_world import make_layout  # noqa: E402


def draw(rng: random.Random, p_impossible: float, p_failed: float) -> str:
    r = rng.random()
    if r < p_impossible:
        return impossible_episode(rng)
    if r < p_impossible + p_failed:
        return _exhausted_episode(rng, make_layout(rng))
    # a normal, solvable episode (ui_episode also draws exhausted ones)
    while True:
        ep = ui_episode(rng)
        if "<|done|>success" in ep:
            return ep


def build(name: str, tokens: int, p_impossible: float, p_failed: float,
          seed: int, tok: BPETokenizer, data_dir: str) -> dict:
    rng = random.Random(seed)
    eot = tok.encode("<|endoftext|>")[0]
    path = os.path.join(data_dir, f"{name}.bin")
    tmp = path + ".tmp"
    buf = array("H")
    total = 0
    episodes = {"success": 0, "impossible": 0, "failed": 0}
    with open(tmp, "wb") as f:
        while total < tokens:
            ep = draw(rng, p_impossible, p_failed)
            for kind, marker in (("success", "<|done|>success"),
                                 ("impossible", "<|done|>impossible"),
                                 ("failed", "<|done|>failed")):
                if marker in ep:
                    episodes[kind] += 1
                    break
            ids = tok.encode(ep)
            ids.append(eot)
            buf.extend(ids)
            total += len(ids)
            if len(buf) > 5_000_000:
                f.write(buf.tobytes())
                buf = array("H")
        if buf:
            f.write(buf.tobytes())
    os.replace(tmp, path)
    n = sum(episodes.values()) or 1
    return {"name": name, "tokens": total, "episodes": n,
            "mix": {k: round(v / n, 3) for k, v in episodes.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--tokens", type=int, default=15_000_000)
    ap.add_argument("--impossible", type=float, default=0.30)
    ap.add_argument("--failed", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    out = []
    for name, n, sd in (("hon_train", args.tokens, args.seed),
                        ("hon_val", max(200_000, args.tokens // 30), args.seed + 1)):
        rec = build(name, n, args.impossible, args.failed, sd, tok, args.data)
        print(f"[honesty] {rec['name']}: {rec['tokens']/1e6:.2f}M tokens, "
              f"{rec['episodes']:,} episodes, mix {rec['mix']}")
        out.append(rec)
    with open(os.path.join(ROOT, "reports", "honesty_mix.json"), "w") as f:
        json.dump({"main_corpus": drive_stats(2000, args.seed), "streams": out}, f, indent=2)


if __name__ == "__main__":
    main()
