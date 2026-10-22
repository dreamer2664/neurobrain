#!/usr/bin/env python3
"""Build the research-and-report training stream (web_train.bin / web_val.bin).

Same tokenizer and same format as the drive stream, so the trainer only needs
one more source. Each episode is a full click-by-click transcript of

    desktop -> chrome -> results -> article -> copy -> docs -> paste -> save

and the scan before each click lists every element on the screen, including the
distractors. The model has to pick the right label out of the list, which is the
skill the 95% bar is actually about.
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

from neurobrain.tokenizer import BPETokenizer  # noqa: E402
from neurobrain.web_world import web_episode  # noqa: E402


def write(name: str, n: int, seed: int, tok, data_dir: str) -> dict:
    rng = random.Random(seed)
    eot = tok.encode("<|endoftext|>")[0]
    path = os.path.join(data_dir, f"{name}.bin")
    tmp = path + ".tmp"
    buf = array("H")
    total = 0
    with open(tmp, "wb") as f:
        for i in range(n):
            ids = tok.encode(web_episode(rng)) + [eot]
            buf.extend(ids)
            total += len(ids)
            if len(buf) > 5_000_000:
                f.write(buf.tobytes())
                buf = array("H")
            if (i + 1) % 2000 == 0:
                print(f"[web] {name}: {i+1:,} episodes, {total/1e6:.2f}M tokens", flush=True)
        if buf:
            f.write(buf.tobytes())
    os.replace(tmp, path)
    return {"name": name, "episodes": n, "tokens": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=57)
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    n_val = max(50, args.episodes // 20)
    recs = [write("web_train", args.episodes - n_val, args.seed, tok, args.data),
            write("web_val", n_val, args.seed + 1, tok, args.data)]
    for r in recs:
        print(f"[web] {r['name']}: {r['episodes']} episodes, {r['tokens']/1e6:.2f}M tokens")
    with open(os.path.join(ROOT, "reports", "web_corpus.json"), "w") as f:
        json.dump({"streams": recs}, f, indent=2)


if __name__ == "__main__":
    main()
