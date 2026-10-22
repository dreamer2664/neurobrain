#!/usr/bin/env python3
"""Regenerate the desktop-protocol token streams with the existing tokenizer.

Use this after changing the synthetic desktop environment: the tokenizer is
loaded (not retrained), so existing language-model checkpoints stay valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.data import write_tokens  # noqa: E402
from neurobrain.tokenizer import BPETokenizer  # noqa: E402
from neurobrain.ui_world import protocol_corpus  # noqa: E402
from scripts.build_corpus import SEED_UI_CORPUS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--samples", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    ui_train = list(protocol_corpus(int(args.samples * 0.97), seed=args.seed))
    ui_val = list(protocol_corpus(int(args.samples * 0.03) + 8, seed=args.seed + 1))
    ui_train = [SEED_UI_CORPUS] * max(1, len(ui_train) // 200) + ui_train

    def encode(texts):
        ids = []
        eot = tok.encode("<|endoftext|>")[0]
        for s in texts:
            ids.extend(tok.encode(s))
            ids.append(eot)
        return ids

    tr, va = encode(ui_train), encode(ui_val)
    write_tokens(os.path.join(args.data, "ui_train.bin"), tr)
    write_tokens(os.path.join(args.data, "ui_val.bin"), va)
    print(f"[ui] ui_train.bin {len(tr)/1e6:.2f}M tokens, ui_val.bin {len(va)/1e6:.2f}M tokens")

    report_path = os.path.join(ROOT, "reports", "corpus_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        report["tokens"]["ui_train"] = len(tr)
        report["tokens"]["ui_val"] = len(va)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
