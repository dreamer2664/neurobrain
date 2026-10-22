#!/usr/bin/env python3
"""Re-encode token streams with an existing tokenizer (no retraining).

Used when corpus binaries are lost in a sandbox rollback but the tokenizer and a
trained checkpoint must stay compatible: the vocabulary is the checkpoint's, so
the streams are rebuilt around it instead of the other way round.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.data import write_tokens  # noqa: E402
from neurobrain.tokenizer import BPETokenizer  # noqa: E402
from neurobrain.ui_world import protocol_corpus  # noqa: E402
from scripts.build_corpus import SEED_UI_CORPUS, iter_files, read_py  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--max-code-bytes", type=int, default=160_000_000)
    ap.add_argument("--ui-samples", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import sysconfig
    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]

    roots = [sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"]]
    files = list(iter_files(roots))
    rng = random.Random(args.seed)
    rng.shuffle(files)

    docs, total = [], 0
    for p in files:
        if total >= args.max_code_bytes:
            break
        src = read_py(p)
        if not src:
            continue
        docs.append(src)
        total += len(src.encode("utf-8"))
    rng.shuffle(docs)
    n_val = max(8, int(0.03 * len(docs)))
    print(f"[retok] {len(docs)} files, {total/1e6:.1f} MB")

    # NOTE(agent-04): stream ids to disk in chunks. Accumulating ~55M python
    # ints in one list costs ~36 B each and OOMs a 2 GB sandbox; writing every
    # 2M tokens keeps peak RSS flat. Output bytes are identical.
    import numpy as np

    def encode_stream(tag: str, texts) -> int:
        path = os.path.join(args.data, f"{tag}.bin")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        total, buf = 0, []
        with open(path, "wb") as f:
            for i, s in enumerate(texts):
                buf.extend(tok.encode(s))
                buf.append(eot)
                if len(buf) >= 2_000_000:
                    np.array(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf = []
                    print(f"[retok] {tag}: {i+1} docs, {total/1e6:.2f}M tokens", flush=True)
            if buf:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
        print(f"[retok] {tag}.bin {total/1e6:.2f}M tokens", flush=True)
        return total

    counts = {}
    counts["code_train"] = encode_stream("code_train", docs[n_val:])
    counts["code_val"] = encode_stream("code_val", docs[:n_val])

    ui_train = list(protocol_corpus(int(args.ui_samples * 0.97), seed=args.seed))
    ui_val = list(protocol_corpus(int(args.ui_samples * 0.03) + 8, seed=args.seed + 1))
    ui_train = [SEED_UI_CORPUS] * max(1, len(ui_train) // 200) + ui_train
    counts["ui_train"] = encode_stream("ui_train", ui_train)
    counts["ui_val"] = encode_stream("ui_val", ui_val)
    print(f"[retok] done: { {k: round(v/1e6, 2) for k, v in counts.items()} } M tokens")


if __name__ == "__main__":
    main()
