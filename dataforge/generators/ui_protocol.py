"""Desktop-control transcripts in the text protocol (the seed world).

Same generator as ``neurobrain.ui_world.protocol_corpus``; this module adds
streaming, a manifest, a train/val split and per-document dedupe with a stable
digest (never Python's salted ``hash``, which would break determinism).
"""
from __future__ import annotations

import hashlib
import os
import sys

from ..profiles import sub_seed
from ..streams import TokenStreamWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.ui_world import protocol_corpus  # noqa: E402


def stable_digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    n = int(cfg["n"])
    n_val = max(8, int(cfg["val_frac"] * n))
    eot = tok.encode("<|endoftext|>")[0]

    shards = []
    for split, count, sd in (("train", n - n_val, sub_seed(seed, "ui_protocol_train")),
                             ("val", n_val, sub_seed(seed, "ui_protocol_val"))):
        w = TokenStreamWriter(os.path.join(dirpath, f"ui_{split}.bin"))
        seen: set[int] = set()
        dup = 0
        eot_txt = "<|endoftext|>"
        for text in protocol_corpus(count, seed=sd):
            # protocol_corpus already terminates every document; appending a
            # second EOT would emit empty documents (and a 50% "duplicate"
            # rate that is really a format bug).
            if text.endswith(eot_txt):
                text = text[: -len(eot_txt)]
            h = stable_digest(text)
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            w.write(tok.encode(text) + [eot])
        rec = w.close()
        rec["dupes_dropped"] = dup
        log(f"[ui_protocol] {rec['name']}: {rec['docs']} docs, {rec['tokens']/1e6:.2f}M tokens, "
            f"{dup} dupes dropped")
        shards.append(rec)

    return {
        "kind": "token_streams",
        "dataset": "ui_protocol",
        "n": sum(s["docs"] for s in shards),
        "units": "docs",
        "shards": shards,
        "content": {"source": "neurobrain.ui_world.protocol_corpus", "requested": n},
    }
