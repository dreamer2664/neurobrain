"""Instruction synthesis at scale (M3).

Output per dataset directory:

    instructions_train.bin / instructions_val.bin   uint16 token stream, one
                                                    instruction + <|endoftext|> per doc
    shard-XXXX.jsonl.gz                             100k structured records per shard
    manifest.json                                   n, unique, rejects, mix, dedupe

The dedupe index is a set of 64-bit blake2b digests over the normalised text.
A collision does not drop the sample: the surface form is re-rolled up to five
times, which is why the emitted corpus has no exact duplicates while the raw
collision rate stays around 4%.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import sys

from ..profiles import sub_seed
from ..streams import TokenStreamWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.instructions import (  # noqa: E402
    apply_variant, instruction_space, normalize, sample_instruction,
)

SHARD_RECORDS = 100_000


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(normalize(text).encode("utf-8"),
                                          digest_size=8).digest(), "little")


class _RecordShard:
    """Deterministic gzip shard of structured records (mtime pinned to 0)."""

    def __init__(self, dirpath: str, index: int):
        self.index = index
        self.name = f"shard-{index:04d}.jsonl.gz"
        self.path = os.path.join(dirpath, self.name)
        self._raw = open(self.path, "wb")
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=6, mtime=0)
        self.n = 0

    def write(self, rec: dict) -> None:
        self._gz.write((json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode())
        self.n += 1

    def close(self) -> dict:
        self._gz.close()
        self._raw.close()
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"name": self.name, "sha256": h.hexdigest(),
                "bytes": os.path.getsize(self.path), "n": self.n}


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    n = int(cfg["n"])
    if n <= 0:
        raise SystemExit("instructions: n must be > 0")
    n_val = max(200, int(cfg.get("val_frac", 0.01) * n))
    eot = tok.encode("<|endoftext|>")[0]

    rng = random.Random(sub_seed(seed, "instructions"))
    log(f"[instructions] target {n:,} ({n_val:,} val), product space ~{instruction_space():,}")

    index: set[int] = set()
    stats = {"raw": 0, "rejects": 0, "chars": 0, "family": {}, "variant": {},
             "len_hist": [0] * 12}
    shards: list[dict] = []   # token streams
    records: list[dict] = []  # jsonl.gz shards

    class _Split:
        def __init__(self, split: str):
            self.split = split
            self.w = TokenStreamWriter(os.path.join(dirpath, f"instructions_{split}.bin"))
            self.shard = _RecordShard(dirpath, 0) if split == "train" else None

        def add(self, text: str, ins) -> bool:
            stats["raw"] += 1
            d = _digest(text)
            if d in index:
                stats["rejects"] += 1
                return False
            index.add(d)
            self.w.write(tok.encode(text) + [eot])
            if self.shard is not None:
                self.shard.write(ins.to_dict())
                if self.shard.n >= SHARD_RECORDS:
                    records.append(self.shard.close())
                    log(f"[instructions] {records[-1]['name']}: {records[-1]['n']:,} records")
                    self.shard = _RecordShard(dirpath, self.shard.index + 1)
            stats["chars"] += len(text)
            stats["len_hist"][min(len(stats["len_hist"]) - 1, len(text) // 10)] += 1
            stats["family"][ins.family] = stats["family"].get(ins.family, 0) + 1
            parts = set(ins.variant.split("+"))
            for key in ("typo", "contraction", "title", "lower", "prefix", "suffix"):
                if key in parts:
                    stats["variant"][key] = stats["variant"].get(key, 0) + 1
            if parts == {"plain"}:
                stats["variant"]["plain"] = stats["variant"].get("plain", 0) + 1
            return True

        def finish(self) -> dict:
            rec = self.w.close()
            if self.shard is not None and self.shard.n:
                records.append(self.shard.close())
                log(f"[instructions] {records[-1]['name']}: {records[-1]['n']:,} records")
            log(f"[instructions] {rec['name']}: {rec['docs']:,} docs, "
                f"{rec['tokens']/1e6:.2f}M tokens")
            return rec

    for split, count, sd in (("train", n, sub_seed(seed, "instructions_train")),
                             ("val", n_val, sub_seed(seed, "instructions_val"))):
        out = _Split(split)
        done = attempts = 0
        while done < count:
            attempts += 1
            if attempts > 40 * count:
                raise RuntimeError("instructions: dedupe index cannot fill the profile; "
                                   "widen the template space")
            ins = sample_instruction(rng)
            ok = out.add(ins.text, ins)
            if not ok:
                for _ in range(5):           # re-roll the surface form only
                    ins2 = apply_variant(sample_instruction(rng), rng)
                    if out.add(ins2.text, ins2):
                        ok = True
                        break
                if not ok:
                    continue
            done += 1
            if done % 250_000 == 0:
                log(f"[instructions] {split}: {done:,}/{count:,}, "
                    f"{stats['rejects']:,} rejects, {out.w.tokens/1e6:.1f}M tokens")
        shards.append(out.finish())

    raw = max(1, stats["raw"])
    uniq = len(index)
    total = n + n_val
    content = {
        "unique": uniq,
        "attempts": stats["raw"],
        "rejects": stats["rejects"],
        "reject_rate": round(stats["rejects"] / raw, 5),
        "chars_mean": round(stats["chars"] / max(1, uniq), 1),
        "len_hist_by_10chars": stats["len_hist"],
        "family": {k: round(v / max(1, total), 4) for k, v in
                   sorted(stats["family"].items(), key=lambda kv: -kv[1])},
        "variant": {k: round(v / max(1, total), 4) for k, v in
                    sorted(stats["variant"].items(), key=lambda kv: -kv[1])},
        "dedupe": f"blake2b64(normalise(text)) set, index {uniq:,} entries, re-roll on hit",
        "product_space_estimate": instruction_space(),
        "record_shards": len(records),
    }
    log(f"[instructions] {uniq:,} unique, reject rate {100*content['reject_rate']:.2f}%, "
        f"mean {content['chars_mean']} chars, {len(content['family'])} families")
    return {
        "kind": "token_streams",
        "dataset": "instructions",
        "n": uniq,
        "units": "instructions",
        "shards": shards,
        "records": records,
        "content": content,
    }
