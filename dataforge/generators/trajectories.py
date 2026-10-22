"""Trajectory forge dataset (M5): task_schema.jsonl.gz shards + manifest.

Every record is validated against ``neurobrain.trajectories.validate_trajectory``
before it is written; a single invalid record fails the build. The manifest
records the mix, the failure/recovery shares and the verdict histogram, and
``dataforge verify`` re-validates a sample straight out of the gzip on disk.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

from ..profiles import sub_seed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.trajectories import (  # noqa: E402
    FAIL_VERDICTS, has_failure, has_recovery, trajectory_corpus, validate_trajectory,
)

SHARD_RECORDS = 5_000


class _Shard:
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
        raise SystemExit("trajectories: n must be > 0")
    rng_seed = sub_seed(seed, "trajectories")
    shards: list[dict] = []
    src_mix: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    failures = recoveries = 0
    n_steps = 0
    n_hist = 0
    ids: set[str] = set()

    shard = _Shard(dirpath, 0)
    for i, rec in enumerate(trajectory_corpus(n, rng_seed)):
        errs = validate_trajectory(rec)
        if errs:
            raise RuntimeError(f"trajectories: invalid record {rec.get('id')}: {errs}")
        if rec["id"] in ids:
            raise RuntimeError(f"trajectories: duplicate id {rec['id']}")
        ids.add(rec["id"])
        shard.write(rec)
        src_mix[rec["source"]] = src_mix.get(rec["source"], 0) + 1
        for h in rec["history"]:
            verdicts[h["verdict"]] = verdicts.get(h["verdict"], 0) + 1
            n_hist += 1
        n_steps += len(rec["plan"])
        if has_failure(rec):
            failures += 1
            if has_recovery(rec):
                recoveries += 1
        if shard.n >= SHARD_RECORDS:
            shards.append(shard.close())
            log(f"[trajectories] {shards[-1]['name']}: {shards[-1]['n']:,} tasks")
            shard = _Shard(dirpath, shard.index + 1)
    if shard.n:
        shards.append(shard.close())
        log(f"[trajectories] {shards[-1]['name']}: {shards[-1]['n']:,} tasks")

    content = {
        "schema": "specs/task_schema.md",
        "source_mix": {k: round(v / max(1, n), 4) for k, v in sorted(src_mix.items())},
        "verdicts": {k: round(v / max(1, n_hist), 4) for k, v in sorted(verdicts.items())},
        "history_per_task": round(n_hist / max(1, n), 2),
        "plan_steps_per_task": round(n_steps / max(1, n), 2),
        "failure_fraction": round(failures / max(1, n), 4),
        "recovery_fraction": round(recoveries / max(1, failures), 4),
        "invalid": 0,
    }
    log(f"[trajectories] {n:,} tasks, {n_hist:,} history entries, "
        f"failure {100*content['failure_fraction']:.1f}%, "
        f"recovery {100*content['recovery_fraction']:.1f}%, "
        f"mix {content['source_mix']}")
    return {
        "kind": "jsonl",
        "dataset": "trajectories",
        "n": n,
        "units": "tasks",
        "shards": shards,
        "content": content,
    }
