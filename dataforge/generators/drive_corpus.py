"""The drive corpus at scale: persistence episodes as a token stream.

At M0 this wraps ``neurobrain.drive.drive_corpus`` (the seed generators) and
adds what scale needs: streaming to disk, a manifest, and the three invariants
checked **while** generating, so a broken episode never reaches a trainer.

M4 replaces the seed generators with the extended ones (wrong-first-attempt,
near-miss, recovery; code bug -> test error -> patch -> rerun; plan obstacle ->
replan) while keeping this file's output format identical.
"""
from __future__ import annotations

import os
import sys

from .. import drive_rules
from ..profiles import sub_seed
from ..streams import TokenStreamWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.drive import drive_corpus  # noqa: E402


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    episodes = int(cfg["episodes"])
    n_val = max(20, int(cfg["val_frac"] * episodes))
    eot = tok.encode("<|endoftext|>")[0]

    shards = []
    all_rows: list[dict] = []
    for split, count, sd in (("train", episodes - n_val, sub_seed(seed, "drive_train")),
                             ("val", n_val, sub_seed(seed, "drive_val"))):
        w = TokenStreamWriter(os.path.join(dirpath, f"drive_{split}.bin"))
        rows = []
        for i, ep in enumerate(drive_corpus(count, seed=sd)):
            r = drive_rules.classify(ep)
            rows.append(r)
            w.write(tok.encode(ep) + [eot])
            if (i + 1) % 100_000 == 0:
                log(f"[drive] {split}: {i+1:,} episodes, {w.tokens/1e6:.1f}M tokens")
        rec = w.close()
        summary = drive_rules.summarise(rows)
        bad = drive_rules.check(summary)
        shards.append(rec | {"invariants": {k: summary[k] for k in
                                            ("terminal_rate", "retry_after_failure_rate",
                                             "failure_fraction", "mix", "mean_retries")},
                             "violations": bad})
        log(f"[drive] {rec['name']}: {rec['docs']:,} episodes, {rec['tokens']/1e6:.2f}M tokens, "
            f"terminal={summary['terminal_rate']} retry_ok={summary['retry_after_failure_rate']} "
            f"failure_frac={summary['failure_fraction']}")
        if bad:
            raise RuntimeError("drive invariants violated at build time: " + "; ".join(bad))
        all_rows.extend(rows)

    summary = drive_rules.summarise(all_rows)
    return {
        "kind": "token_streams",
        "dataset": "drive",
        "n": sum(s["docs"] for s in shards),
        "units": "episodes",
        "shards": shards,
        "content": {
            "source": "neurobrain.drive.drive_corpus (seed)",
            "invariants": {k: summary[k] for k in
                           ("terminal_rate", "retry_after_failure_rate",
                            "failure_fraction", "mix", "mean_retries")},
        },
    }
