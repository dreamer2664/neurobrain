"""Range table. A number outside these bounds fails the build.

The bounds are deliberately tight enough to catch a broken generator and loose
enough to survive a different seed. Every one of them was set after measuring
the seed corpora; the measured values live in agents/05-forge/NOTES.md.
"""

from __future__ import annotations

import math

RANGES = {
    # duplicates
    "dup_rate_max": 0.02,              # exact, sampled docs/images
    "near_dup_rate_max": 0.05,
    # token balance
    "special_share_max": 0.30,         # markers + EOT
    "alpha_share_min": 0.30,           # streams must be mostly letters
    "vocab_used_min": 512,             # distinct token ids in a stream
    # label / action balance (share of the most common class)
    "label_entropy_ratio_min": 0.90,   # H / log2(n_labels)
    "label_min_share_of_uniform": 0.40,
    "action_entropy_ratio_min": 0.50,
    "action_min_share_min": 0.06,      # every action class must be learnable
    "theme_min_share": 0.20,           # both light and dark present
    # coordinates
    "heat_entropy_ratio_min": 0.45,    # H of argmax-cell histogram / log2(1024)
    "heat_max_cell_share_max": 0.06,
    "heat_cells_used_min": 150,
    "offset_clipped_max": 0.05,
    # target centres are uniform inside a cell, so mean |offset| is ~0.5 in a
    # healthy world; near 1.0 means the regression target saturates, near 0
    # means every target is parked on a cell centre. Two-sided on purpose.
    "offset_mean_abs_band": (0.30, 0.60),
    # pixels
    "entropy_bits_min": 2.5,
    "entropy_bits_max": 8.0,
    # drive invariants (hard requirements from the brief)
    "terminal_rate_min": 1.0,
    "retry_after_failure_min": 1.0,
    "failure_fraction_min": 0.50,
    # format sanity
    "image_black_share_max": 0.60,     # a shard of near-blank screens is broken
}


# Measured properties of the *seed* artefacts, not of the format. Each entry is
# a number that the seed world cannot meet and the hard world (M1/M2) must:
#   ui_seed      heat cells 72/1024 and 51% clipped offsets are real properties
#                of ui_world's 4x3 coarse placement grid and its 0.45 cell
#                margin (the click target saturates at the cell edge). The seed
#                world is also mostly flat fill -> 2.26 bits of entropy.
#   ui_protocol  a serialized element tree is mostly punctuation, not letters.
#   drive        the seed generators reuse a small vocabulary.
# Every relaxed bound is printed by `stats` so nobody has to guess which
# dataset is running with a weaker gate.
DATASET_OVERRIDES = {
    "ui_seed": {"heat_cells_used_min": 60, "heat_entropy_ratio_min": 0.45,
                "offset_clipped_max": 0.60, "offset_mean_abs_band": (0.30, 0.80),
                "entropy_bits_min": 2.0},
    "ui_protocol": {"alpha_share_min": 0.15, "vocab_used_min": 150},
    "drive": {"vocab_used_min": 380},
}


def effective_ranges(dataset: str) -> dict:
    r = dict(RANGES)
    r.update(DATASET_OVERRIDES.get(dataset, {}))
    return r


def entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def entropy_ratio(counts) -> float:
    n = sum(1 for c in counts if c > 0)
    if n <= 1:
        return 1.0
    return entropy(counts) / math.log2(n)


def balanced(counts, decay: float = 0.1) -> float:
    """Share of samples that would be left after dropping every surplus copy.

    This is the fraction the fleet cares about -- 1.0 means every class is
    exactly as frequent as the smallest one. It is reported next to entropy
    because entropy is blind to a single rare class.
    """
    total = sum(counts)
    present = [c for c in counts if c > 0]
    if not present or total == 0:
        return 0.0
    k = len(counts)
    return min(1.0, (min(present) * k) / total)


def check_dataset(d: dict) -> list[str]:
    """Violations for one dataset's stats dict. Empty list == in range."""
    bad: list[str] = []
    r = effective_ranges(d.get("dataset") or "")
    dup = d.get("duplicates", {})
    if dup.get("exact_rate", 0.0) > r["dup_rate_max"]:
        bad.append(f"dup_rate {dup['exact_rate']:.4f} > {r['dup_rate_max']}")
    if dup.get("near_rate", 0.0) > r["near_dup_rate_max"]:
        bad.append(f"near_dup_rate {dup['near_rate']:.4f} > {r['near_dup_rate_max']}")

    if d.get("kind") == "token_streams":
        b = d["balance"]
        if b["_share"]["special"] > r["special_share_max"]:
            bad.append(f"special_share {b['_share']['special']:.3f} > {r['special_share_max']}")
        if b["_share"]["alpha"] < r["alpha_share_min"]:
            bad.append(f"alpha_share {b['_share']['alpha']:.3f} < {r['alpha_share_min']}")
        if d["vocab_used"] < r["vocab_used_min"]:
            bad.append(f"vocab_used {d['vocab_used']} < {r['vocab_used_min']}")

    if d.get("kind") == "ui_shards":
        lb = d["label_balance"]
        if lb["entropy_ratio"] < r["label_entropy_ratio_min"]:
            bad.append(f"label_entropy_ratio {lb['entropy_ratio']:.3f} < {r['label_entropy_ratio_min']}")
        if lb["min_share"] < r["label_min_share_of_uniform"] / max(1, lb["n_labels"]):
            bad.append(f"label_min_share {lb['min_share']:.4f} too low")
        if d["action_balance"]["entropy_ratio"] < r["action_entropy_ratio_min"]:
            bad.append(f"action_entropy_ratio {d['action_balance']['entropy_ratio']:.3f} < "
                       f"{r['action_entropy_ratio_min']}")
        if d["action_balance"]["min_share"] < r["action_min_share_min"]:
            bad.append(f"action_min_share {d['action_balance']['min_share']:.3f} < "
                       f"{r['action_min_share_min']} (a class is too rare to learn)")
        # every theme that the world defines must be represented (the seed
        # world has two, the hard world three), and none below 20%
        tb = [c for c in d["theme_balance"]["counts"] if c > 0]
        if len(tb) < 2 or min(tb) / max(1, sum(tb)) < r["theme_min_share"]:
            bad.append(f"theme_balance {d['theme_balance']['counts']}: a theme below "
                       f"{int(100*r['theme_min_share'])}% or missing")
        co = d["coordinate"]
        if co["entropy_ratio"] < r["heat_entropy_ratio_min"]:
            bad.append(f"heat_entropy_ratio {co['entropy_ratio']:.3f} < {r['heat_entropy_ratio_min']}")
        if co["max_cell_share"] > r["heat_max_cell_share_max"]:
            bad.append(f"heat_max_cell_share {co['max_cell_share']:.4f} > {r['heat_max_cell_share_max']}")
        if co["cells_used"] < r["heat_cells_used_min"]:
            bad.append(f"heat_cells_used {co['cells_used']} < {r['heat_cells_used_min']}")
        if co["offset_clipped"] > r["offset_clipped_max"]:
            bad.append(f"offset_clipped {co['offset_clipped']:.4f} > {r['offset_clipped_max']}")
        lo, hi = r["offset_mean_abs_band"]
        if not (lo <= co["offset_mean_abs"] <= hi):
            bad.append(f"offset_mean_abs {co['offset_mean_abs']:.3f} outside [{lo}, {hi}]")
        e = d["image_entropy"]
        if not (r["entropy_bits_min"] <= e["mean"] <= r["entropy_bits_max"]):
            bad.append(f"image_entropy {e['mean']:.2f} outside "
                       f"[{r['entropy_bits_min']}, {r['entropy_bits_max']}]")
        if e["black_share"] > r["image_black_share_max"]:
            bad.append(f"black_share {e['black_share']:.2f} > {r['image_black_share_max']}")

    if "invariants" in d:
        inv = d["invariants"]
        if inv["terminal_rate"] < r["terminal_rate_min"]:
            bad.append(f"I1 terminal_rate {inv['terminal_rate']} < {r['terminal_rate_min']}")
        if inv["retry_after_failure_rate"] < r["retry_after_failure_min"]:
            bad.append(f"I2 retry_after_failure_rate {inv['retry_after_failure_rate']} < "
                       f"{r['retry_after_failure_min']}")
        if inv["failure_fraction"] < r["failure_fraction_min"]:
            bad.append(f"I3 failure_fraction {inv['failure_fraction']} < {r['failure_fraction_min']}")
    return bad
