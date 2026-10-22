"""Profile table: the only place corpus sizes are written down.

``smoke`` exists so the whole pipeline (build -> stats -> verify) runs in a
couple of minutes inside a test; ``standard`` is the number the trainers
consume; ``max`` is what a longer sandbox window buys.

Every knob is a count. Nothing here is a path or a byte offset: the generators
derive their shard boundaries from these counts, so a profile fully determines
the output of a build together with the seed.
"""

from __future__ import annotations

# dataset id -> (implemented?, unit)
DATASETS = {
    "code": ("implemented", "docs"),
    "ui_protocol": ("implemented", "docs"),
    "drive": ("implemented", "episodes"),
    "ui_seed": ("implemented", "samples"),
    "hard_ui": ("todo", "samples"),          # M1/M2
    "instructions": ("implemented", "instructions"),  # M3
    "trajectories": ("implemented", "tasks"),  # M5
    "code_v2": ("implemented", "functions"),  # M6
}

BUILD_ORDER = ["code", "ui_protocol", "drive", "ui_seed",
               "hard_ui", "instructions", "trajectories", "code_v2"]

PROFILES: dict[str, dict] = {
    "smoke": {
        "code":        {"max_bytes": 1_500_000, "val_frac": 0.03, "min_bytes": 400},
        "ui_protocol": {"n": 2_000, "val_frac": 0.03},
        "drive":       {"episodes": 3_000, "val_frac": 0.02},
        "ui_seed":     {"samples": 10_000, "shard_size": 5_000},
        "hard_ui":     {"samples": 2_000, "shard_size": 5_000},
        "instructions": {"n": 20_000, "val_frac": 0.01},
        "trajectories": {"n": 2_000},
        "code_v2":     {"max_bytes": 2_000_000, "val_frac": 0.03, "min_bytes": 400,
                        "tok_bytes": 2_000_000},
    },
    "standard": {
        "code":        {"max_bytes": 40_000_000, "val_frac": 0.03, "min_bytes": 400},
        "ui_protocol": {"n": 150_000, "val_frac": 0.03},
        "drive":       {"episodes": 1_000_000, "val_frac": 0.025},
        "ui_seed":     {"samples": 25_000, "shard_size": 5_000},
        "hard_ui":     {"samples": 100_000, "shard_size": 5_000},
        "instructions": {"n": 1_000_000, "val_frac": 0.005},
        "trajectories": {"n": 50_000},
        "code_v2":     {"max_bytes": 40_000_000, "val_frac": 0.03, "min_bytes": 400,
                        "tok_bytes": 12_000_000},
    },
    "max": {
        "code":        {"max_bytes": 160_000_000, "val_frac": 0.03, "min_bytes": 400},
        "ui_protocol": {"n": 500_000, "val_frac": 0.03},
        "drive":       {"episodes": 5_000_000, "val_frac": 0.02},
        "ui_seed":     {"samples": 50_000, "shard_size": 5_000},
        "hard_ui":     {"samples": 300_000, "shard_size": 5_000},
        "instructions": {"n": 5_000_000, "val_frac": 0.002},
        "trajectories": {"n": 200_000},
        "code_v2":     {"max_bytes": 160_000_000, "val_frac": 0.03, "min_bytes": 400,
                        "tok_bytes": 40_000_000},
    },
}


def resolve(profile: str) -> dict:
    if profile not in PROFILES:
        raise SystemExit(
            f"dataforge: unknown profile {profile!r}; choose from {sorted(PROFILES)}"
        )
    return PROFILES[profile]


def sub_seed(seed: int, *parts: int | str) -> int:
    """A stable child seed. Never uses Python's salted ``hash``."""
    h = (seed * 1_000_003) & 0xFFFFFFFF
    for p in parts:
        if isinstance(p, str):
            p = sum((i + 1) * ord(c) for i, c in enumerate(p))
        h = (h * 31 + int(p) + 7) & 0xFFFFFFFF
    return h
