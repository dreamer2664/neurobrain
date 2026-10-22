"""UI grounding shards in the exact format of specs/data_formats.md.

This is the **seed** world (``neurobrain.ui_world``) emitted properly: 5k
samples per ``.npz``, deterministic bytes, a manifest with a sha256 per shard.
``hard_ui`` (M1/M2) reuses this writer and only swaps the renderer.

Peak RSS is one image: samples go straight into an on-disk memmap and the shard
is streamed into the zip. 100k samples never sit in RAM together.
"""
from __future__ import annotations

import hashlib
import os
import random
import sys

import numpy as np

from ..profiles import sub_seed
from ..streams import ShardAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.ui_world import (  # noqa: E402
    ACTION_CLASSES, CANVAS, LABELS, ROLES, encode_task, render,
    sample_example, target_heatmap, target_offset,
)

THEMES = ["light", "dark"]


def heatmap_vec(ex, grid: int = 32, sigma: float = 0.9) -> np.ndarray:
    """Bit-identical to ``ui_world.target_heatmap`` but without the 1024-iteration loop."""
    cell = CANVAS / grid
    gx, gy = ex.target.cx / cell - 0.5, ex.target.cy / cell - 0.5
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    d2 = (yy - gy) ** 2 + (xx - gx) ** 2
    hm = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
    return (hm / hm.sum()).astype(np.float32)


def dhash(image: np.ndarray, bits: int = 8) -> int:
    """Tiny perceptual hash: 8x8 horizontal differences -> 64 bits."""
    small = image[:: max(1, image.shape[0] // (bits + 1)), :: max(1, image.shape[1] // (bits + 1))]
    small = small[: bits + 1, :bits, 0].astype(np.int16)
    diff = small[1:] > small[:-1]
    return int("".join("1" if b else "0" for b in diff.reshape(-1)), 2)


def image_entropy(image: np.ndarray) -> float:
    hist = np.bincount(image.reshape(-1, 3).mean(axis=1).astype(np.uint8), minlength=256)
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    total = int(cfg["samples"])
    shard_size = int(cfg.get("shard_size", 5000))
    if total <= 0:
        raise SystemExit("ui_seed: samples must be > 0")

    rng = random.Random(sub_seed(seed, "ui_seed"))
    tmpdir = os.path.join(dirpath, ".tmp")
    acc = ShardAccumulator(dirpath=dirpath, shard_size=shard_size, index=0, tmpdir=tmpdir)

    label_hist = np.zeros(len(LABELS), dtype=np.int64)
    role_hist = np.zeros(len(ROLES), dtype=np.int64)
    action_hist = np.zeros(len(ACTION_CLASSES), dtype=np.int64)
    theme_hist = np.zeros(2, dtype=np.int64)
    digests: set[str] = set()
    near: dict[int, int] = {}
    exact_dupes = neardupes = 0
    entropies: list[float] = []
    shards: list[dict] = []

    for i in range(total):
        ex = sample_example(rng)
        img = render(ex.layout)
        dig = hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
        if dig in digests:
            exact_dupes += 1
        digests.add(dig)
        dh = dhash(img)
        near[dh] = near.get(dh, 0) + 1

        cell, off = target_offset(ex, 32)
        acc.add(
            image=img,
            task_ids=encode_task(ex.task),
            heat=heatmap_vec(ex).astype(np.float16),
            offset=off,
            action=ACTION_CLASSES.index(ex.action),
            label=LABELS.index(ex.target.label),
            role=ROLES.index(ex.target.role),
            theme=THEMES.index(ex.layout.theme),
        )
        label_hist[LABELS.index(ex.target.label)] += 1
        role_hist[ROLES.index(ex.target.role)] += 1
        action_hist[ACTION_CLASSES.index(ex.action)] += 1
        theme_hist[THEMES.index(ex.layout.theme)] += 1
        if i % 50 == 0:
            entropies.append(image_entropy(img))

        if acc.full:
            rec = acc.flush()
            if rec:
                shards.append(rec)
                log(f"[ui_seed] {rec['name']}: {rec['n']} samples, {rec['bytes']/1e6:.1f} MB")

    rec = acc.flush()
    if rec:
        shards.append(rec)
        log(f"[ui_seed] {rec['name']}: {rec['n']} samples, {rec['bytes']/1e6:.1f} MB")
    acc.close()

    neardupes = sum(c - 1 for c in near.values() if c > 1)
    content = {
        "source": "neurobrain.ui_world (seed world)",
        "label_hist": label_hist.tolist(),
        "role_hist": role_hist.tolist(),
        "action_hist": action_hist.tolist(),
        "theme_hist": theme_hist.tolist(),
        "exact_dupes": exact_dupes,
        "near_dupes": neardupes,
        "entropy_mean": round(float(np.mean(entropies)) if entropies else 0.0, 4),
        "entropy_std": round(float(np.std(entropies)) if entropies else 0.0, 4),
    }
    log(f"[ui_seed] {total} samples, dupes exact={exact_dupes} near={neardupes}, "
        f"entropy={content['entropy_mean']}±{content['entropy_std']}")
    return {
        "kind": "ui_shards",
        "dataset": "ui_seed",
        "n": sum(s["n"] for s in shards),
        "units": "samples",
        "shards": shards,
        "content": content,
    }
