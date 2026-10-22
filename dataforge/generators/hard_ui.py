"""Hard UI worlds as grounding shards (M1/M2).

Same on-disk format as ``ui_seed`` (specs/data_formats.md) so agent 03 can swap
the directory without touching the loader, plus two extra arrays:
``tiers`` (0/1/2 difficulty) and ``dpi`` (100-200).

The manifest records the difficulty mix and the per-tier counts, so a trainer
can hold tiers out or curriculum on them.
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

from neurobrain.ui_world import ACTION_OF_ROLE, LABELS, ROLES, encode_task  # noqa: E402
from neurobrain.ui_world_hard import (  # noqa: E402
    THEME_NAMES, heatmap_vec, render, sample_example, target_offset,
)
from .ui_seed import dhash, image_entropy  # noqa: E402

ACTION_ORDER = ["click", "type", "toggle", "drag"]


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    total = int(cfg["samples"])
    shard_size = int(cfg.get("shard_size", 5000))
    if total <= 0:
        raise SystemExit("hard_ui: samples must be > 0")

    rng = random.Random(sub_seed(seed, "hard_ui"))
    tmpdir = os.path.join(dirpath, ".tmp")
    acc = ShardAccumulator(dirpath=dirpath, shard_size=shard_size, index=0, tmpdir=tmpdir)
    acc.extra = {}  # type: ignore[attr-defined]

    label_hist = np.zeros(len(LABELS), dtype=np.int64)
    role_hist = np.zeros(len(ROLES), dtype=np.int64)
    action_hist = np.zeros(len(ACTION_ORDER), dtype=np.int64)
    theme_hist = np.zeros(len(THEME_NAMES), dtype=np.int64)
    tier_hist = np.zeros(3, dtype=np.int64)
    dpi_hist = np.zeros(4, dtype=np.int64)
    section_hist = np.zeros(6, dtype=np.int64)
    tiers, dpis, sections = [], [], []
    digests: set[str] = set()
    near: dict[int, int] = {}
    exact_dupes = 0
    entropies: list[float] = []
    shards: list[dict] = []
    SECTIONS = ["toolbar", "tabs", "sidebar", "content", "dialog", "title"]

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
            action=ACTION_ORDER.index(ACTION_OF_ROLE[ex.target.role]),
            label=LABELS.index(ex.target.label),
            role=ROLES.index(ex.target.role),
            theme=THEME_NAMES.index(ex.layout.theme),
        )
        tiers.append(ex.layout.tier)
        dpis.append([100, 125, 150, 200].index(ex.layout.dpi))
        sections.append(SECTIONS.index(ex.target.section) if ex.target.section in SECTIONS else 3)
        label_hist[LABELS.index(ex.target.label)] += 1
        role_hist[ROLES.index(ex.target.role)] += 1
        action_hist[ACTION_ORDER.index(ACTION_OF_ROLE[ex.target.role])] += 1
        theme_hist[THEME_NAMES.index(ex.layout.theme)] += 1
        tier_hist[ex.layout.tier] += 1
        dpi_hist[[100, 125, 150, 200].index(ex.layout.dpi)] += 1
        if i % 25 == 0:
            entropies.append(image_entropy(img))

        if acc.full:
            rec = acc.flush()
            if rec:
                shards.append(rec)
                log(f"[hard_ui] {rec['name']}: {rec['n']} samples, {rec['bytes']/1e6:.1f} MB")

    rec = acc.flush()
    if rec:
        shards.append(rec)
        log(f"[hard_ui] {rec['name']}: {rec['n']} samples, {rec['bytes']/1e6:.1f} MB")
    acc.close()

    _write_sidecar(dirpath, shards, tiers, dpis, sections)
    near_dupes = sum(c - 1 for c in near.values() if c > 1)
    content = {
        "source": "neurobrain.ui_world_hard",
        "label_hist": label_hist.tolist(),
        "role_hist": role_hist.tolist(),
        "action_hist": action_hist.tolist(),
        "theme_hist": theme_hist.tolist(),
        "tier_hist": tier_hist.tolist(),
        "dpi_hist": dpi_hist.tolist(),
        "section_hist": section_hist.tolist(),
        "sections": SECTIONS,
        "exact_dupes": exact_dupes,
        "near_dupes": near_dupes,
        "entropy_mean": round(float(np.mean(entropies)) if entropies else 0.0, 4),
        "entropy_std": round(float(np.std(entropies)) if entropies else 0.0, 4),
    }
    log(f"[hard_ui] {total} samples, tiers {tier_hist.tolist()}, "
        f"themes {theme_hist.tolist()}, dupes exact={exact_dupes} near={near_dupes}, "
        f"entropy {content['entropy_mean']}")
    return {
        "kind": "ui_shards",
        "dataset": "hard_ui",
        "n": sum(s["n"] for s in shards),
        "units": "samples",
        "shards": shards,
        "content": content,
    }


def _write_sidecar(dirpath: str, shards: list[dict], tiers, dpis, sections) -> None:
    """Per-shard difficulty index: which tier/dpi/section each sample targets.

    Kept out of the npz so the specced keys stay specced; a trainer that wants
    tier curriculums reads ``tiers/<shard>.npy``.
    """
    tdir = os.path.join(dirpath, "tiers")
    os.makedirs(tdir, exist_ok=True)
    pos = 0
    for sh in shards:
        n = sh["n"]
        base = sh["name"].replace(".npz", "")
        np.save(os.path.join(tdir, f"{base}-tier.npy"), np.asarray(tiers[pos:pos + n], np.int8))
        np.save(os.path.join(tdir, f"{base}-dpi.npy"), np.asarray(dpis[pos:pos + n], np.int8))
        np.save(os.path.join(tdir, f"{base}-section.npy"), np.asarray(sections[pos:pos + n], np.int8))
        pos += n
