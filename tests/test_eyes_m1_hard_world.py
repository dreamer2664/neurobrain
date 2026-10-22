"""M1 tests: Hard UI world v1 generator, OOD font/theme isolation, shard format & sha256 verification."""

from __future__ import annotations

import json
import os
import random
import tempfile

import numpy as np

from neurobrain.ui_world import ACTION_CLASSES, LABELS, ROLES
from neurobrain.ui_world_hard import (
    ARCHETYPES,
    HARD_CANVAS,
    OOD_FONT_PATHS,
    OOD_THEMES,
    SEEN_FONT_PATHS,
    SEEN_THEMES,
    build_hard_shards,
    make_hard_val_split,
    render_hard,
    sample_hard_example,
    verify_hard_shards,
)


def test_seen_vs_ood_disjoint():
    assert set(SEEN_THEMES).isdisjoint(set(OOD_THEMES))
    assert set(SEEN_FONT_PATHS).isdisjoint(set(OOD_FONT_PATHS))
    assert len(SEEN_FONT_PATHS) >= 2
    assert len(OOD_FONT_PATHS) >= 2


def test_deterministic_and_valid_geometry():
    s1 = make_hard_val_split(n=30, seed=202604, split="seen")
    s2 = make_hard_val_split(n=30, seed=202604, split="seen")
    for a, b in zip(s1, s2):
        assert a.task == b.task
        assert a.target == b.target
        assert a.layout.theme == b.layout.theme
        assert a.layout.archetype == b.layout.archetype
        img_a = render_hard(a.layout)
        img_b = render_hard(b.layout)
        assert np.array_equal(img_a, img_b)
        assert img_a.shape == (HARD_CANVAS, HARD_CANVAS, 3)
        assert img_a.dtype == np.uint8
        for el in a.layout.elements:
            assert el.role in ROLES
            assert el.label in LABELS
            assert 0 <= el.x < el.x + el.w <= HARD_CANVAS
            assert 0 <= el.y < el.y + el.h <= HARD_CANVAS
            assert el.w >= 14 and el.h >= 10


def test_all_archetypes_tiers_and_modes():
    rng = random.Random(123)
    seen_arch = set()
    seen_states = set()
    for tier in ("easy", "medium", "crowded"):
        for _ in range(25):
            ex = sample_hard_example(rng, tier=tier, split="seen")
            seen_arch.add(ex.layout.archetype)
            for el in ex.layout.elements:
                seen_states.add(el.state)
            assert ex.action in ACTION_CLASSES
    assert seen_arch == set(ARCHETYPES)
    assert {"normal", "focused", "selected", "disabled"}.issubset(seen_states)

    # OOD split uses strictly OOD themes & fonts
    for _ in range(15):
        ex_ood = sample_hard_example(rng, tier="mixed", split="ood")
        assert ex_ood.layout.theme in OOD_THEMES
        assert ex_ood.layout.font_path in OOD_FONT_PATHS

    # Text vs glyph ablation mode renders valid distinct pixels
    ex_t = sample_hard_example(random.Random(77), tier="medium", label_mode="text")
    ex_g = sample_hard_example(random.Random(77), tier="medium", label_mode="glyph")
    im_t = render_hard(ex_t.layout)
    im_g = render_hard(ex_g.layout)
    assert not np.array_equal(im_t, im_g)


def test_shard_builder_and_manifest_verification():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_hard_shards(
            out_dir=tmp,
            total_samples=64,
            shard_size=32,
            seed=101,
            workers=1,
        )
        assert manifest["n"] == 64
        assert manifest["version"] == "v1"
        assert manifest["generated_by"] == "agent-03"
        assert len(manifest["shards"]) == 2

        ok, detail = verify_hard_shards(tmp)
        assert ok, detail

        shard0 = np.load(os.path.join(tmp, manifest["shards"][0]["file"]))
        assert shard0["images"].shape == (32, 128, 128, 3)
        assert shard0["images"].dtype == np.uint8
        assert shard0["tasks"].shape == (32, 48)
        assert shard0["tasks"].dtype == np.int64
        assert shard0["heat"].shape == (32, 32, 32)
        assert shard0["heat"].dtype == np.float16
        assert shard0["offsets"].shape == (32, 2)
        assert shard0["offsets"].dtype == np.float32
        assert shard0["actions"].shape == (32,)
        assert shard0["actions"].dtype == np.int64

        # Corrupt manifest hash -> verify must fail
        mpath = os.path.join(tmp, "manifest.json")
        with open(mpath) as f:
            m = json.load(f)
        m["shards"][0]["sha256"] = "0" * 64
        with open(mpath, "w") as f:
            json.dump(m, f)
        ok_bad, _ = verify_hard_shards(tmp)
        assert not ok_bad
