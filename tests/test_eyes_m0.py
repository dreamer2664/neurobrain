"""M0 tests: deterministic validation split, diagnostic metrics, confusion tracking, and 100-miss failure gallery."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

from neurobrain.gripnet import GripNet
from neurobrain.diagnose_grip import (
    make_easy_val_split,
    diagnose_predictions,
    evaluate_gripnet_model,
    render_failure_gallery,
)


def test_deterministic_easy_val_split():
    s1 = make_easy_val_split(n=32, seed=202603)
    s2 = make_easy_val_split(n=32, seed=202603)
    assert len(s1) == 32
    for a, b in zip(s1, s2):
        assert a.task == b.task
        assert a.action == b.action
        assert a.target == b.target
        assert a.layout.theme == b.layout.theme
        assert len(a.layout.elements) == len(b.layout.elements)


def test_diagnose_oracle_and_misses():
    split = make_easy_val_split(n=40, seed=42)
    # Oracle predictions: exact target centers + right action
    from neurobrain.ui_world import ACTION_CLASSES

    oracle_preds = [
        (ex.target.cx, ex.target.cy, ACTION_CLASSES.index(ex.action))
        for ex in split
    ]
    rep, misses = diagnose_predictions(split, oracle_preds, canvas_w=128, canvas_h=128)
    assert rep["n"] == 40
    assert rep["hit_widget"] == 1.0
    assert rep["median_center_err_px"] == 0.0
    assert rep["action_acc"] == 1.0
    assert len(misses) == 0

    # Deliberately click a distractor element on every screen to verify confusion tracking
    bad_preds = []
    for ex in split:
        others = [e for e in ex.layout.elements if e is not ex.target]
        other = others[0]
        bad_preds.append((other.cx, other.cy, 0))
    rep_bad, misses_bad = diagnose_predictions(split, bad_preds, canvas_w=128, canvas_h=128)
    assert rep_bad["hit_widget"] < 0.2
    assert len(misses_bad) >= 32
    assert rep_bad["miss_taxonomy"]["same_role_wrong_label"] + rep_bad["miss_taxonomy"]["wrong_role_widget"] >= 30
    assert len(rep_bad["worst_confusions"]["label_confusions"]) > 0
    assert "by_role" in rep_bad and "by_label" in rep_bad and "by_theme" in rep_bad and "by_size" in rep_bad


def test_failure_gallery_renders_100_tiles():
    split = make_easy_val_split(n=120, seed=7)
    preds = [(0.0, 0.0, 0) for _ in split]
    _, misses = diagnose_predictions(split, preds, canvas_w=128, canvas_h=128)
    assert len(misses) >= 100
    with tempfile.TemporaryDirectory() as tmp:
        out_png = os.path.join(tmp, "gallery.png")
        n_drawn = render_failure_gallery(misses, out_png, max_tiles=100)
        assert n_drawn == 100
        assert os.path.exists(out_png)
        im = Image.open(out_png)
        assert im.width > 500 and im.height > 500


def test_evaluate_gripnet_model_smoke():
    model = GripNet()
    split = make_easy_val_split(n=24, seed=99)
    rep, misses = evaluate_gripnet_model(model, split, threads=2)
    assert 0.0 <= rep["hit_widget"] <= 1.0
    assert rep["latency_ms_per_frame"] > 0.0
    assert rep["peak_rss_mb"] > 0.0
    assert rep["n"] == 24
