"""Tests for M6 `export_grounder.py`: runtime checkpoint layout, SHA-256 verification, int8 quantization, and parity."""

from __future__ import annotations

import os
import torch

from brainbox.checkpoints import verify_checkpoint_dir
from neurobrain.eyes_grounder import Stage1CoarseNet, Stage2RefineNet
from neurobrain.ui_world_hard_v2 import make_hard_val_split, render_hard
from neurobrain.export_grounder import export_grounder_checkpoint, load_exported_grounder


def test_export_grounder_layout_int8_and_parity(tmp_path):
    torch.manual_seed(7)
    s1 = Stage1CoarseNet()
    s2 = Stage2RefineNet()
    raw_pt = str(tmp_path / "raw_two_stage.pt")
    torch.save({"stage1": s1.state_dict(), "stage2": s2.state_dict(), "step_stage2": 100}, raw_pt)

    out_dir = str(tmp_path / "grip_v2")
    info = export_grounder_checkpoint(
        ckpt_path=raw_pt,
        out_dir=out_dir,
        step=100,
        card_markdown="# Test Card\n",
    )

    for fname in ("config.json", "weights.pt", "weights.int8.pt", "tokenizer.json", "card.md"):
        assert os.path.exists(os.path.join(out_dir, fname)), f"Missing {fname}"

    assert verify_checkpoint_dir(out_dir) is True
    assert info["weights_int8_bytes"] < info["weights_bytes"] * 0.40  # >= 2.5x smaller on disk

    g_fp32 = load_exported_grounder(out_dir, precision="fp32")
    g_int8_storage = load_exported_grounder(out_dir, precision="int8_storage")
    g_int8_dyn = load_exported_grounder(out_dir, precision="int8_dynamic")

    ex = make_hard_val_split(n=1, seed=99)[0]
    rgb = render_hard(ex.layout, size=256)

    r_fp32 = g_fp32.ground(rgb, ex.task)
    r_i8s = g_int8_storage.ground(rgb, ex.task)
    r_i8d = g_int8_dyn.ground(rgb, ex.task)

    assert abs(r_fp32["x"] - r_i8s["x"]) <= 4
    assert abs(r_fp32["y"] - r_i8s["y"]) <= 4
    assert r_i8d["action"] in ("click", "type", "toggle", "drag")
