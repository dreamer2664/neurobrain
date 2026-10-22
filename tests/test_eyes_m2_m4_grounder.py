"""Tests for Stage-1 Coarse Grounder, Stage-2 Crop Refiner, multi-scale candidates, and recovery-conditioned exclude_box."""

from __future__ import annotations

import io
import torch
import numpy as np
from PIL import Image

from neurobrain.ui_world import encode_task
from neurobrain.ui_world_hard_v2 import make_hard_val_split, render_hard
from neurobrain.eyes_grounder import (
    EyesGrounder,
    Stage1CoarseNet,
    Stage2RefineNet,
    extract_visual_channels,
    topk_grid_candidates,
)


def test_visual_channels_theme_contrast():
    # Dark text on white vs white text on dark at exact same location -> identical |contrast|!
    light = torch.full((1, 3, 64, 64), 0.9)
    light[:, :, 20:28, 20:40] = -0.9
    dark = -light
    ch_l = extract_visual_channels(light)
    ch_d = extract_visual_channels(dark)
    assert ch_l.shape == (1, 7, 64, 64)
    # Channel 4 is |local_contrast|, channels 5,6 are |dx|, |dy| -> invariant to polarity flip!
    assert torch.allclose(ch_l[:, 4:], ch_d[:, 4:], atol=1e-4)


def test_stage1_and_topk_nms_and_exclude_box():
    net = Stage1CoarseNet()
    x = torch.randn(2, 3, 128, 128).clamp(-1, 1)
    t = torch.tensor([encode_task("click the save button"), encode_task("open settings tab")], dtype=torch.long)
    scores, offsets, box_wh, act_logits, q = net(x, t)
    assert scores.shape == (2, 1024)
    assert offsets.shape == (2, 1024, 2)
    assert box_wh.shape == (2, 1024, 2)
    assert act_logits.shape == (2, 4)
    assert q.shape[0] == 2

    cands = topk_grid_candidates(scores[0], offsets[0], box_wh[0], canvas_w=256, canvas_h=256, k=4)
    assert len(cands) == 4
    first = cands[0]
    ex_box = (first["x"] - 12, first["y"] - 12, 24, 24)
    cands_excl = topk_grid_candidates(
        scores[0], offsets[0], box_wh[0], canvas_w=256, canvas_h=256, k=4, exclude_box=ex_box
    )
    second = cands_excl[0]
    assert not (ex_box[0] <= second["x"] <= ex_box[0] + ex_box[2] and ex_box[1] <= second["y"] <= ex_box[1] + ex_box[3])


def test_stage2_refiner_shapes():
    s1 = Stage1CoarseNet()
    s2 = Stage2RefineNet(d_text=s1.d_text)
    crops = torch.randn(6, 3, 64, 64).clamp(-1, 1)
    t = torch.tensor([encode_task("click save")] * 6, dtype=torch.long)
    q = s1.encode_instruction(t)
    verify_logit, fine_xy, fine_wh = s2(crops, q)
    assert verify_logit.shape == (6,)
    assert fine_xy.shape == (6, 2)
    assert fine_wh.shape == (6, 2)
    assert fine_xy.abs().max().item() <= 1.0


def test_eyes_grounder_arbitrary_resolution_and_recovery_mode():
    s1 = Stage1CoarseNet()
    s2 = Stage2RefineNet(d_text=s1.d_text)
    grounder = EyesGrounder(stage1=s1, stage2=s2)

    ex = make_hard_val_split(n=1, seed=11)[0]
    rgb256 = render_hard(ex.layout, size=256)
    # Encode as 512x384 PNG bytes to verify arbitrary-aspect PNG decoding & coordinate scaling
    im_big = Image.fromarray(rgb256).resize((512, 384), Image.BILINEAR)
    buf = io.BytesIO()
    im_big.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res1 = grounder.ground(png_bytes, ex.task, use_stage2=True, multi_scale=True)
    assert 0 <= res1["x"] < 512
    assert 0 <= res1["y"] < 384
    assert res1["action"] in ("click", "type", "toggle", "drag")
    assert "box" in res1

    # Recovery-conditioned second call excluding a box around first prediction
    excl = (max(0, res1["x"] - 30), max(0, res1["y"] - 30), 60, 60)
    res2 = grounder.ground(png_bytes, ex.task, exclude_box=excl, use_stage2=True)
    assert not (excl[0] <= res2["x"] <= excl[0] + excl[2] and excl[1] <= res2["y"] <= excl[1] + excl[3])
