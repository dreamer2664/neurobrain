"""Tests for Stage-1 Coarse Grounder, Stage-2 Native-Strip Crop Refiner, multi-scale candidates, and recovery exclude_box."""

from __future__ import annotations

import io
import time
import torch
import numpy as np
from PIL import Image

from neurobrain.ui_world import encode_task
from neurobrain.ui_world_hard_v2 import make_hard_val_split, render_hard
from neurobrain.eyes_grounder import (
    EyesGrounder,
    Stage1CoarseNet,
    Stage2RefineNet,
    extract_crop_tensor,
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

    cands = topk_grid_candidates(
        scores[0], offsets[0], box_wh[0], canvas_w=256, canvas_h=256, k=7, nms_radius_cells=1
    )
    assert len(cands) == 7
    first = cands[0]
    ex_box = (first["x"] - 12, first["y"] - 12, 24, 24)
    cands_excl = topk_grid_candidates(
        scores[0], offsets[0], box_wh[0], canvas_w=256, canvas_h=256, k=7, exclude_box=ex_box, nms_radius_cells=1
    )
    second = cands_excl[0]
    assert not (ex_box[0] <= second["x"] <= ex_box[0] + ex_box[2] and ex_box[1] <= second["y"] <= ex_box[1] + ex_box[3])


def test_extract_crop_tensor_and_stage2_strip_refiner():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Test both interior and screen-corner candidate anchors
    for cx, cy in [(120.0, 80.0), (8.0, 8.0), (248.0, 248.0)]:
        arr, rect = extract_crop_tensor(img, cx, cy, span_w=160.0, span_h=32.0)
        assert arr.shape == (Stage2RefineNet.CROP_H, Stage2RefineNet.CROP_W, 6)
        rx, ry, rw, rh = rect
        assert rw == 160.0 and rh == 32.0
        # Peak of beacon channel (channel 5) must sit right at (cx, cy)!
        beacon = arr[:, :, 5]
        py_idx, px_idx = np.unravel_index(int(beacon.argmax()), beacon.shape)
        est_x = rx + (px_idx + 0.5) * (rw / Stage2RefineNet.CROP_W)
        est_y = ry + (py_idx + 0.5) * (rh / Stage2RefineNet.CROP_H)
        assert abs(est_x - cx) <= 1.5
        assert abs(est_y - cy) <= 1.5

    s2 = Stage2RefineNet()
    crops = torch.randn(6, 6, Stage2RefineNet.CROP_H, Stage2RefineNet.CROP_W).clamp(-1.5, 1.5)
    t = torch.tensor([encode_task(f"click save {i}") for i in range(6)], dtype=torch.long)
    q2 = s2.encode_instruction(t)
    verify_logit, delta_uv, fine_wh = s2(crops, q2)
    assert verify_logit.shape == (6,)
    assert delta_uv.shape == (6, 2)
    assert fine_wh.shape == (6, 2)
    assert delta_uv.abs().max().item() <= 1.0

    # Matrix mode (Q=6 instructions x K=6 strips on same screen) must match diagonal!
    crop_repr = s2.encode_crops(crops)
    mat_logits = s2.score_matrix(crop_repr, q2)
    assert mat_logits.shape == (6, 6)
    assert torch.allclose(torch.diag(mat_logits), verify_logit, atol=1e-4)


def test_eyes_grounder_arbitrary_resolution_latency_and_recovery_mode():
    torch.set_num_threads(2)
    s1 = Stage1CoarseNet()
    s2 = Stage2RefineNet(d_text=128)
    grounder = EyesGrounder(stage1=s1, stage2=s2, top_k=7, nms_radius_cells=1)

    ex = make_hard_val_split(n=1, seed=11)[0]
    rgb256 = render_hard(ex.layout, size=256)
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

    # Latency budget check on 256x256 native RGB (< 60 ms/frame on 2 CPU threads)
    for _ in range(3):
        grounder.ground(rgb256, ex.task, use_stage2=True)
    t0 = time.perf_counter()
    for _ in range(15):
        grounder.ground(rgb256, ex.task, use_stage2=True)
    ms = (time.perf_counter() - t0) * 1000.0 / 15.0
    assert ms < 60.0, f"Latency {ms:.2f} ms exceeded 60 ms budget"
