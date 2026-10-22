"""Trainer & ablation runner for Agent 03 (`eyes`):

Supports:
  1. In-domain Baseline `GripNet` (0.44M) training on 50k Hard World shards (M2 baseline bar).
  2. `Stage1CoarseNet` shard training + `label-as-text vs glyph` ablation (M2 & M3).
  3. `Stage2RefineNet` native-crop verifier & sub-pixel pointer training with Stage-1 candidate mining (M4).
  4. Curriculum (`easy -> medium -> crowded`) + multi-target & confusable hard-negative mining (M5 & M6).
  5. Full diagnostic + 100-miss gallery + OOD generalisation evaluation every milestone.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .diagnose_grip import diagnose_predictions, peak_rss_mb, render_failure_gallery
from .eyes_grounder import (
    EyesGrounder,
    Stage1CoarseNet,
    Stage2RefineNet,
    extract_crop_tensor,
    topk_grid_candidates,
)
from .gripnet import GripNet
from .ui_world import ACTION_CLASSES, ACTION_OF_ROLE, CANVAS as COARSE_CANVAS, encode_task
from .ui_world_hard_v2 import (
    HARD_CANVAS,
    TASKS_HARD,
    make_hard_layout,
    make_hard_val_split,
    render_hard,
    sample_hard_example,
    target_heatmap_fast,
    target_offset_fast,
    verify_hard_shards,
)


class ShardStreamer:
    """Streams one verified 5k-sample .npz shard at a time so RSS stays < 800 MB."""

    def __init__(self, shard_dir: str, seed: int = 42):
        ok, detail = verify_hard_shards(shard_dir)
        if not ok:
            raise RuntimeError(f"Shard verification failed: {detail}")
        with open(os.path.join(shard_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.shard_dir = shard_dir
        self.files = [s["file"] for s in self.manifest["shards"]]
        self.rng = random.Random(seed)
        self._cur_idx = -1
        self._data: dict[str, np.ndarray] | None = None
        self._order: list[int] = []
        self._pos = 0

    def _load_next_shard(self) -> None:
        self._data = None
        gc.collect()
        self._cur_idx = (self._cur_idx + 1) % len(self.files)
        if self._cur_idx == 0:
            self.rng.shuffle(self.files)
        path = os.path.join(self.shard_dir, self.files[self._cur_idx])
        with np.load(path) as z:
            self._data = {
                "images": z["images"],
                "tasks": z["tasks"],
                "heat": z["heat"],
                "offsets": z["offsets"],
                "actions": z["actions"],
            }
        n = self._data["images"].shape[0]
        self._order = list(range(n))
        self.rng.shuffle(self._order)
        self._pos = 0

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._data is None or self._pos + batch_size > len(self._order):
            self._load_next_shard()
        assert self._data is not None
        idxs = self._order[self._pos : self._pos + batch_size]
        self._pos += batch_size

        imgs = self._data["images"][idxs].astype(np.float32) / 127.5 - 1.0
        xs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        tasks = torch.from_numpy(self._data["tasks"][idxs]).long()
        heat = torch.from_numpy(self._data["heat"][idxs].astype(np.float32))
        offs = torch.from_numpy(self._data["offsets"][idxs]).float()
        acts = torch.from_numpy(self._data["actions"][idxs]).long()
        return xs, tasks, heat, offs, acts


def apply_palette_aug(xs: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Fast zero-geometry photometric/palette jitter so OOD themes transfer cleanly."""
    out = xs.clone()
    for i in range(out.shape[0]):
        r = rng.random()
        if r < 0.22:
            # Invert polarity (light <-> dark)
            out[i] = -out[i]
        elif r < 0.44:
            # Channel permutation (simulates arbitrary accent/chrome palettes)
            perm = [0, 1, 2]
            rng.shuffle(perm)
            out[i] = out[i, perm]
        if rng.random() < 0.35:
            scale = rng.uniform(0.78, 1.22)
            shift = rng.uniform(-0.18, 0.18)
            out[i] = (out[i] * scale + shift).clamp(-1.0, 1.0)
    return out


def evaluate_grounder_on_split(
    grounder: EyesGrounder,
    split: Sequence[Any],
    use_stage2: bool = True,
    multi_scale: bool = False,
    threads: int = 2,
    bench_frames: int = 50,
    pre_rendered: Sequence[np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.set_num_threads(threads)
    preds: list[tuple[float, float, int]] = []
    latencies: list[float] = []

    # Warmup
    w_img = pre_rendered[0] if pre_rendered is not None else render_hard(split[0].layout, size=HARD_CANVAS)
    for _ in range(3):
        grounder.ground(w_img, split[0].task, use_stage2=use_stage2, multi_scale=multi_scale)

    for i, ex in enumerate(split):
        rgb = pre_rendered[i] if pre_rendered is not None else render_hard(ex.layout, size=HARD_CANVAS)
        t0 = time.perf_counter()
        res = grounder.ground(rgb, ex.task, use_stage2=use_stage2, multi_scale=multi_scale)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i < bench_frames:
            latencies.append(dt_ms)
        preds.append((float(res["fx"]), float(res["fy"]), int(res["action_id"])))

    rep, misses = diagnose_predictions(
        split,
        preds,
        canvas_w=HARD_CANVAS,
        canvas_h=HARD_CANVAS,
        render_fn=lambda e: render_hard(e.layout, size=HARD_CANVAS),
    )
    rep["latency_ms_per_frame"] = round(float(statistics.median(latencies)), 2)
    rep["latency_mean_ms"] = round(float(statistics.mean(latencies)), 2)
    rep["peak_rss_mb"] = peak_rss_mb()
    p1 = sum(p.numel() for p in grounder.stage1.parameters())
    p2 = sum(p.numel() for p in grounder.stage2.parameters()) if grounder.stage2 is not None and use_stage2 else 0
    rep["params_stage1"] = p1
    rep["params_stage2"] = p2
    rep["params_active"] = p1 + p2
    rep["use_stage2"] = bool(use_stage2 and grounder.stage2 is not None)
    rep["multi_scale"] = bool(multi_scale)
    return rep, misses


def train_baseline_on_shards(
    shard_dir: str,
    steps: int = 1500,
    batch_size: int = 32,
    lr: float = 2.5e-3,
    seed: int = 42,
    threads: int = 2,
    out_ckpt: str = "checkpoints/m2_baseline_hard.pt",
) -> GripNet:
    """Train unmodified 0.44M GripNet on 50k Hard World shards to set honest M2 baseline."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = GripNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    streamer = ShardStreamer(shard_dir, seed=seed)

    t0 = time.time()
    for step in range(1, steps + 1):
        xs, tasks, heat, offs, acts = streamer.next_batch(batch_size)
        scores, offsets, act_logits = model(xs, tasks)
        loss, parts = model.loss(scores, offsets, act_logits, heat.flatten(1), offs, acts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if step % 300 == 0 or step == steps:
            print(f"[baseline-hard] step {step}/{steps} loss={loss.item():.4f} ce={parts['ce']:.3f} elapsed={time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "step": steps}, out_ckpt)
    return model


def train_stage1_shards(
    shard_dir: str,
    steps: int = 2500,
    batch_size: int = 32,
    lr: float = 2.5e-3,
    seed: int = 42,
    threads: int = 2,
    color_aug: bool = True,
    init_ckpt: str | None = None,
    out_ckpt: str = "checkpoints/eyes_stage1.pt",
) -> Stage1CoarseNet:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Stage1CoarseNet()
    if init_ckpt and os.path.exists(init_ckpt):
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["stage1"] if "stage1" in ck else ck["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    streamer = ShardStreamer(shard_dir, seed=seed)
    t0 = time.time()

    for step in range(1, steps + 1):
        xs, tasks, heat, offs, acts = streamer.next_batch(batch_size)
        if color_aug:
            xs = apply_palette_aug(xs, rng)
        scores, offsets, box_wh, act_logits, _ = model(xs, tasks)
        loss, parts = model.loss(scores, offsets, box_wh, act_logits, heat, offs, acts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if step % 250 == 0 or step == steps:
            print(
                f"[stage1-shards] step {step}/{steps} loss={loss.item():.4f} "
                f"ce={parts['ce']:.3f} off={parts['off']:.3f} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"stage1": model.state_dict(), "step": steps}, out_ckpt)
    return model


def train_curriculum_hardneg_stage1(
    model: Stage1CoarseNet,
    steps: int = 1800,
    screens_per_step: int = 8,
    queries_per_screen: int = 4,
    lr: float = 1.5e-3,
    seed: int = 108,
    threads: int = 2,
    use_curriculum: bool = True,
    use_hard_neg: bool = True,
    out_ckpt: str = "checkpoints/eyes_stage1_curr.pt",
) -> Stage1CoarseNet:
    """Multi-query contrastive + curriculum (`easy -> medium -> crowded`) + online hard-miss replay."""
    torch.set_num_threads(threads)
    rng = random.Random(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    hard_buffer: list[Any] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        prog = step / float(steps)
        if use_curriculum:
            if prog < 0.25:
                tier = "easy" if rng.random() < 0.75 else "medium"
            elif prog < 0.60:
                tier = "medium" if rng.random() < 0.65 else ("crowded" if rng.random() < 0.7 else "easy")
            else:
                tier = "crowded" if rng.random() < 0.65 else "medium"
        else:
            tier = "mixed"

        screen_imgs = []
        repeat_counts = []
        tasks_list = []
        hm_list = []
        off_list = []
        wh_list = []
        act_list = []
        meta_els = []

        for s_i in range(screens_per_step):
            if use_hard_neg and hard_buffer and s_i < 2 and rng.random() < 0.6:
                layout = hard_buffer[rng.randrange(len(hard_buffer))]
            else:
                layout = make_hard_layout(rng, tier=tier, split="seen", label_mode="text")

            img128 = render_hard(layout, size=COARSE_CANVAS).astype(np.float32) / 127.5 - 1.0
            enabled = [e for e in layout.elements if e.state != "disabled"] or layout.elements

            if use_hard_neg and len(enabled) >= 2:
                # Prefer roles that have multiple distractors on this screen!
                role_counts: dict[str, int] = {}
                for e in enabled:
                    role_counts[e.role] = role_counts.get(e.role, 0) + 1
                ranked = sorted(enabled, key=lambda e: (-role_counts[e.role], rng.random()))
                chosen = ranked[: min(queries_per_screen, len(ranked))]
            else:
                chosen = [rng.choice(enabled)]

            screen_imgs.append(img128)
            repeat_counts.append(len(chosen))

            for tgt in chosen:
                t_str = rng.choice(TASKS_HARD[tgt.role]).format(label=tgt.label)
                tasks_list.append(encode_task(t_str))
                hm_list.append(target_heatmap_fast(tgt.cx, tgt.cy, HARD_CANVAS, 32))
                _, (dx, dy) = target_offset_fast(tgt.cx, tgt.cy, HARD_CANVAS, 32)
                off_list.append((dx, dy))
                wh_list.append((tgt.w / float(HARD_CANVAS), tgt.h / float(HARD_CANVAS)))
                act_list.append(ACTION_CLASSES.index(ACTION_OF_ROLE[tgt.role]))
                meta_els.append((layout, tgt))

        xs_unique = torch.from_numpy(np.stack(screen_imgs, axis=0)).permute(0, 3, 1, 2).contiguous()
        xs_unique = apply_palette_aug(xs_unique, rng)
        ts = torch.tensor(tasks_list, dtype=torch.long)
        hms = torch.from_numpy(np.stack(hm_list, axis=0)).float()
        offs = torch.tensor(off_list, dtype=torch.float32)
        whs = torch.tensor(wh_list, dtype=torch.float32)
        acts = torch.tensor(act_list, dtype=torch.long)

        fpn_unique = model.encode_vision(xs_unique)
        fpn_exp = torch.repeat_interleave(fpn_unique, torch.tensor(repeat_counts, dtype=torch.long), dim=0)
        q = model.encode_instruction(ts)
        scores, pred_offs, pred_wh, act_logits = model.match_features(fpn_exp, q)
        n_pairs = ts.shape[0]

        # Online hard-example weighting + hard-screen replay buffer
        if use_hard_neg:
            with torch.no_grad():
                xy = model.click_xy(scores, pred_offs, HARD_CANVAS)
                weights = torch.ones(n_pairs, dtype=torch.float32)
                for i, (lay, tgt) in enumerate(meta_els):
                    px, py = float(xy[i, 0].item()), float(xy[i, 1].item())
                    if not tgt.contains(px, py):
                        weights[i] = 1.85
                        if len(hard_buffer) < 192:
                            hard_buffer.append(lay)
                        elif rng.random() < 0.25:
                            hard_buffer[rng.randrange(len(hard_buffer))] = lay
                weights = weights / weights.mean()

            hm_flat = hms.flatten(1)
            hm_flat = hm_flat / hm_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
            logp = F.log_softmax(scores, dim=1)
            cell = hm_flat.argmax(dim=1).long()
            per_ce = 0.5 * (-(hm_flat * logp).sum(dim=1)) + 0.5 * F.cross_entropy(scores, cell, reduction="none")
            ce_loss = (per_ce * weights).mean()
            idx = torch.arange(n_pairs)
            off_loss = F.smooth_l1_loss(pred_offs[idx, cell], offs, beta=0.15)
            wh_loss = F.smooth_l1_loss(pred_wh[idx, cell], whs, beta=0.05)
            act_loss = F.cross_entropy(act_logits, acts)

            # Symmetric in-screen CLIP contrastive loss across co-occurring widgets on each screen
            contrast_loss = torch.tensor(0.0)
            n_grp = 0
            cur = 0
            for kc in repeat_counts:
                if kc >= 2:
                    sub_scores = scores[cur : cur + kc]          # (kc, 1024)
                    sub_cells = cell[cur : cur + kc]             # (kc,)
                    # Only apply column/row contrast if target cells are distinct
                    if len(torch.unique(sub_cells)) == kc:
                        mat = sub_scores[:, sub_cells]           # (kc, kc)
                        tgt_diag = torch.arange(kc)
                        contrast_loss = contrast_loss + 0.5 * (
                            F.cross_entropy(mat, tgt_diag) + F.cross_entropy(mat.t(), tgt_diag)
                        )
                        n_grp += 1
                cur += kc
            if n_grp > 0:
                contrast_loss = contrast_loss / n_grp

            loss = ce_loss + 0.6 * contrast_loss + 0.8 * off_loss + 0.4 * wh_loss + 0.2 * act_loss
        else:
            loss, _ = model.loss(scores, pred_offs, pred_wh, act_logits, hms, offs, acts, target_wh=whs)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if step % 200 == 0 or step == steps:
            print(
                f"[stage1-curr-hn] step {step}/{steps} tier={tier} loss={loss.item():.4f} "
                f"buf={len(hard_buffer)} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"stage1": model.state_dict(), "step": steps}, out_ckpt)
    return model


def _crop_heatmap_16(ux: float, uy: float, grid: int = 16) -> np.ndarray:
    """ux, uy in [-1, 1] -> 16x16 soft heatmap."""
    gx = (ux * 0.5 + 0.5) * grid - 0.5
    gy = (uy * 0.5 + 0.5) * grid - 0.5
    ys, xs = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="ij")
    d2 = (ys - gy) ** 2 + (xs - gx) ** 2
    hm = np.exp(-d2 / (2.0 * 0.85 * 0.85))
    return hm / hm.sum()


def train_stage2_refiner(
    stage1: Stage1CoarseNet,
    steps: int = 2000,
    screens_per_step: int = 6,
    lr: float = 2.2e-3,
    seed: int = 303,
    threads: int = 2,
    hard_neg_mining: bool = True,
    finetune_text: bool = False,
    init_stage2: Stage2RefineNet | None = None,
    out_ckpt: str = "checkpoints/eyes_two_stage.pt",
) -> Stage2RefineNet:
    """Train Stage2RefineNet on native 256x256 crops (jittered positives + distractor/Stage1-miss negatives)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    stage1.eval()
    stage2 = init_stage2 if init_stage2 is not None else Stage2RefineNet(d_text=stage1.d_text)
    stage2.train()

    params = list(stage2.parameters())
    if finetune_text:
        stage1.text.train()
        params += list(stage1.text.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

    span = Stage2RefineNet.SPAN_FRAC * HARD_CANVAS
    t0 = time.time()

    for step in range(1, steps + 1):
        prog = step / float(steps)
        tier = "mixed" if prog < 0.4 else ("crowded" if rng.random() < 0.65 else "medium")

        crops_list = []
        tasks_list = []
        is_pos_list = []
        ux_uy_list = []
        wh_list = []
        hm16_list = []

        for _ in range(screens_per_step):
            ex = sample_hard_example(rng, tier=tier, split="seen", label_mode="text")
            rgb = render_hard(ex.layout, size=HARD_CANVAS)
            tgt = ex.target
            t_ids = encode_task(ex.task)

            # 1) Positive crop jittered around target center so Stage 2 corrects up to +-26px coarse error!
            jx = tgt.cx + rng.uniform(-22.0, 22.0)
            jy = tgt.cy + rng.uniform(-22.0, 22.0)
            pos_arr, (rx, ry, rw, rh) = extract_crop_tensor(rgb, jx, jy, span, span)
            ux = float(np.clip(((tgt.cx - rx) / rw - 0.5) * 2.0, -0.98, 0.98))
            uy = float(np.clip(((tgt.cy - ry) / rh - 0.5) * 2.0, -0.98, 0.98))
            crops_list.append(pos_arr)
            tasks_list.append(t_ids)
            is_pos_list.append(1.0)
            ux_uy_list.append((ux, uy))
            wh_list.append((min(0.98, tgt.w / rw), min(0.98, tgt.h / rh)))
            hm16_list.append(_crop_heatmap_16(ux, uy, 16))

            # 2) Distractor widget crops on the SAME screen (prioritizing same-role wrong-label!)
            others = [
                e for e in ex.layout.elements
                if e is not tgt and math.hypot(e.cx - tgt.cx, e.cy - tgt.cy) > 14.0
            ]
            if hard_neg_mining:
                others.sort(key=lambda e: (0 if e.role == tgt.role else 1, rng.random()))
            else:
                rng.shuffle(others)

            for neg_el in others[:2]:
                ncx = neg_el.cx + rng.uniform(-8.0, 8.0)
                ncy = neg_el.cy + rng.uniform(-8.0, 8.0)
                neg_arr, (nrx, nry, nrw, nrh) = extract_crop_tensor(rgb, ncx, ncy, span, span)
                # Check whether target center accidentally also fell inside this distractor crop!
                tgt_inside = (nrx + 6.0 <= tgt.cx <= nrx + nrw - 6.0) and (nry + 6.0 <= tgt.cy <= nry + nrh - 6.0)
                if tgt_inside:
                    tux = float(np.clip(((tgt.cx - nrx) / nrw - 0.5) * 2.0, -0.98, 0.98))
                    tuy = float(np.clip(((tgt.cy - nry) / nrh - 0.5) * 2.0, -0.98, 0.98))
                    crops_list.append(neg_arr)
                    tasks_list.append(t_ids)
                    is_pos_list.append(1.0)
                    ux_uy_list.append((tux, tuy))
                    wh_list.append((min(0.98, tgt.w / nrw), min(0.98, tgt.h / nrh)))
                    hm16_list.append(_crop_heatmap_16(tux, tuy, 16))
                else:
                    crops_list.append(neg_arr)
                    tasks_list.append(t_ids)
                    is_pos_list.append(0.0)
                    ux_uy_list.append((0.0, 0.0))
                    wh_list.append((0.2, 0.2))
                    hm16_list.append(np.full((16, 16), 1.0 / 256.0, dtype=np.float32))

            # 3) Confusable-query hard negative on the positive crop (zero extra crop extraction!)
            if hard_neg_mining and others:
                wrong_lbl = others[0].label
                wrong_task = encode_task(rng.choice(TASKS_HARD[tgt.role]).format(label=wrong_lbl))
                # Make sure wrong_lbl widget isn't also inside pos_arr
                w_el = others[0]
                w_inside = (rx + 6.0 <= w_el.cx <= rx + rw - 6.0) and (ry + 6.0 <= w_el.cy <= ry + rh - 6.0)
                if not w_inside:
                    crops_list.append(pos_arr)
                    tasks_list.append(wrong_task)
                    is_pos_list.append(0.0)
                    ux_uy_list.append((0.0, 0.0))
                    wh_list.append((0.2, 0.2))
                    hm16_list.append(np.full((16, 16), 1.0 / 256.0, dtype=np.float32))

        crop_t = torch.from_numpy(np.stack(crops_list, axis=0)).permute(0, 3, 1, 2).contiguous()
        crop_t = apply_palette_aug(crop_t, rng)
        task_t = torch.tensor(tasks_list, dtype=torch.long)
        pos_t = torch.tensor(is_pos_list, dtype=torch.float32)
        xy_t = torch.tensor(ux_uy_list, dtype=torch.float32)
        wh_t = torch.tensor(wh_list, dtype=torch.float32)
        hm_t = torch.from_numpy(np.stack(hm16_list, axis=0)).flatten(1).float()

        if finetune_text:
            q = stage1.encode_instruction(task_t)
        else:
            with torch.no_grad():
                q = stage1.encode_instruction(task_t)

        v_logit, f_xy, f_wh, heat_logits = stage2.forward_with_map(crop_t, q)
        verify_loss = F.binary_cross_entropy_with_logits(v_logit, pos_t)

        pos_mask = pos_t > 0.5
        if pos_mask.any():
            logp16 = F.log_softmax(heat_logits[pos_mask], dim=1)
            hm_pos = hm_t[pos_mask]
            ce16 = 0.5 * (-(hm_pos * logp16).sum(dim=1).mean()) + 0.5 * F.cross_entropy(
                heat_logits[pos_mask], hm_pos.argmax(dim=1).long()
            )
            xy_loss = F.smooth_l1_loss(f_xy[pos_mask], xy_t[pos_mask], beta=0.08)
            wh_loss = F.smooth_l1_loss(f_wh[pos_mask], wh_t[pos_mask], beta=0.08)
        else:
            ce16 = xy_loss = wh_loss = torch.tensor(0.0)

        loss = verify_loss + 0.7 * ce16 + 2.0 * xy_loss + 0.4 * wh_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()

        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if step % 250 == 0 or step == steps:
            with torch.no_grad():
                v_acc = ((v_logit > 0).float() == pos_t).float().mean().item()
            print(
                f"[stage2] step {step}/{steps} loss={loss.item():.4f} v_bce={verify_loss.item():.3f} "
                f"v_acc={v_acc:.3f} xy={xy_loss.item():.4f} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    stage1.eval()
    stage2.eval()
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({
        "stage1": stage1.state_dict(),
        "stage2": stage2.state_dict(),
        "step_stage2": steps,
    }, out_ckpt)
    return stage2
