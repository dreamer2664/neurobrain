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


def build_screen_bank(n_per_tier: int = 700, seed: int = 8080) -> dict[str, list[tuple[np.ndarray, list[Any]]]]:
    """Pre-render native 256x256 Hard World screens into a compact uint8 RAM bank (~412 MB for 2,100 screens)."""
    rng = random.Random(seed)
    bank: dict[str, list[tuple[np.ndarray, list[Any]]]] = {"easy": [], "medium": [], "crowded": []}
    t0 = time.time()
    for tier in ("easy", "medium", "crowded"):
        items = []
        while len(items) < n_per_tier:
            layout = make_hard_layout(rng, tier=tier, split="seen", label_mode="text")
            valid_els = [e for e in layout.elements if getattr(e, "state", "normal") != "disabled"]
            if len(valid_els) < 2:
                continue
            rgb = render_hard(layout, size=HARD_CANVAS)
            items.append((rgb, valid_els))
        bank[tier] = items
    n_tot = sum(len(v) for v in bank.values())
    n_els = sum(len(els) for v in bank.values() for _, els in v)
    print(
        f"[screen_bank] rendered {n_tot} native 256x256 screens ({n_els} active widgets) "
        f"in {time.time()-t0:.1f}s rss={peak_rss_mb():.0f}MB",
        flush=True,
    )
    return bank


def train_stage2_refiner(
    stage1: Stage1CoarseNet,
    steps: int = 2000,
    screens_per_step: int = 8,
    max_siblings: int = 6,
    lr: float = 2.4e-3,
    seed: int = 303,
    threads: int = 2,
    hard_neg_mining: bool = True,
    curriculum: bool = True,
    bank: dict[str, list[tuple[np.ndarray, list[Any]]]] | None = None,
    init_stage2: Stage2RefineNet | None = None,
    out_ckpt: str = "checkpoints/eyes_two_stage.pt",
) -> Stage2RefineNet:
    """Train Stage2RefineNet on native 32x192 strips with controlled curriculum + hard-negative switches."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if bank is None:
        bank = build_screen_bank(n_per_tier=700, seed=seed + 5000)

    stage1.eval()
    stage2 = init_stage2 if init_stage2 is not None else Stage2RefineNet()
    stage2.train()

    opt = torch.optim.AdamW(stage2.parameters(), lr=lr, weight_decay=1e-2)
    span_w = Stage2RefineNet.SPAN_W_FRAC * HARD_CANVAS
    span_h = Stage2RefineNet.SPAN_H_FRAC * HARD_CANVAS

    # Online label-confusion boost counts for hard-negative mining
    conf_counts: dict[tuple[str, str], int] = {}
    t0 = time.time()

    for step in range(1, steps + 1):
        prog = step / float(steps)
        if curriculum:
            if prog < 0.25:
                tier_probs = [("easy", 0.65), ("medium", 0.30), ("crowded", 0.05)]
            elif prog < 0.55:
                tier_probs = [("easy", 0.20), ("medium", 0.50), ("crowded", 0.30)]
            else:
                tier_probs = [("easy", 0.10), ("medium", 0.30), ("crowded", 0.60)]
        else:
            tier_probs = [("easy", 0.333), ("medium", 0.333), ("crowded", 0.334)]

        if hard_neg_mining:
            # Multi-sibling (M x (M+1)) in-screen contrastive + spatial-rejection negative + offset refinement
            all_crops: list[np.ndarray] = []
            all_extra_neg_crops: list[np.ndarray] = []
            all_tasks: list[list[int]] = []
            all_duv: list[tuple[float, float]] = []
            all_wh: list[tuple[float, float]] = []
            group_sizes: list[int] = []
            group_labels: list[list[str]] = []

            for _ in range(screens_per_step):
                r_tier = rng.random()
                cum = 0.0
                chosen_tier = "crowded"
                for tn, tp in tier_probs:
                    cum += tp
                    if r_tier <= cum:
                        chosen_tier = tn
                        break
                rgb, els = rng.choice(bank[chosen_tier])

                # Oversample dense / worst-confusion roles (tab, menu, button, icon) as anchor
                dense_els = [e for e in els if e.role in ("tab", "menu", "button", "icon")]
                anchor = rng.choice(dense_els) if (dense_els and rng.random() < 0.72) else rng.choice(els)

                # Pick distinct-label siblings on the SAME screen, prioritizing same-role & confused labels!
                used_labels = {anchor.label}
                chosen = [anchor]
                cands_other = [e for e in els if e.label not in used_labels]
                cands_other.sort(
                    key=lambda e: (
                        0 if e.role == anchor.role else 1,
                        -conf_counts.get((anchor.label, e.label), 0),
                        abs(e.cy - anchor.cy) * 0.5 + abs(e.cx - anchor.cx) * 0.15 + rng.uniform(0.0, 8.0),
                    )
                )
                for e in cands_other:
                    if e.label not in used_labels:
                        chosen.append(e)
                        used_labels.add(e.label)
                        if len(chosen) >= max_siblings:
                            break
                if len(chosen) < 2:
                    continue

                for el in chosen:
                    # Realistic Stage-1 candidate proposal jitter (matching measured dx/dy + 15% wider tail)
                    if rng.random() < 0.85:
                        jx = float(np.clip(el.cx + rng.gauss(0.0, 2.5), 2.0, HARD_CANVAS - 3.0))
                        jy = float(np.clip(el.cy + rng.gauss(0.0, 2.5), 2.0, HARD_CANVAS - 3.0))
                    else:
                        jx = float(np.clip(el.cx + rng.uniform(-8.5, 8.5), 2.0, HARD_CANVAS - 3.0))
                        jy = float(np.clip(el.cy + rng.uniform(-8.0, 8.0), 2.0, HARD_CANVAS - 3.0))
                    arr6, (_, _, rw, rh) = extract_crop_tensor(rgb, jx, jy, span_w, span_h)
                    max_dx = Stage2RefineNet.MAX_DX_PX * (rw / float(Stage2RefineNet.CROP_W))
                    max_dy = Stage2RefineNet.MAX_DY_PX * (rh / float(Stage2RefineNet.CROP_H))
                    du = float(np.clip((el.cx - jx) / max_dx, -0.99, 0.99))
                    dv = float(np.clip((el.cy - jy) / max_dy, -0.99, 0.99))
                    t_str = rng.choice(TASKS_HARD[el.role]).format(label=el.label)

                    all_crops.append(arr6)
                    all_tasks.append(encode_task(t_str))
                    all_duv.append((du, dv))
                    all_wh.append((min(0.96, el.w / float(HARD_CANVAS)), min(0.25, el.h / float(HARD_CANVAS))))

                # Add 1 off-center / background spatial-rejection negative per screen
                for _ in range(8):
                    if rng.random() < 0.55:
                        sgn_x = 1.0 if rng.random() < 0.5 else -1.0
                        sgn_y = 1.0 if rng.random() < 0.5 else -1.0
                        nx = float(np.clip(anchor.cx + sgn_x * rng.uniform(0.55 * anchor.w + 8.0, 0.55 * anchor.w + 26.0), 4.0, HARD_CANVAS - 5.0))
                        ny = float(np.clip(anchor.cy + sgn_y * rng.uniform(0.55 * anchor.h + 6.0, 0.55 * anchor.h + 20.0), 4.0, HARD_CANVAS - 5.0))
                    else:
                        nx, ny = rng.uniform(12.0, 244.0), rng.uniform(12.0, 244.0)
                    if all(not (e.x - 3 <= nx <= e.x + e.w + 3 and e.y - 3 <= ny <= e.y + e.h + 3) for e in chosen):
                        break
                neg_arr6, _ = extract_crop_tensor(rgb, nx, ny, span_w, span_h)
                all_extra_neg_crops.append(neg_arr6)

                group_sizes.append(len(chosen))
                group_labels.append([e.label for e in chosen])

            pos_crop_t = torch.from_numpy(np.stack(all_crops, axis=0)).permute(0, 3, 1, 2).contiguous()
            neg_crop_t = torch.from_numpy(np.stack(all_extra_neg_crops, axis=0)).permute(0, 3, 1, 2).contiguous()
            joint_crops = torch.cat([pos_crop_t, neg_crop_t], dim=0)
            joint_crops[:, :3] = apply_palette_aug(joint_crops[:, :3], rng)

            n_pos = pos_crop_t.shape[0]
            task_t = torch.tensor(all_tasks, dtype=torch.long)
            duv_t = torch.tensor(all_duv, dtype=torch.float32)
            wh_t = torch.tensor(all_wh, dtype=torch.float32)

            q_all = stage2.encode_instruction(task_t)
            joint_repr = stage2.encode_crops(joint_crops)
            pos_repr = {k: v[:n_pos] for k, v in joint_repr.items()}
            neg_repr = {k: v[n_pos:] for k, v in joint_repr.items()}

            # Diagonal offset/wh loss on the positive widget strips
            q_heads = F.normalize(stage2.q_proj(q_all).view(n_pos, stage2.n_heads, stage2.head_dim), dim=2, eps=1e-6)
            sim_h = torch.einsum("khd,khdp->khp", q_heads, pos_repr["k_map"])
            attn = F.softmax(sim_h.mean(dim=1) * stage2.attn_temp.clamp(1.0, 12.0) + stage2.col_attn_bias, dim=-1)
            v_attn = torch.einsum("kp,kdp->kd", attn, pos_repr["v_map"])
            sub = stage2.sub_off(torch.cat([pos_repr["anc_vec"], v_attn, q_all], dim=1))
            pred_duv = torch.tanh(sub[:, :2])
            pred_wh = torch.sigmoid(sub[:, 2:4])
            xy_loss = F.smooth_l1_loss(pred_duv, duv_t, beta=0.08)
            wh_loss = F.smooth_l1_loss(pred_wh, wh_t, beta=0.05)

            ce_loss = torch.tensor(0.0)
            bce_loss = torch.tensor(0.0)
            diag_hits = 0
            diag_tot = 0
            cur = 0
            for g_idx, (gs, glbls) in enumerate(zip(group_sizes, group_labels)):
                sub_repr = {
                    k: torch.cat([pos_repr[k][cur : cur + gs], neg_repr[k][g_idx : g_idx + 1]], dim=0)
                    for k in pos_repr
                }
                sub_q = q_all[cur : cur + gs]
                mat = stage2.score_matrix(sub_repr, sub_q)  # (gs, gs + 1)
                eye_idx = torch.arange(gs)
                tgt_bce = torch.zeros((gs, gs + 1), dtype=torch.float32)
                tgt_bce[eye_idx, eye_idx] = 1.0
                ce_loss = ce_loss + 0.55 * F.cross_entropy(mat, eye_idx) + 0.45 * F.cross_entropy(mat[:, :gs].t(), eye_idx)
                bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                    mat, tgt_bce, pos_weight=torch.tensor(float(gs))
                )
                with torch.no_grad():
                    preds = mat.argmax(dim=1).tolist()
                    for qi, pi in enumerate(preds):
                        diag_tot += 1
                        if pi == qi:
                            diag_hits += 1
                        elif pi < gs:
                            pair = (glbls[qi], glbls[pi])
                            conf_counts[pair] = min(50, conf_counts.get(pair, 0) + 1)
                cur += gs

            n_g = max(1, len(group_sizes))
            ce_loss = ce_loss / n_g
            bce_loss = bce_loss / n_g
            loss = ce_loss + 0.75 * bce_loss + 1.5 * xy_loss + 0.25 * wh_loss
            acc_metric = diag_hits / max(1, diag_tot)
        else:
            # Controlled baseline without sibling hard-negative mining: 1 positive + 1 random negative per screen
            crops_list = []
            tasks_list = []
            is_pos_list = []
            duv_list = []
            wh_list = []
            for _ in range(screens_per_step):
                r_tier = rng.random()
                cum = 0.0
                chosen_tier = "crowded"
                for tn, tp in tier_probs:
                    cum += tp
                    if r_tier <= cum:
                        chosen_tier = tn
                        break
                rgb, els = rng.choice(bank[chosen_tier])
                tgt = rng.choice(els)
                t_ids = encode_task(rng.choice(TASKS_HARD[tgt.role]).format(label=tgt.label))

                jx = float(np.clip(tgt.cx + rng.gauss(0.0, 2.5), 2.0, HARD_CANVAS - 3.0))
                jy = float(np.clip(tgt.cy + rng.gauss(0.0, 2.5), 2.0, HARD_CANVAS - 3.0))
                pos_arr, (_, _, rw, rh) = extract_crop_tensor(rgb, jx, jy, span_w, span_h)
                max_dx = Stage2RefineNet.MAX_DX_PX * (rw / float(Stage2RefineNet.CROP_W))
                max_dy = Stage2RefineNet.MAX_DY_PX * (rh / float(Stage2RefineNet.CROP_H))
                du = float(np.clip((tgt.cx - jx) / max_dx, -0.99, 0.99))
                dv = float(np.clip((tgt.cy - jy) / max_dy, -0.99, 0.99))
                crops_list.append(pos_arr)
                tasks_list.append(t_ids)
                is_pos_list.append(1.0)
                duv_list.append((du, dv))
                wh_list.append((min(0.96, tgt.w / float(HARD_CANVAS)), min(0.25, tgt.h / float(HARD_CANVAS))))

                others = [e for e in els if e.label != tgt.label]
                if others and rng.random() < 0.5:
                    nel = rng.choice(others)
                    nx, ny = nel.cx + rng.gauss(0.0, 2.5), nel.cy + rng.gauss(0.0, 2.5)
                else:
                    nx, ny = rng.uniform(20.0, 236.0), rng.uniform(20.0, 236.0)
                    if math.hypot(nx - tgt.cx, ny - tgt.cy) < 24.0:
                        nx = (nx + 96.0) % 240.0 + 8.0
                neg_arr, _ = extract_crop_tensor(rgb, nx, ny, span_w, span_h)
                crops_list.append(neg_arr)
                tasks_list.append(t_ids)
                is_pos_list.append(0.0)
                duv_list.append((0.0, 0.0))
                wh_list.append((0.2, 0.08))

            crop_t = torch.from_numpy(np.stack(crops_list, axis=0)).permute(0, 3, 1, 2).contiguous()
            crop_t[:, :3] = apply_palette_aug(crop_t[:, :3], rng)
            task_t = torch.tensor(tasks_list, dtype=torch.long)
            pos_t = torch.tensor(is_pos_list, dtype=torch.float32)
            duv_t = torch.tensor(duv_list, dtype=torch.float32)
            wh_t = torch.tensor(wh_list, dtype=torch.float32)

            q = stage2.encode_instruction(task_t)
            v_logit, pred_duv, pred_wh = stage2(crop_t, q)
            bce_loss = F.binary_cross_entropy_with_logits(v_logit, pos_t)
            pos_mask = pos_t > 0.5
            xy_loss = F.smooth_l1_loss(pred_duv[pos_mask], duv_t[pos_mask], beta=0.08)
            wh_loss = F.smooth_l1_loss(pred_wh[pos_mask], wh_t[pos_mask], beta=0.05)
            loss = bce_loss + 1.5 * xy_loss + 0.25 * wh_loss
            with torch.no_grad():
                acc_metric = float(((v_logit > 0).float() == pos_t).float().mean().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage2.parameters(), 2.0)
        opt.step()

        cur_lr = lr * (0.06 + 0.94 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if step % 200 == 0 or step == steps:
            print(
                f"[stage2 hn={int(hard_neg_mining)} curr={int(curriculum)}] "
                f"step {step}/{steps} loss={loss.item():.4f} acc={acc_metric:.3f} "
                f"xy={xy_loss.item():.4f} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
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


def build_stage1_episode_cache(
    stage1: Stage1CoarseNet,
    bank: dict[str, list[tuple[np.ndarray, list[Any]]]],
    episodes_per_screen: int = 3,
    top_k: int = 7,
    seed: int = 707,
) -> list[dict[str, Any]]:
    """Precompute real Stage-1 Top-K candidates (`< 3 MB` RAM) across all screens in `bank`."""
    from PIL import Image

    rng = random.Random(seed)
    stage1.eval()
    episodes: list[dict[str, Any]] = []
    t0 = time.time()

    flat_screens: list[tuple[str, int, np.ndarray, list[Any]]] = []
    for tier, items in bank.items():
        for idx, (rgb, els) in enumerate(items):
            flat_screens.append((tier, idx, rgb, els))

    with torch.no_grad():
        for b0 in range(0, len(flat_screens), 64):
            chunk = flat_screens[b0 : b0 + 64]
            x128_list = []
            for _, _, rgb, _ in chunk:
                im128 = np.asarray(Image.fromarray(rgb).resize((COARSE_CANVAS, COARSE_CANVAS), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
                x128_list.append(im128)
            x128 = torch.from_numpy(np.stack(x128_list, 0)).permute(0, 3, 1, 2).contiguous()
            fpn = stage1.encode_vision(x128)

            q_tasks = []
            q_meta = []
            rep_counts = []
            for _, (tier, s_idx, _, els) in enumerate(chunk):
                ordered = list(els)
                ordered.sort(key=lambda e: (0 if e.role in ("icon", "button", "menu", "tab") else 1, rng.random()))
                chosen_els = ordered[: min(len(ordered), episodes_per_screen)]
                rep_counts.append(len(chosen_els))
                for el in chosen_els:
                    t_ids = encode_task(rng.choice(TASKS_HARD[el.role]).format(label=el.label))
                    q_tasks.append(t_ids)
                    q_meta.append((tier, s_idx, el))

            if not q_tasks:
                continue
            rc_t = torch.tensor(rep_counts, dtype=torch.long)
            fpn_r = torch.repeat_interleave(fpn, rc_t, dim=0)
            q1 = stage1.encode_instruction(torch.tensor(q_tasks, dtype=torch.long))
            sc, off, wh, _ = stage1.match_features(fpn_r, q1)

            for qi, ((tier, s_idx, el), t_ids) in enumerate(zip(q_meta, q_tasks)):
                cands = topk_grid_candidates(
                    sc[qi], off[qi], wh[qi], HARD_CANVAS, HARD_CANVAS, k=top_k, nms_radius_cells=1
                )
                # Keep episodes where at least one Stage-1 candidate hits or is within 10px of target
                has_hit = any(
                    (el.x - 2 <= c["x"] <= el.x + el.w + 2 and el.y - 2 <= c["y"] <= el.y + el.h + 2)
                    or (abs(c["x"] - el.cx) <= 10.0 and abs(c["y"] - el.cy) <= 9.0)
                    for c in cands
                )
                if has_hit and len(cands) == top_k:
                    episodes.append({
                        "tier": tier,
                        "s_idx": s_idx,
                        "task": t_ids,
                        "tgt": (float(el.x), float(el.y), float(el.w), float(el.h), float(el.cx), float(el.cy)),
                        "cands": [(float(c["x"]), float(c["y"]), float(c["stage1_logp"])) for c in cands],
                    })

    print(
        f"[episode_cache] built {len(episodes)} exact Stage-1 Top-{top_k} episodes "
        f"in {time.time()-t0:.1f}s rss={peak_rss_mb():.0f}MB",
        flush=True,
    )
    return episodes


def train_stage2_listwise_reranker(
    stage1: Stage1CoarseNet,
    stage2: Stage2RefineNet,
    bank: dict[str, list[tuple[np.ndarray, list[Any]]]],
    episodes: list[dict[str, Any]],
    steps: int = 900,
    batch_episodes: int = 12,
    lr: float = 9.0e-4,
    seed: int = 808,
    threads: int = 2,
    out_ckpt: str = "checkpoints/eyes_m6_best.pt",
) -> Stage2RefineNet:
    """End-to-end listwise fine-tuning of Stage2 + `stage1_alpha` directly on real Stage-1 Top-7 candidates."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    stage1.eval()
    stage2.train()
    opt = torch.optim.AdamW(stage2.parameters(), lr=lr, weight_decay=1e-2)

    span_w = Stage2RefineNet.SPAN_W_FRAC * HARD_CANVAS
    span_h = Stage2RefineNet.SPAN_H_FRAC * HARD_CANVAS
    hard_replay: list[dict[str, Any]] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        batch: list[dict[str, Any]] = []
        n_hard = min(len(hard_replay), batch_episodes // 3)
        for _ in range(n_hard):
            batch.append(rng.choice(hard_replay))
        while len(batch) < batch_episodes:
            batch.append(rng.choice(episodes))

        B = len(batch)
        K = len(batch[0]["cands"])
        crops_list = []
        tasks_list = []
        s1_logp_mat = np.zeros((B, K), dtype=np.float32)
        soft_tgt_mat = np.zeros((B, K), dtype=np.float32)
        valid_pos_mask = []
        duv_targets = []
        wh_targets = []

        for bi, ep in enumerate(batch):
            rgb, _ = bank[ep["tier"]][ep["s_idx"]]
            tx, ty, tw, th, tcx, tcy = ep["tgt"]
            tasks_list.append(ep["task"])
            weights_k = np.zeros(K, dtype=np.float32)

            for ki, (cx, cy, lp) in enumerate(ep["cands"]):
                # Small sub-pixel training jitter (0.6px) for generalization
                jcx = float(np.clip(cx + rng.gauss(0.0, 0.5), 1.0, HARD_CANVAS - 2.0))
                jcy = float(np.clip(cy + rng.gauss(0.0, 0.5), 1.0, HARD_CANVAS - 2.0))
                arr6, (_, _, rw, rh) = extract_crop_tensor(rgb, jcx, jcy, span_w, span_h)
                crops_list.append(arr6)
                s1_logp_mat[bi, ki] = lp

                inside = (tx - 1.5 <= cx <= tx + tw + 1.5) and (ty - 1.5 <= cy <= ty + th + 1.5)
                dist = math.hypot(cx - tcx, cy - tcy)
                if inside or dist <= 9.0:
                    weights_k[ki] = math.exp(-dist / 6.0)
                    valid_pos_mask.append(True)
                    max_dx = Stage2RefineNet.MAX_DX_PX * (rw / float(Stage2RefineNet.CROP_W))
                    max_dy = Stage2RefineNet.MAX_DY_PX * (rh / float(Stage2RefineNet.CROP_H))
                    du = float(np.clip((tcx - jcx) / max_dx, -0.99, 0.99))
                    dv = float(np.clip((tcy - jcy) / max_dy, -0.99, 0.99))
                    duv_targets.append((du, dv))
                    wh_targets.append((min(0.96, tw / float(HARD_CANVAS)), min(0.25, th / float(HARD_CANVAS))))
                else:
                    valid_pos_mask.append(False)
                    duv_targets.append((0.0, 0.0))
                    wh_targets.append((0.2, 0.08))

            if weights_k.sum() <= 0:
                # Fallback to nearest candidate
                dists = [math.hypot(cx - tcx, cy - tcy) for cx, cy, _ in ep["cands"]]
                weights_k[int(np.argmin(dists))] = 1.0
            soft_tgt_mat[bi] = weights_k / weights_k.sum()

        crop_t = torch.from_numpy(np.stack(crops_list, 0)).permute(0, 3, 1, 2).contiguous()
        # Apply palette aug per episode so all K candidates of the same screen share the exact same theme!
        crop_b = crop_t.view(B, K, 6, Stage2RefineNet.CROP_H, Stage2RefineNet.CROP_W)
        for bi in range(B):
            aug_rgb = apply_palette_aug(crop_b[bi : bi + 1, 0, :3], rng)
            # Re-apply identical transform by computing linear scale/perm cleanly on all K candidates:
            r = rng.random()
            if r < 0.20:
                crop_b[bi, :, :3] = -crop_b[bi, :, :3]
            elif r < 0.40:
                perm = [0, 1, 2]
                rng.shuffle(perm)
                crop_b[bi, :, :3] = crop_b[bi, :, perm]
            if rng.random() < 0.30:
                sc_f = rng.uniform(0.82, 1.18)
                sh_f = rng.uniform(-0.14, 0.14)
                crop_b[bi, :, :3] = (crop_b[bi, :, :3] * sc_f + sh_f).clamp(-1.0, 1.0)
        crop_t = crop_b.view(B * K, 6, Stage2RefineNet.CROP_H, Stage2RefineNet.CROP_W)

        task_t = torch.tensor(tasks_list, dtype=torch.long)
        q2 = stage2.encode_instruction(task_t)  # (B, D)
        q2_rep = q2[:, None, :].expand(B, K, -1).reshape(B * K, -1)

        v_logits, pred_duv, pred_wh = stage2(crop_t, q2_rep)
        v_mat = v_logits.view(B, K)
        s1_t = torch.from_numpy(s1_logp_mat)
        alpha = stage2.stage1_alpha.clamp(0.05, 1.50)
        comb_mat = alpha * s1_t + v_mat  # (B, K)

        soft_tgt = torch.from_numpy(soft_tgt_mat)
        log_probs = F.log_softmax(comb_mat, dim=1)
        list_loss = -(soft_tgt * log_probs).sum(dim=1).mean()
        bin_tgt = (soft_tgt > 0.05).float()
        bce_loss = F.binary_cross_entropy_with_logits(v_mat, bin_tgt, pos_weight=torch.tensor(4.0))

        pm = torch.tensor(valid_pos_mask, dtype=torch.bool)
        duv_t = torch.tensor(duv_targets, dtype=torch.float32)
        wh_t = torch.tensor(wh_targets, dtype=torch.float32)
        xy_loss = F.smooth_l1_loss(pred_duv[pm], duv_t[pm], beta=0.06) if pm.any() else torch.tensor(0.0)
        wh_loss = F.smooth_l1_loss(pred_wh[pm], wh_t[pm], beta=0.05) if pm.any() else torch.tensor(0.0)

        loss = list_loss + 0.4 * bce_loss + 1.8 * xy_loss + 0.2 * wh_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage2.parameters(), 2.0)
        opt.step()

        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        with torch.no_grad():
            picks = comb_mat.argmax(dim=1).tolist()
            hits = 0
            for bi, pk in enumerate(picks):
                if soft_tgt_mat[bi, pk] > 0.05:
                    hits += 1
                else:
                    if len(hard_replay) >= 600:
                        hard_replay.pop(0)
                    hard_replay.append(batch[bi])

        if step % 150 == 0 or step == steps:
            print(
                f"[stage2-listwise] step {step}/{steps} loss={loss.item():.4f} hit7={hits/B:.3f} "
                f"alpha={float(stage2.stage1_alpha.item()):.3f} xy={xy_loss.item():.4f} "
                f"rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    stage2.eval()
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({
        "stage1": stage1.state_dict(),
        "stage2": stage2.state_dict(),
        "step_stage2": steps,
    }, out_ckpt)
    return stage2
