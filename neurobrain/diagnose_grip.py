"""Comprehensive diagnostic harness + 100-miss failure gallery for GripNet & grounders.

Computes:
  - hit_widget, hit_point_6px, hit_point_8px, action_acc
  - median / mean / p90 centre error in pixels
  - single-frame latency (ms/frame on 2 CPU threads) and peak RSS (MB)
  - accuracy & median error breakdowns by role, label, theme, and widget size
  - miss taxonomy (same-role wrong-label vs wrong-role widget vs near-border vs background)
  - top confusions (label->label, role->role, and role:label -> role:label)
  - 100-miss PNG failure gallery with green target box, red crosshair, distractor box, and task caption
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import resource
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .gripnet import GripNet
from .ui_world import ACTION_CLASSES, CANVAS, encode_task, render, sample_example


def peak_rss_mb() -> float:
    # ru_maxrss is in KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def make_easy_val_split(n: int = 1000, seed: int = 202603) -> list[Any]:
    rng = random.Random(seed)
    return [sample_example(rng) for _ in range(n)]


def _size_bucket(area: float, scale_area: float = 1.0) -> str:
    norm_area = area / max(1e-6, scale_area)
    if norm_area < 500:
        return "small"
    if norm_area <= 900:
        return "medium"
    return "large"


def _summarize_group(items: list[tuple[bool, float]]) -> dict[str, Any]:
    if not items:
        return {"n": 0, "hit_widget": 0.0, "median_err_px": 0.0}
    hits = sum(1 for h, _ in items if h)
    errs = [e for _, e in items]
    return {
        "n": len(items),
        "hit_widget": round(hits / len(items), 4),
        "median_err_px": round(float(statistics.median(errs)), 2),
    }


def diagnose_predictions(
    examples: Sequence[Any],
    preds: Sequence[tuple[float, float, int]],
    canvas_w: int = 128,
    canvas_h: int = 128,
    render_fn: Callable[[Any], np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assert len(examples) == len(preds)
    n = len(examples)
    scale_area = (canvas_w * canvas_h) / (128.0 * 128.0)

    hits_widget = 0
    hits_6px = 0
    hits_8px = 0
    act_ok = 0
    errs: list[float] = []

    by_role: dict[str, list[tuple[bool, float]]] = collections.defaultdict(list)
    by_label: dict[str, list[tuple[bool, float]]] = collections.defaultdict(list)
    by_theme: dict[str, list[tuple[bool, float]]] = collections.defaultdict(list)
    by_size: dict[str, list[tuple[bool, float]]] = collections.defaultdict(list)

    miss_taxonomy = {
        "same_role_wrong_label": 0,
        "wrong_role_widget": 0,
        "near_miss_border": 0,
        "background_far": 0,
    }
    label_conf = collections.Counter()
    role_conf = collections.Counter()
    pair_conf = collections.Counter()

    misses: list[dict[str, Any]] = []

    for idx, (ex, (px, py, pred_act)) in enumerate(zip(examples, preds)):
        tgt = ex.target
        hit = bool(tgt.contains(px, py))
        err = math.hypot(px - tgt.cx, py - tgt.cy)
        errs.append(err)

        if hit:
            hits_widget += 1
        if err <= 6.0:
            hits_6px += 1
        if err <= 8.0:
            hits_8px += 1
        true_act = ACTION_CLASSES.index(ex.action) if isinstance(ex.action, str) else int(ex.action)
        if int(pred_act) == true_act:
            act_ok += 1

        area = tgt.w * tgt.h
        by_role[tgt.role].append((hit, err))
        by_label[tgt.label].append((hit, err))
        by_theme[ex.layout.theme].append((hit, err))
        by_size[_size_bucket(area, scale_area)].append((hit, err))

        if not hit:
            clicked_el = None
            for el in ex.layout.elements:
                if el is not tgt and el.contains(px, py, margin=2.0):
                    clicked_el = el
                    break

            if clicked_el is not None:
                if clicked_el.role == tgt.role:
                    miss_taxonomy["same_role_wrong_label"] += 1
                    why = f"same_role({clicked_el.label})"
                else:
                    miss_taxonomy["wrong_role_widget"] += 1
                    why = f"wrong_role({clicked_el.role}:{clicked_el.label})"
                label_conf[f"{tgt.label} -> {clicked_el.label}"] += 1
                role_conf[f"{tgt.role} -> {clicked_el.role}"] += 1
                pair_conf[f"{tgt.role}:{tgt.label} -> {clicked_el.role}:{clicked_el.label}"] += 1
            else:
                if tgt.contains(px, py, margin=6.0 * math.sqrt(scale_area)):
                    miss_taxonomy["near_miss_border"] += 1
                    why = "near_border"
                    role_conf[f"{tgt.role} -> near_border"] += 1
                else:
                    miss_taxonomy["background_far"] += 1
                    why = "background"
                    role_conf[f"{tgt.role} -> background"] += 1

            misses.append({
                "index": idx,
                "example": ex,
                "px": float(px),
                "py": float(py),
                "err_px": round(err, 2),
                "why": why,
                "clicked_el": clicked_el,
                "render_fn": render_fn,
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
            })

    errs_sorted = sorted(errs)
    p90_idx = min(len(errs_sorted) - 1, int(0.9 * len(errs_sorted))) if errs_sorted else 0

    report = {
        "n": n,
        "hit_widget": round(hits_widget / max(1, n), 4),
        "hit_point_6px": round(hits_6px / max(1, n), 4),
        "hit_point_8px": round(hits_8px / max(1, n), 4),
        "action_acc": round(act_ok / max(1, n), 4),
        "median_center_err_px": round(float(statistics.median(errs)) if errs else 0.0, 2),
        "mean_center_err_px": round(float(statistics.mean(errs)) if errs else 0.0, 2),
        "p90_center_err_px": round(float(errs_sorted[p90_idx]) if errs_sorted else 0.0, 2),
        "n_misses": len(misses),
        "miss_taxonomy": miss_taxonomy,
        "worst_confusions": {
            "top_3_pair_confusions": pair_conf.most_common(3),
            "top_10_pair_confusions": pair_conf.most_common(10),
            "label_confusions": label_conf.most_common(10),
            "role_confusions": role_conf.most_common(10),
        },
        "by_role": {k: _summarize_group(by_role[k]) for k in sorted(by_role)},
        "by_label": {k: _summarize_group(by_label[k]) for k in sorted(by_label)},
        "by_theme": {k: _summarize_group(by_theme[k]) for k in sorted(by_theme)},
        "by_size": {k: _summarize_group(by_size[k]) for k in ("small", "medium", "large") if k in by_size},
    }
    return report, misses


def render_failure_gallery(
    misses: Sequence[dict[str, Any]],
    out_path: str,
    max_tiles: int = 100,
    tile_w: int = 160,
    caption_h: int = 34,
    cols: int = 10,
) -> int:
    selected = list(misses[:max_tiles])
    if not selected:
        # Render a tiny banner indicating 0 failures
        sheet = Image.new("RGB", (320, 60), (20, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((12, 22), "0 failures on validation split", fill=(80, 230, 120))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        sheet.save(out_path, optimize=True)
        return 0

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        font = ImageFont.load_default()

    rows = (len(selected) + cols - 1) // cols
    tile_h = tile_w + caption_h
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (18, 18, 22))

    for i, m in enumerate(selected):
        ex = m["example"]
        rfn = m.get("render_fn")
        if rfn is not None:
            raw_img = rfn(ex)
        elif hasattr(ex, "image") and ex.image is not None:
            raw_img = ex.image
        else:
            raw_img = render(ex.layout)

        cw = m.get("canvas_w", raw_img.shape[1])
        ch = m.get("canvas_h", raw_img.shape[0])
        sx = tile_w / float(cw)
        sy = tile_w / float(ch)

        im_pil = Image.fromarray(raw_img).resize((tile_w, tile_w), Image.BILINEAR if cw > 128 else Image.NEAREST)
        tile = Image.new("RGB", (tile_w, tile_h), (14, 15, 20))
        tile.paste(im_pil, (0, 0))
        d = ImageDraw.Draw(tile)

        # Distractor box if clicked on wrong widget
        cel = m.get("clicked_el")
        if cel is not None:
            d.rectangle(
                [cel.x * sx, cel.y * sy, (cel.x + cel.w) * sx, (cel.y + cel.h) * sy],
                outline=(255, 195, 40),
                width=1,
            )

        # Target box in bright green
        t = ex.target
        d.rectangle(
            [t.x * sx, t.y * sy, (t.x + t.w) * sx, (t.y + t.h) * sy],
            outline=(0, 230, 90),
            width=2,
        )

        # Predicted click crosshair in bright red
        cx, cy = m["px"] * sx, m["py"] * sy
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(255, 50, 50), width=2)
        d.line([cx - 9, cy, cx + 9, cy], fill=(255, 50, 50), width=1)
        d.line([cx, cy - 9, cx, cy + 9], fill=(255, 50, 50), width=1)

        # Caption
        d.text((3, tile_w + 2), ex.task[:26], fill=(235, 238, 245), font=font)
        d.text(
            (3, tile_w + 13),
            f"tgt:{t.role[:4]}/{t.label[:8]} e={m['err_px']:.0f}p",
            fill=(150, 220, 170),
            font=font,
        )
        d.text((3, tile_w + 23), m["why"][:26], fill=(255, 160, 140), font=font)
        # Subtle tile border
        d.rectangle([0, 0, tile_w - 1, tile_h - 1], outline=(40, 42, 50), width=1)

        sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Quantize to 256-color PNG to keep 100-tile gallery crisp & small (~150-300 KB) for git
    sheet_q = sheet.quantize(colors=256, method=Image.MEDIANCUT)
    sheet_q.save(out_path, optimize=True)
    return len(selected)


def evaluate_gripnet_model(
    model: GripNet,
    examples: Sequence[Any],
    threads: int = 2,
    batch_size: int = 32,
    bench_frames: int = 64,
    canvas_size: int = CANVAS,
    render_128_fn: Callable[[Any], np.ndarray] | None = None,
    render_full_fn: Callable[[Any], np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.set_num_threads(threads)
    model.eval()
    preds: list[tuple[float, float, int]] = []
    r128 = render_128_fn or (lambda e: render(e.layout))
    rfull = render_full_fn or r128

    # Pre-render images & tasks in mini-batches so memory stays tiny
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            imgs = np.stack([r128(ex) for ex in chunk], axis=0)
            xs = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2)
            ts = torch.tensor([encode_task(ex.task) for ex in chunk], dtype=torch.long)
            scores, offsets, act_logits = model(xs, ts)
            xy = model.click_xy(scores, offsets, canvas_size).cpu().numpy()
            acts = act_logits.argmax(dim=1).cpu().numpy()
            for i in range(len(chunk)):
                preds.append((float(xy[i, 0]), float(xy[i, 1]), int(acts[i])))

        # Single-frame latency benchmark on 2 threads (warmed up)
        n_bench = min(bench_frames, len(examples))
        sample_imgs = [
            torch.from_numpy(r128(examples[i]).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
            for i in range(n_bench)
        ]
        sample_ts = [torch.tensor([encode_task(examples[i].task)], dtype=torch.long) for i in range(n_bench)]
        for i in range(min(4, n_bench)):
            s, o, _ = model(sample_imgs[i], sample_ts[i])
            _ = model.click_xy(s, o, canvas_size)
        latencies_ms = []
        for i in range(n_bench):
            t0 = time.perf_counter()
            s, o, _ = model(sample_imgs[i], sample_ts[i])
            _ = model.click_xy(s, o, canvas_size)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    rep, misses = diagnose_predictions(examples, preds, canvas_w=canvas_size, canvas_h=canvas_size, render_fn=rfull)
    rep["latency_ms_per_frame"] = round(float(statistics.median(latencies_ms)), 2)
    rep["latency_mean_ms"] = round(float(statistics.mean(latencies_ms)), 2)
    rep["peak_rss_mb"] = peak_rss_mb()
    rep["params"] = sum(p.numel() for p in model.parameters())
    return rep, misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grip_latest.pt")
    ap.add_argument("--world", choices=["easy", "hard", "hard-ood"], default="easy")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=202603)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-json", default="reports/m0_diagnosis.json")
    ap.add_argument("--out-gallery", default="samples/m0_failures_100.png")
    args = ap.parse_args()

    model = GripNet()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    step = ck.get("step", -1) if isinstance(ck, dict) else -1

    if args.world == "easy":
        split = make_easy_val_split(n=args.n, seed=args.seed)
        rep, misses = evaluate_gripnet_model(model, split, threads=args.threads)
    else:
        from .ui_world_hard import HARD_CANVAS, make_hard_val_split, render_hard

        split_kind = "seen" if args.world == "hard" else "ood"
        split = make_hard_val_split(n=args.n, seed=args.seed, split=split_kind)
        rep, misses = evaluate_gripnet_model(
            model,
            split,
            threads=args.threads,
            canvas_size=HARD_CANVAS,
            render_128_fn=lambda e: render_hard(e.layout, size=CANVAS),
            render_full_fn=lambda e: render_hard(e.layout, size=HARD_CANVAS),
        )
    rep["world"] = args.world
    rep["step"] = step
    rep["ckpt"] = args.ckpt

    n_drawn = render_failure_gallery(misses, args.out_gallery, max_tiles=100)
    rep["gallery_tiles"] = n_drawn
    rep["gallery_path"] = args.out_gallery

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2)

    print(json.dumps({
        "step": step,
        "params": rep["params"],
        "hit_widget": rep["hit_widget"],
        "median_center_err_px": rep["median_center_err_px"],
        "action_acc": rep["action_acc"],
        "latency_ms_per_frame": rep["latency_ms_per_frame"],
        "peak_rss_mb": rep["peak_rss_mb"],
        "miss_taxonomy": rep["miss_taxonomy"],
        "top_3_pair_confusions": rep["worst_confusions"]["top_3_pair_confusions"],
    }, indent=2))


if __name__ == "__main__":
    main()
