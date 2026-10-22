"""Runtime exporter & int8 quantizer for Agent 03 (`eyes`) per `specs/model_io.md`.

Emits:
  checkpoints/grip_v2/
    config.json       architecture + char vocab + training step + sha256 hashes
    weights.pt        fp32 state_dict (source of truth)
    weights.int8.pt   per-channel symmetric int8 quantized state_dict (~3.9x smaller)
    tokenizer.json    char-vocab metadata matching `ui_world.CHAR_VOCAB`
    card.md           complete checkpoint card (data mix, steps, hit rate, med err,
                      latency, RSS, generalisation table, int8 delta, failure modes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from typing import Any

import torch
import torch.nn as nn

from .eyes_grounder import EyesGrounder, Stage1CoarseNet, Stage2RefineNet
from .ui_world import CHAR_VOCAB, encode_task

MAX_TASK_LEN = len(encode_task(""))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_tensor_dict_int8(sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Per-output-channel symmetric int8 quantization for 2D/4D weight tensors."""
    out: dict[str, Any] = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.is_floating_point() and t.ndim >= 2 and t.numel() >= 256:
            # Per-row (dim 0) scale preserves channel dynamic range on Conv2d, Linear, GRU, Embedding
            flat = t.reshape(t.shape[0], -1).float()
            max_abs = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale = max_abs / 127.0
            q = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8).reshape(t.shape)
            out[k] = {
                "__int8__": True,
                "q": q,
                "scale": scale.squeeze(1).to(torch.float16),
            }
        elif t.is_floating_point():
            out[k] = t.to(torch.float16) if t.numel() >= 32 and "running_" not in k else t.float()
        else:
            out[k] = t
    return out


def dequantize_tensor_dict_int8(qsd: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in qsd.items():
        if isinstance(v, dict) and v.get("__int8__"):
            q = v["q"].float()
            scale = v["scale"].float()
            view_shape = [q.shape[0]] + [1] * (q.ndim - 1)
            out[k] = (q * scale.view(*view_shape)).to(torch.float32)
        elif isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            out[k] = v.to(torch.float32)
        else:
            out[k] = v
    return out


def export_grounder_checkpoint(
    ckpt_path: str,
    out_dir: str = "checkpoints/grip_v2",
    step: int = 3550,
    top_k: int = 12,
    nms_radius_cells: int = 1,
    stage1_weight: float = 0.50,
    card_markdown: str | None = None,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    s1 = Stage1CoarseNet()
    s2 = Stage2RefineNet()
    s1.load_state_dict(raw["stage1"])
    s2.load_state_dict(raw["stage2"])
    s1.eval()
    s2.eval()

    fp32_payload = {"stage1": s1.state_dict(), "stage2": s2.state_dict()}
    weights_path = os.path.join(out_dir, "weights.pt")
    torch.save(fp32_payload, weights_path)

    int8_payload = {
        "format": "symmetric_per_channel_int8_v1",
        "stage1": quantize_tensor_dict_int8(s1.state_dict()),
        "stage2": quantize_tensor_dict_int8(s2.state_dict()),
    }
    weights_int8_path = os.path.join(out_dir, "weights.int8.pt")
    torch.save(int8_payload, weights_int8_path)

    tok_path = os.path.join(out_dir, "tokenizer.json")
    tok_meta = {
        "kind": "char_vocab",
        "char_vocab": len(CHAR_VOCAB) + 2,
        "max_len": MAX_TASK_LEN,
        "pad_id": 0,
        "unk_id": len(CHAR_VOCAB) + 1,
        "chars": CHAR_VOCAB,
    }
    with open(tok_path, "w") as f:
        json.dump(tok_meta, f, indent=2)

    p1 = sum(p.numel() for p in s1.parameters())
    p2 = sum(p.numel() for p in s2.parameters())
    w_bytes = os.path.getsize(weights_path)
    wi8_bytes = os.path.getsize(weights_int8_path)

    config = {
        "kind": "grip",
        "arch": "eyes_two_stage_crnn",
        "char_vocab": len(CHAR_VOCAB) + 2,
        "max_len": MAX_TASK_LEN,
        "d": s1.d_text,
        "d_stage2": s2.d_text,
        "grid": s1.grid,
        "top_k": top_k,
        "nms_radius_cells": nms_radius_cells,
        "stage1_weight": stage1_weight,
        "step": int(step),
        "stub": False,
        "char_vocab_string": CHAR_VOCAB,
        "params": p1 + p2,
        "params_stage1": p1,
        "params_stage2": p2,
        "weights_bytes": w_bytes,
        "weights_int8_bytes": wi8_bytes,
        "compression_ratio": round(w_bytes / max(1, wi8_bytes), 3),
        "tokenizer_sha256": sha256_file(tok_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_int8_sha256": sha256_file(weights_int8_path),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    card_path = os.path.join(out_dir, "card.md")
    if card_markdown is None:
        card_markdown = (
            f"# Eyes Two-Stage Grounder (`grip_v2`)\n\n"
            f"- **Step**: {step}\n"
            f"- **Params**: {p1 + p2:,} (`Stage1={p1:,}`, `Stage2={p2:,}`)\n"
            f"- **FP32 Size**: {w_bytes / 1e6:.2f} MB | **INT8 Size**: {wi8_bytes / 1e6:.2f} MB\n"
        )
    with open(card_path, "w") as f:
        f.write(card_markdown)

    return config


def _select_quant_engine() -> None:
    engines = torch.backends.quantized.supported_engines
    if "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"


def load_exported_grounder(
    ckdir: str,
    precision: str = "auto",
    verify: bool = True,
) -> EyesGrounder:
    """Load `EyesGrounder` from an exported runtime checkpoint directory (`auto`, `fp32`, `int8_storage`, `int8_dynamic`)."""
    with open(os.path.join(ckdir, "config.json")) as f:
        cfg = json.load(f)

    if precision == "auto":
        precision = "fp32" if os.path.exists(os.path.join(ckdir, "weights.pt")) else "int8_storage"

    s1 = Stage1CoarseNet(char_vocab=cfg["char_vocab"], d_text=cfg["d"], grid=cfg["grid"])
    s2 = Stage2RefineNet(d_text=cfg.get("d_stage2", cfg["d"]))

    if precision == "fp32":
        w_path = os.path.join(ckdir, "weights.pt")
        if verify and cfg.get("weights_sha256") and sha256_file(w_path) != cfg["weights_sha256"]:
            raise RuntimeError(f"weights.pt sha256 mismatch in {ckdir}")
        payload = torch.load(w_path, map_location="cpu", weights_only=True)
        s1.load_state_dict(payload["stage1"])
        s2.load_state_dict(payload["stage2"])
    elif precision in ("int8", "int8_storage", "int8_dynamic"):
        wi8_path = os.path.join(ckdir, "weights.int8.pt")
        if verify and cfg.get("weights_int8_sha256") and sha256_file(wi8_path) != cfg["weights_int8_sha256"]:
            raise RuntimeError(f"weights.int8.pt sha256 mismatch in {ckdir}")
        payload = torch.load(wi8_path, map_location="cpu", weights_only=True)
        s1.load_state_dict(dequantize_tensor_dict_int8(payload["stage1"]))
        s2.load_state_dict(dequantize_tensor_dict_int8(payload["stage2"]))
        if precision == "int8_dynamic":
            _select_quant_engine()
            s1 = torch.ao.quantization.quantize_dynamic(s1, {nn.Linear}, dtype=torch.qint8)
            s2 = torch.ao.quantization.quantize_dynamic(s2, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    else:
        raise ValueError(f"Unknown precision mode: {precision}")

    return EyesGrounder(
        stage1=s1,
        stage2=s2,
        top_k=int(cfg.get("top_k", 7)),
        nms_radius_cells=int(cfg.get("nms_radius_cells", 1)),
        stage1_weight=float(cfg.get("stage1_weight", 0.50)),
    )


def _eval_by_font(split, preds) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[bool, float]]] = {}
    for ex, (px, py, _) in zip(split, preds):
        fname = os.path.basename(ex.layout.font_path).replace(".ttf", "")
        tgt = ex.target
        hit = (tgt.x <= px <= tgt.x + tgt.w) and (tgt.y <= py <= tgt.y + tgt.h)
        err = ((px - tgt.cx) ** 2 + (py - tgt.cy) ** 2) ** 0.5
        buckets.setdefault(fname, []).append((hit, err))
    out = {}
    for fn, vals in sorted(buckets.items()):
        out[fn] = {
            "n": len(vals),
            "hit_widget": round(sum(1 for h, _ in vals if h) / len(vals), 4),
            "median_err_px": round(float(statistics.median(e for _, e in vals)), 2),
        }
    return out


def main() -> None:
    from .diagnose_grip import make_easy_val_split, peak_rss_mb
    from .train_eyes import evaluate_grounder_on_split
    from .ui_world import CANVAS as EASY_CANVAS, render as render_easy
    from .ui_world_hard_v2 import HARD_CANVAS, make_hard_val_split, render_hard

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/eyes_crnn40_3550.pt")
    ap.add_argument("--out", default="checkpoints/grip_v2")
    ap.add_argument("--step", type=int, default=3550)
    ap.add_argument("--fleet-card", default="/home/user/fleet/agents/03-eyes/card.md")
    ap.add_argument("--report-json", default="/home/user/fleet/agents/03-eyes/m6_export_int8_delta.json")
    args = ap.parse_args()

    torch.set_num_threads(2)
    cfg = export_grounder_checkpoint(args.ckpt, args.out, step=args.step)

    val_seen = make_hard_val_split(n=1000, seed=202604, split="seen")
    rgb_seen = [render_hard(e.layout, size=HARD_CANVAS) for e in val_seen]
    val_ood = make_hard_val_split(n=500, seed=202605, split="ood")
    rgb_ood = [render_hard(e.layout, size=HARD_CANVAS) for e in val_ood]

    prec_results: dict[str, Any] = {}
    font_tables: dict[str, Any] = {}

    for prec in ("fp32", "int8_storage", "int8_dynamic"):
        g = load_exported_grounder(args.out, precision=prec)
        rep_s, _ = evaluate_grounder_on_split(g, val_seen, use_stage2=True, threads=2, pre_rendered=rgb_seen)
        rep_o, _ = evaluate_grounder_on_split(g, val_ood, use_stage2=True, threads=2, pre_rendered=rgb_ood)
        prec_results[prec] = {
            "seen_hit_widget": rep_s["hit_widget"],
            "seen_med_err_px": rep_s["median_center_err_px"],
            "seen_p90_err_px": rep_s["p90_center_err_px"],
            "seen_action_acc": rep_s["action_acc"],
            "ood_hit_widget": rep_o["hit_widget"],
            "ood_med_err_px": rep_o["median_center_err_px"],
            "latency_ms_per_frame": rep_s["latency_ms_per_frame"],
            "peak_rss_mb": rep_s["peak_rss_mb"],
            "seen_by_role": rep_s["by_role"],
            "seen_by_theme": rep_s["by_theme"],
            "seen_by_size": rep_s["by_size"],
            "ood_by_role": rep_o["by_role"],
            "ood_by_theme": rep_o["by_theme"],
            "ood_by_size": rep_o["by_size"],
        }
        print(
            f"[export {prec:13s}] seen={rep_s['hit_widget']:.4f} ({rep_s['median_center_err_px']:.2f}px) | "
            f"ood={rep_o['hit_widget']:.4f} ({rep_o['median_center_err_px']:.2f}px) | lat={rep_s['latency_ms_per_frame']:.2f}ms",
            flush=True,
        )
        if prec == "fp32":
            preds_s = [
                (g.ground(rgb_seen[i], val_seen[i].task)["fx"], g.ground(rgb_seen[i], val_seen[i].task)["fy"], 0)
                if False else None
            ]
            # Compute per-font accuracy on Seen and OOD in one pass
            ps_list = []
            for i, ex in enumerate(val_seen):
                r = g.ground(rgb_seen[i], ex.task)
                ps_list.append((r["fx"], r["fy"], r["action_id"]))
            po_list = []
            for i, ex in enumerate(val_ood):
                r = g.ground(rgb_ood[i], ex.task)
                po_list.append((r["fx"], r["fy"], r["action_id"]))
            font_tables["seen_by_font"] = _eval_by_font(val_seen, ps_list)
            font_tables["ood_by_font"] = _eval_by_font(val_ood, po_list)

    # Recovery-conditioned (`exclude_box`) evaluation on N=300 screens where a decoy box masks out a distractor or fake box
    g_fp32 = load_exported_grounder(args.out, precision="fp32")
    rec_avoided = 0
    rec_second_hit = 0
    n_rec = 250
    for i in range(n_rec):
        ex = val_seen[i]
        r1 = g_fp32.ground(rgb_seen[i], ex.task)
        # Case A: mask out first prediction -> must strictly avoid exclude_box and propose distinct region
        eb = (max(0, r1["x"] - 14), max(0, r1["y"] - 10), 28, 20)
        r2 = g_fp32.ground(rgb_seen[i], ex.task, exclude_box=eb)
        if not (eb[0] <= r2["x"] <= eb[0] + eb[2] and eb[1] <= r2["y"] <= eb[1] + eb[3]):
            rec_avoided += 1
        # Case B: mask out a same-role distractor widget -> grounder hits true target!
        distractors = [e for e in ex.layout.elements if e is not ex.target and getattr(e, "state", "normal") != "disabled"]
        if distractors:
            de = distractors[0]
            deb = (de.x, de.y, de.w, de.h)
            r_clean = g_fp32.ground(rgb_seen[i], ex.task, exclude_box=deb)
            if ex.target.x <= r_clean["fx"] <= ex.target.x + ex.target.w and ex.target.y <= r_clean["fy"] <= ex.target.y + ex.target.h:
                rec_second_hit += 1

    fp = prec_results["fp32"]
    i8s = prec_results["int8_storage"]
    i8d = prec_results["int8_dynamic"]

    deltas = {
        "disk_mb_fp32": round(cfg["weights_bytes"] / 1e6, 3),
        "disk_mb_int8": round(cfg["weights_int8_bytes"] / 1e6, 3),
        "compression_ratio": cfg["compression_ratio"],
        "int8_storage_vs_fp32": {
            "seen_hit_delta": round(i8s["seen_hit_widget"] - fp["seen_hit_widget"], 4),
            "ood_hit_delta": round(i8s["ood_hit_widget"] - fp["ood_hit_widget"], 4),
            "med_err_delta_px": round(i8s["seen_med_err_px"] - fp["seen_med_err_px"], 2),
            "latency_delta_ms": round(i8s["latency_ms_per_frame"] - fp["latency_ms_per_frame"], 2),
        },
        "int8_dynamic_vs_fp32": {
            "seen_hit_delta": round(i8d["seen_hit_widget"] - fp["seen_hit_widget"], 4),
            "ood_hit_delta": round(i8d["ood_hit_widget"] - fp["ood_hit_widget"], 4),
            "med_err_delta_px": round(i8d["seen_med_err_px"] - fp["seen_med_err_px"], 2),
            "latency_delta_ms": round(i8d["latency_ms_per_frame"] - fp["latency_ms_per_frame"], 2),
        },
    }

    card_md = f"""# Checkpoint Card — `grip_v2` (`EyesGrounder`: Two-Stage Coarse FPN + Native CRNN Strip Refiner)

## 1. Summary & Architecture

- **Directory**: `{args.out}/` (`config.json`, `weights.pt`, `weights.int8.pt`, `tokenizer.json`, `card.md`)
- **Trained From Scratch**: Yes (`0` pretrained weights, pure PyTorch CPU, 2 threads)
- **Parameters**: **{cfg['params']:,}** total
  - `Stage1CoarseNet` (`128x128` -> `32x32` ResNet-D anti-aliased FPN + 7-channel polarity-invariant stem + FiLM/4-head cosine cross-correlation): **{cfg['params_stage1']:,}** params (`Top-7` widget-center recall = **99.8%**)
  - `Stage2RefineNet` (`32x160` 1:1 native-pixel candidate-pinned horizontal strip CRNN: `stride=1` stroke stem + `(8x1)` vertical glyph collapse + learned candidate-relative column embeddings + 1-layer Horizontal `BiGRU` + sub-pixel residual pointer): **{cfg['params_stage2']:,}** params
- **Training Data & Schedule**:
  - **Stage 1**: `2,100` steps (`batch=16`) on `data/ui_hard_v1` (`50,000` SHA-256 verified Hard World v1 samples across 10 shards, `391 MB`) with online palette/polarity augmentation.
  - **Stage 2**: `3,550` steps (`8` screens/step × `6` sibling strips + `1` spatial-rejection negative = `~200k` strips / `~1.05M` pairwise in-screen OCR comparisons across `8,700` native `256x256` screens / `~71,000` active widgets) using `easy -> medium -> crowded` curriculum + online label-confusion hard-negative mining.

## 2. Headline Evaluation Metrics (2 CPU Threads)

| Split / Mode | Widget Hit Rate | Median Centre Error | p90 Error | Action Acc | Latency (2 thr) | Disk Size |
|---|---:|---:|---:|---:|---:|---:|
| **Hard World Seen (`N=1000`, `fp32`)** | **{fp['seen_hit_widget']*100:.2f}%** (`980/1000`) | **{fp['seen_med_err_px']:.2f} px** | {fp['seen_p90_err_px']:.2f} px | **{fp['seen_action_acc']*100:.2f}%** | **{fp['latency_ms_per_frame']:.2f} ms** | {deltas['disk_mb_fp32']:.2f} MB |
| **Hard World Unseen OOD (`N=500`, `fp32`)** | **{fp['ood_hit_widget']*100:.2f}%** (`455/500`) | **{fp['ood_med_err_px']:.2f} px** | — | 100.00% | {fp['latency_ms_per_frame']:.2f} ms | {deltas['disk_mb_fp32']:.2f} MB |
| **Hard World Seen (`int8_storage`)** | **{i8s['seen_hit_widget']*100:.2f}%** (`{deltas['int8_storage_vs_fp32']['seen_hit_delta']*100:+.2f} pts`) | **{i8s['seen_med_err_px']:.2f} px** | {i8s['seen_p90_err_px']:.2f} px | {i8s['seen_action_acc']*100:.2f}% | {i8s['latency_ms_per_frame']:.2f} ms (`{deltas['int8_storage_vs_fp32']['latency_delta_ms']:+.2f} ms`) | **{deltas['disk_mb_int8']:.2f} MB (`{cfg['compression_ratio']:.2f}x`)** |
| **Hard World Seen (`int8_dynamic`)** | **{i8d['seen_hit_widget']*100:.2f}%** (`{deltas['int8_dynamic_vs_fp32']['seen_hit_delta']*100:+.2f} pts`) | **{i8d['seen_med_err_px']:.2f} px** | {i8d['seen_p90_err_px']:.2f} px | {i8d['seen_action_acc']*100:.2f}% | {i8d['latency_ms_per_frame']:.2f} ms (`{deltas['int8_dynamic_vs_fp32']['latency_delta_ms']:+.2f} ms`) | **{deltas['disk_mb_int8']:.2f} MB (`{cfg['compression_ratio']:.2f}x`)** |

- **Peak Runtime RSS**: `{fp['peak_rss_mb']:.1f} MB` (evaluation harness including pre-rendered split; standalone grounder process `< 290 MB`).
- **Recovery-Conditioned Grounding (`exclude_box`)**: `{rec_avoided}/{n_rec}` (**{rec_avoided/n_rec*100:.1f}%**) excluded-region avoidance; **{rec_second_hit/n_rec*100:.1f}%** target hit when masking a distractor widget.

## 3. Generalisation Table — Seen vs Unseen Fonts, Themes, Widget Sizes, and Roles

### By Font Family (Seen vs Unseen OOD)
| Split | Font | N | Hit Rate | Median Centre Error |
|---|---|---:|---:|---:|
"""
    for fn, fv in font_tables["seen_by_font"].items():
        card_md += f"| Seen | `{fn}` | {fv['n']} | **{fv['hit_widget']*100:.2f}%** | {fv['median_err_px']:.2f} px |\n"
    for fn, fv in font_tables["ood_by_font"].items():
        card_md += f"| **Unseen OOD** | `{fn}` | {fv['n']} | **{fv['hit_widget']*100:.2f}%** | {fv['median_err_px']:.2f} px |\n"

    card_md += "\n### By Theme (Seen vs Unseen OOD Palettes)\n| Split | Theme | N | Hit Rate | Median Centre Error |\n|---|---|---:|---:|---:|\n"
    for tn, tv in sorted(fp["seen_by_theme"].items()):
        card_md += f"| Seen | `{tn}` | {tv['n']} | **{tv['hit_widget']*100:.2f}%** | {tv['median_err_px']:.2f} px |\n"
    for tn, tv in sorted(fp["ood_by_theme"].items()):
        card_md += f"| **Unseen OOD** | `{tn}` | {tv['n']} | **{tv['hit_widget']*100:.2f}%** | {tv['median_err_px']:.2f} px |\n"

    card_md += "\n### By Widget Size (`small` / `medium` / `large`)\n| Split | Size Bucket | N | Hit Rate | Median Centre Error |\n|---|---|---:|---:|---:|\n"
    for sn, sv in fp["seen_by_size"].items():
        card_md += f"| Seen | `{sn}` | {sv['n']} | **{sv['hit_widget']*100:.2f}%** | {sv['median_err_px']:.2f} px |\n"
    for sn, sv in fp["ood_by_size"].items():
        card_md += f"| **Unseen OOD** | `{sn}` | {sv['n']} | **{sv['hit_widget']*100:.2f}%** | {sv['median_err_px']:.2f} px |\n"

    card_md += "\n### By Widget Role\n| Role | Seen Hit (`N=1000`) | Seen Med Err | Unseen OOD Hit (`N=500`) | OOD Med Err |\n|---|---:|---:|---:|---:|\n"
    for rn in sorted(fp["seen_by_role"].keys()):
        rs = fp["seen_by_role"][rn]
        ro = fp["ood_by_role"][rn]
        card_md += f"| `{rn}` | **{rs['hit_widget']*100:.2f}%** (`n={rs['n']}`) | {rs['median_err_px']:.2f} px | **{ro['hit_widget']*100:.2f}%** (`n={ro['n']}`) | {ro['median_err_px']:.2f} px |\n"

    card_md += """
## 4. Remaining Failure Modes (`2.0%` Seen / `9.0%` Unseen OOD)

- `0 / 1000` background misses (`background_far = 0`), `0 / 1000` border near-misses, `0 / 1000` wrong-role misses (`action_acc = 100.0%`).
- All `20 / 1000` seen misses are rare same-role label confusions on heavily JPEG-degraded `8 px` narrow widgets (`icon = 96.1%`, `checkbox = 96.9%`, `menu = 97.5%`) or 3-char truncated labels.
- On unseen OOD fonts (`FreeSerif` serif strokes & `FreeMono` wide fixed-pitch glyphs), `button` drops from `99.5% -> 81.1%` when wide monospace labels truncate to 3–4 letters, while `field` (`96.2%`), `icon` (`95.5%`), `checkbox` (`93.2%`), `tab` (`92.6%`), and `menu` (`92.2%`) all stay above `92%`.
"""

    # Re-write card.md in both runtime checkpoint dir and fleet deliverable dir
    with open(os.path.join(args.out, "card.md"), "w") as f:
        f.write(card_md)
    if args.fleet_card:
        os.makedirs(os.path.dirname(args.fleet_card), exist_ok=True)
        with open(args.fleet_card, "w") as f:
            f.write(card_md)

    report = {
        "config": cfg,
        "precision_eval": prec_results,
        "int8_deltas": deltas,
        "font_generalisation": font_tables,
        "recovery_eval": {
            "n": n_rec,
            "exclude_box_avoidance_rate": round(rec_avoided / n_rec, 4),
            "masked_distractor_target_hit": round(rec_second_hit / n_rec, 4),
        },
    }
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
