"""Train GripNet: instruction-conditioned screen pointing, from scratch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gripnet import GripNet
from .ui_world import batch, ACTION_CLASSES, CANVAS
from .ui_world import sample_example, render


def evaluate(model: GripNet, rng: random.Random, n: int = 256) -> dict:
    model.eval()
    hits_widget = hits_point = act_ok = 0
    iou_sum = 0.0
    with torch.no_grad():
        for _ in range(4):
            xs, tasks, targets, offs, actions, examples = batch(n // 4, rng)
            scores, offsets, act_logits = model(
                torch.from_numpy(xs).permute(0, 3, 1, 2), torch.from_numpy(tasks))
            xy = model.click_xy(scores, offsets, CANVAS)
            pred_act = act_logits.argmax(1).numpy()
            for i, ex in enumerate(examples):
                px, py = xy[i, 0].item(), xy[i, 1].item()
                if ex.target.contains(px, py):
                    hits_widget += 1
                if math.hypot(px - ex.target.cx, py - ex.target.cy) <= 6:
                    hits_point += 1
                if pred_act[i] == actions[i]:
                    act_ok += 1
                # IoU of an 8x8 bbox around the click vs the widget box
                bx0, by0 = max(0, px - 4), max(0, py - 4)
                bx1, by1 = min(CANVAS - 1, px + 4), min(CANVAS - 1, py + 4)
                ix0, iy0 = max(bx0, ex.target.x), max(by0, ex.target.y)
                ix1, iy1 = min(bx1, ex.target.x + ex.target.w), min(by1, ex.target.y + ex.target.h)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = 64 + ex.target.w * ex.target.h - inter
                iou_sum += inter / max(1, union)
    model.train()
    return {
        "hit_widget": hits_widget / n,
        "hit_point": hits_point / n,
        "action_acc": act_ok / n,
        "click_iou": iou_sum / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--time_budget", type=float, default=float("inf"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model = GripNet()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    step = 0
    if args.resume and os.path.exists(os.path.join(args.out, "grip_latest.pt")):
        ck = torch.load(os.path.join(args.out, "grip_latest.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"[grip] resumed from step {step}")

    rng = random.Random(args.seed + step)
    evrng = random.Random(99)
    t0 = time.time()
    log_path = os.path.join("reports", "grip_log.jsonl")
    print(f"[grip] params: {n_params/1e6:.2f}M, batch {args.batch}, steps {args.steps}")
    running, n_run = 0.0, 0

    while step < args.steps:
        if time.time() - t0 > args.time_budget:
            print("[grip] time budget reached, stopping")
            break
        xs, tasks, targets, offs, actions, _ = batch(args.batch, rng)
        scores, offsets, act_logits = model(
            torch.from_numpy(xs).permute(0, 3, 1, 2), torch.from_numpy(tasks))
        loss, parts = model.loss(scores, offsets, act_logits, torch.from_numpy(targets),
                                 torch.from_numpy(offs), torch.from_numpy(actions).long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))))
        step += 1
        running += loss.item()
        n_run += 1

        if step % 25 == 0:
            print(f"[grip] step {step}/{args.steps} loss {running/n_run:.4f} "
                  f"ce {parts['ce']:.4f} off {parts['off']:.3f} act {parts['act']:.4f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": running / n_run, **parts,
                                    "elapsed": round(time.time() - t0, 1)}) + "\n")
            running, n_run = 0.0, 0

        if step % 250 == 0 or step == args.steps:
            ev = evaluate(model, evrng, 256)
            print(f"[grip] eval @ {step}: " + "  ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "eval": ev}) + "\n")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                       os.path.join(args.out, "grip_latest.pt"))
            torch.save({"model": model.state_dict(), "step": step, "eval": ev},
                       os.path.join(args.out, "grip_best.pt"))

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
               os.path.join(args.out, "grip_latest.pt"))
    ev = evaluate(model, random.Random(7), 400)
    with open(os.path.join("reports", "grip_summary.json"), "w") as f:
        json.dump({"step": step, "eval": ev, "params": n_params}, f, indent=2)
    print("[grip] done:", json.dumps(ev))


if __name__ == "__main__":
    main()
