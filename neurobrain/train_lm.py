"""Train the language brain from scratch on code + desktop-control transcripts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .data import Mixture
from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer


def cosine_lr(step: int, total: int, base: float, floor_ratio: float = 0.1, warmup: int = 100) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return base * floor_ratio
    p = (step - warmup) / max(1, total - warmup)
    floor = base * floor_ratio
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--code-train", default=None, help="override code train bin (e.g. corpus v2)")
    ap.add_argument("--code-val", default=None)
    ap.add_argument("--drive-train", default=None)
    ap.add_argument("--drive-val", default=None)
    ap.add_argument("--ui-train", default=None)
    ap.add_argument("--ui-val", default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--anneal-steps", type=int, default=0,
                    help="if >0: override the cosine schedule with a linear "
                         "anneal to 0 over this many steps from the resume "
                         "point (deliverable-3 final low-LR anneal experiment)")
    ap.add_argument("--anneal-lr", type=float, default=None,
                    help="anneal start lr (default: lr * 0.1, the cosine floor)")
    ap.add_argument("--n-layer", type=int, default=6,
                    help="transformer depth (deliverable-3 7-vs-8-layer experiment)")
    ap.add_argument("--n-head", type=int, default=7)
    ap.add_argument("--d-model", type=int, default=224)
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="if >0: maintain an EMA shadow of the weights "
                         "(deliverable-3 EMA experiment); ema eval is logged "
                         "alongside the raw eval and stored in checkpoints")
    ap.add_argument("--min_steps_between_evals", type=int, default=200)
    ap.add_argument("--time_budget", type=float, default=float("inf"), help="seconds; stop-training guard")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))

    def _bin(name: str, override: str | None) -> str:
        return override or os.path.join(args.data, f"{name}.bin")

    # code teaches the language, drive teaches it to finish, ui teaches the screen
    mix = Mixture({
        "code": (_bin("code_train", args.code_train), 0.55),
        "drive": (_bin("drive_train", args.drive_train), 0.25),
        "ui": (_bin("ui_train", args.ui_train), 0.20),
    }, args.ctx)
    val_mix = {
        "code": (_bin("code_val", args.code_val), 1.0),
        "drive": (_bin("drive_val", args.drive_val), 1.0),
        "ui": (_bin("ui_val", args.ui_val), 1.0),
    }
    val_sources = {k: __import__("neurobrain.data", fromlist=["TokenSource"]).TokenSource(v[0], args.ctx)
                   for k, v in val_mix.items()}
    val_batches = {}
    vrng = random.Random(1234)
    for k, src in val_sources.items():
        val_batches[k] = [src.sample(4, vrng) for _ in range(8)]

    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                    d_model=args.d_model, ctx=args.ctx)
    model = GPT(cfg)
    if args.compile:
        print("[lm] compiling model with torch.compile (inductor)")
        model = torch.compile(model)
    print(f"[lm] params: {model.num_params()/1e6:.2f}M (non-embedding {model.num_params(True)/1e6:.2f}M)")
    print("[lm] vocab: %d, tokens: " % tok.vocab_size +
          " ".join(f"{k}={len(v)/1e6:.2f}M" for k, v in mix.sources.items()))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(os.path.join(args.out, "lm_latest.pt")):
        ck = torch.load(os.path.join(args.out, "lm_latest.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
        print(f"[lm] resumed from step {step}")

    anneal_start = step
    anneal_from = args.anneal_lr if args.anneal_lr is not None else args.lr * 0.1

    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        if args.resume:
            try:
                ck_ema = torch.load(os.path.join(args.out, "lm_latest.pt"),
                                    map_location="cpu", weights_only=False)
                if isinstance(ck_ema.get("ema"), dict):
                    ema = {k: v.float() for k, v in ck_ema["ema"].items()}
                    print("[lm] resumed EMA shadow")
                del ck_ema
            except FileNotFoundError:
                pass

    rng = random.Random(args.seed + step)
    t_start = time.time()
    t0 = time.time()
    log_path = os.path.join("reports", "lm_log.jsonl")
    running = 0.0
    n_run = 0

    while step < args.steps:
        if time.time() - t_start > args.time_budget:
            print("[lm] time budget reached, stopping")
            break
        if args.anneal_steps > 0:
            lr = anneal_from * max(0.0, 1.0 - (step - anneal_start) / args.anneal_steps)
        else:
            lr = cosine_lr(step, args.steps, args.lr, warmup=args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_losses = {}
        for _ in range(args.accum):
            name, x, y = mix.sample(args.batch, rng)
            _, loss = model(x, y)
            (loss / args.accum).backward()
            step_losses[name] = step_losses.get(name, 0.0) + loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                d = args.ema_decay
                for k, v in model.state_dict().items():
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
        loss_val = sum(step_losses.values()) / len(step_losses)
        running += loss_val
        n_run += 1

        if step % 20 == 0:
            tok_s = 20 * args.batch * args.accum * args.ctx / max(1e-9, time.time() - t0)
            print(f"[lm] step {step:5d}/{args.steps}  loss {running/n_run:.4f}  ppl {math.exp(min(20, running/n_run)):.1f}"
                  f"  lr {lr:.2e}  {tok_s:.0f} tok/s  elapsed {time.time()-t0:.0f}s", flush=True)
            rec = {"step": step, "loss": running / n_run, "lr": lr, "tok_s": tok_s,
                   "elapsed": round(time.time() - t0, 1), **{f"loss_{k}": v for k, v in step_losses.items()}}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            running, n_run = 0.0, 0
            t0 = time.time()

        if step % args.min_steps_between_evals == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                evals = {}
                for k, batches in val_batches.items():
                    tot = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
                    evals[k] = tot
                mean_val = sum(evals.values()) / len(evals)
            ema_evals, ema_mean = None, None
            if ema is not None:
                raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(raw[k].dtype) for k, v in ema.items()})
                with torch.no_grad():
                    ema_evals = {k: sum(model(x, y)[1].item() for x, y in batches) / len(batches)
                                 for k, batches in val_batches.items()}
                ema_mean = sum(ema_evals.values()) / len(ema_evals)
                model.load_state_dict(raw)
            model.train()
            print(f"[lm] eval @ {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in evals.items()), flush=True)
            if ema_evals is not None:
                print(f"[lm] ema  @ {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in ema_evals.items())
                      + f"  (mean {ema_mean:.4f})", flush=True)
            rec = {"step": step, "eval": evals, "eval_mean": mean_val}
            if ema_evals is not None:
                rec["eval_ema"] = ema_evals
                rec["eval_mean_ema"] = ema_mean
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            improved = mean_val < best_val
            if improved:
                best_val = mean_val
            ck_latest = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                         "best_val": best_val, "cfg": cfg.to_dict()}
            if ema is not None:
                ck_latest["ema"] = ema
            torch.save(ck_latest, os.path.join(args.out, "lm_latest.pt"))
            if improved:
                # BUGFIX: lm_best used to be saved at EVERY eval (= last-eval
                # weights). Now it is the argmin of eval_mean, as the name says.
                ck_best = {"model": model.state_dict(), "step": step, "best_val": best_val,
                           "cfg": cfg.to_dict()}
                if ema is not None:
                    ck_best["ema"] = ema
                torch.save(ck_best, os.path.join(args.out, "lm_best.pt"))

    # final save
    ck_final = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "cfg": cfg.to_dict()}
    if ema is not None:
        ck_final["ema"] = ema
    torch.save(ck_final, os.path.join(args.out, "lm_latest.pt"))
    model.eval()
    with torch.no_grad():
        evals = {}
        for k, batches in val_batches.items():
            evals[k] = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
    summary = {"step": step, "eval": evals, "eval_mean": sum(evals.values()) / len(evals),
               "params": model.num_params(), "tokens_seen": step * args.batch * args.accum * args.ctx}
    with open(os.path.join("reports", "lm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[lm] done:", json.dumps(summary))


if __name__ == "__main__":
    main()
