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
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=150)
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
    mix = Mixture({
        "code": (os.path.join(args.data, "code_train.bin"), 0.75),
        "ui": (os.path.join(args.data, "ui_train.bin"), 0.25),
    }, args.ctx)
    val_mix = {
        "code": (os.path.join(args.data, "code_val.bin"), 1.0),
        "ui": (os.path.join(args.data, "ui_val.bin"), 1.0),
    }
    val_sources = {k: __import__("neurobrain.data", fromlist=["TokenSource"]).TokenSource(v[0], args.ctx)
                   for k, v in val_mix.items()}
    val_batches = {}
    vrng = random.Random(1234)
    for k, src in val_sources.items():
        val_batches[k] = [src.sample(4, vrng) for _ in range(8)]

    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=8, n_head=8, d_model=256, ctx=args.ctx)
    model = GPT(cfg)
    if args.compile:
        print("[lm] compiling model with torch.compile (inductor)")
        model = torch.compile(model)
    print(f"[lm] params: {model.num_params()/1e6:.2f}M (non-embedding {model.num_params(True)/1e6:.2f}M)")
    print(f"[lm] vocab: {tok.vocab_size}, tokens: code={len(mix.sources['code'])/1e6:.2f}M "
          f"ui={len(mix.sources['ui'])/1e6:.2f}M")

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

    rng = random.Random(args.seed + step)
    t0 = time.time()
    log_path = os.path.join("reports", "lm_log.jsonl")
    running = 0.0
    n_run = 0

    while step < args.steps:
        if time.time() - t0 > args.time_budget:
            print("[lm] time budget reached, stopping")
            break
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
            model.train()
            print(f"[lm] eval @ {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in evals.items()), flush=True)
            rec = {"step": step, "eval": evals, "eval_mean": mean_val}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if mean_val < best_val:
                best_val = mean_val
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "best_val": best_val, "cfg": cfg.to_dict()},
                       os.path.join(args.out, "lm_latest.pt"))
            torch.save({"model": model.state_dict(), "step": step, "best_val": best_val,
                        "cfg": cfg.to_dict()}, os.path.join(args.out, "lm_best.pt"))

    # final save
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_val": best_val, "cfg": cfg.to_dict()},
               os.path.join(args.out, "lm_latest.pt"))
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
