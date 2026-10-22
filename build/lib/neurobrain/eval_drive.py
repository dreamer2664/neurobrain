"""Measure the drive: does the model finish what it starts?

Four numbers, all computed from generations conditioned on a fresh objective:

  terminal     -- fraction of episodes that end with <|done|> inside the budget
  retry        -- fraction that emit a <|retry|> after their first failed check
  steps        -- mean steps before <|done|>
  honest       -- fraction whose final <|check|> says satisfied before <|done|>

`terminal` is the headline. A model that writes pretty plans but stops without
<|done|> is worth nothing to this project.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from .sample import load
from .ui_world import CANVAS, make_layout, sample_example, TASKS


def objectives(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ex = sample_example(rng)
        out.append(f"<|drive|>objective: {rng.choice(TASKS[ex.target.role]).format(label=ex.target.label)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_best.pt")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--tokens", type=int, default=220)
    ap.add_argument("--out", default="reports/drive_eval.json")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[drive-eval] no checkpoint at {args.ckpt}")
        return
    model, tok, step = load(args.ckpt)
    eot = tok.special_id("<|endoftext|>")

    terminal = retry_after_fail = honest = 0
    steps_total = 0
    examples = []
    for obj in objectives(args.n, seed=11):
        ids = torch.tensor([tok.encode(obj)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=args.tokens, temperature=0.4, top_k=30, eos_id=eot)
        text = tok.decode(out[0].tolist())[len(obj):]
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>")[0]
        ok = "<|done|>" in text
        terminal += ok
        if "<|check|>" in text and "not satisfied" in text:
            retry_after_fail += "<|retry|>" in text.split("not satisfied", 1)[1]
        if ok:
            tail = text.split("<|done|>")[0]
            honest += "objective satisfied" in tail
            steps_total += tail.count("<|step|>") + tail.count("<|try|>")
        if len(examples) < 6:
            examples.append({"objective": obj, "terminal": ok,
                             "text": text[:400]})

    result = {
        "step": step,
        "terminal_rate": round(terminal / args.n, 3),
        "retry_after_failure": round(retry_after_fail / max(1, args.n), 3),
        "honest_completion": round(honest / max(1, terminal), 3),
        "mean_steps": round(steps_total / max(1, terminal), 1),
        "n": args.n,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({**result, "examples": examples}, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
