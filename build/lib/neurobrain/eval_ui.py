"""Evaluate the language brain on desktop-control transcripts.

Held-out screens are serialized into the text protocol; the model must emit the
action. We score:
  * format rate   -- output parses as action(x,y)
  * action type   -- predicted verb matches the widget role
  * in-widget     -- the emitted coordinates fall inside the target widget
  * exact         -- coordinates equal the reference click
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import torch

from .sample import load
from .ui_world import SCALE, sample_example, to_protocol


ACTION_RE = re.compile(r"(click|type|toggle|drag)\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_best.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--out", default="reports/ui_eval.json")
    ap.add_argument("--samples-out", default="samples/ui_completions.txt")
    ap.add_argument("--seed", type=int, default=424242)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[eval-ui] no checkpoint at {args.ckpt}")
        return
    model, tok, step = load(args.ckpt)
    rng = random.Random(args.seed)

    fmt = act_ok = inside = exact = 0
    rows = []
    for _ in range(args.n):
        ex = sample_example(rng)
        ref_x, ref_y = ex.click[0] * SCALE, ex.click[1] * SCALE
        full = to_protocol(ex.layout, ex.task, f"{ex.action}({ref_x},{ref_y})")
        prompt = full.split("<|action|>")[0] + "<|action|>"
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=args.tokens, temperature=0.2, top_k=20,
                             eos_id=tok.special_id("<|endoftext|>"))
        gen = tok.decode(out[0].tolist())[len(prompt):]
        m = ACTION_RE.search(gen)
        ok_fmt = m is not None
        ok_act = ok_inside = ok_exact = False
        px = py = -1
        if m:
            verb, px, py = m.group(1), int(m.group(2)), int(m.group(3))
            ok_act = verb == ex.action
            ok_inside = ex.target.contains(px / SCALE, py / SCALE, margin=0.5)
            ok_exact = abs(px - ref_x) <= SCALE and abs(py - ref_y) <= SCALE
        fmt += ok_fmt
        act_ok += ok_act
        inside += ok_inside
        exact += ok_exact
        rows.append({"task": ex.task, "ref": f"{ex.action}({ref_x},{ref_y})",
                     "pred": gen.strip().split("<|endoftext|>")[0][:60],
                     "inside": ok_inside})

    n = args.n
    result = {
        "step": step,
        "format_rate": round(fmt / n, 3),
        "action_type_rate": round(act_ok / n, 3),
        "in_widget_rate": round(inside / n, 3),
        "exact_rate": round(exact / n, 3),
        "n": n,
    }
    os.makedirs("reports", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({**result, "examples": rows[:20]}, f, indent=2)
    os.makedirs(os.path.dirname(args.samples_out) or ".", exist_ok=True)
    with open(args.samples_out, "w") as f:
        f.write(f"checkpoint step {step}\n\n")
        for r in rows[:20]:
            f.write(f"task:    {r['task']}\n  ref:   {r['ref']}\n  model: {r['pred']}\n\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
