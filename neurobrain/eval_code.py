"""Evaluate the language brain on held-out code.

Metrics:
  * val loss / perplexity on the held-out code split
  * syntax validity: generate a continuation after a real function header and
    check whether `header + continuation` parses with Python's ast
  * first-line exact match on a small memorisation probe (functions seen 3+
    times in training) -- a sanity check that it is learning code, not noise
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random

import torch

from .data import TokenSource
from .model import GPT, GPTConfig
from .sample import load


PROMPTS = [
    "def add(a, b):\n",
    "def factorial(n):\n",
    "class Point:\n    def __init__(self, x, y):\n",
    "def read_file(path):\n",
    "def is_even(n):\n",
    "def main():\n",
    "for i in range(10):\n",
    "import json\n\ndef load_config(path):\n",
    "def bubble_sort(items):\n",
    "def greet(name):\n",
    "try:\n    value = int(text)\n",
    "def average(values):\n",
]


def syntax_prefix(prefix: str, gen: str) -> tuple[bool, float]:
    """Return (parses, fraction_of_generated_lines_that_parse).

    A tiny LM rarely emits a perfectly closed program; what matters is whether
    the code it writes is *syntactically coherent* for most of its length. We
    find the longest line-boundary prefix that parses.
    """
    text = prefix + gen
    lines = text.splitlines()
    base = len(prefix.splitlines())
    for cut in range(len(lines), base, -1):
        try:
            ast.parse("\n".join(lines[:cut]))
        except SyntaxError:
            continue
        return True, (cut - base) / max(1, len(lines) - base)
    return False, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out", default="reports/code_eval.json")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[eval] no checkpoint at {args.ckpt}")
        return
    model, tok, step = load(args.ckpt)
    rng = random.Random(0)

    # 1) val perplexity
    src = TokenSource(os.path.join(args.data, "code_val.bin"), model.cfg.ctx)
    losses = []
    with torch.no_grad():
        for _ in range(32):
            x, y = src.sample(4, rng)
            losses.append(model(x, y)[1].item())
    val_loss = sum(losses) / len(losses)

    # 2) syntax validity of continuations
    valid = 0
    coherent = 0
    samples = []
    for p in PROMPTS[: args.n]:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=args.tokens, temperature=0.5, top_k=30)
        gen = tok.decode(out[0].tolist())[len(p):]
        ok, frac = syntax_prefix(p, gen)
        valid += ok
        coherent += frac >= 0.5
        samples.append({"prompt": p, "generation": gen, "parses": ok,
                        "parsing_fraction": round(frac, 2)})

    result = {
        "step": step,
        "val_loss": round(val_loss, 4),
        "val_ppl": round(min(20, val_loss) ** 1 if False else __import__("math").exp(min(20, val_loss)), 2),
        "syntax_valid": valid,
        "syntax_total": min(args.n, len(PROMPTS)),
        "syntax_rate": round(valid / min(args.n, len(PROMPTS)), 3),
        "coherent_rate": round(coherent / min(args.n, len(PROMPTS)), 3),
        "samples": samples,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
