#!/usr/bin/env python3
"""Ask the model 50 basic questions and mark them, honestly.

    python3 scripts/qa_eval.py --ckpt checkpoints/lm_latest.pt --n 50

Two sets, reported separately:

  seen      facts that were in training, asked with templates held out of training.
            This is recall of the knowledge base.
  heldout   facts that never appeared in training at all. This is generalisation,
            and a 4.25M model is not expected to do well here -- it is reported
            so nobody mistakes the first number for the second.

Matching: the model's answer is normalised (case, punctuation, articles) and the
expected answer must appear in it. Nothing fuzzy, nothing scored by a judge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.kb_basic import tables  # noqa: E402
from neurobrain.sample import load  # noqa: E402

# Templates that do NOT appear in build_qa_corpus.py: the question wording is
# unseen even when the fact is seen.
EVAL_Q = [
    "Can you tell me the capital of {s}?",
    "The capital of {s} is what?",
    "What is {s}'s capital city?",
    "Which city serves as the capital of {s}?",
    "Currency used in {s}?",
    "What money is used in {s}?",
    "Which continent does {s} belong to?",
    "What is the continent of {s}?",
    "Which language is spoken by people in {s}?",
    "The language of {s} is what?",
    "Chemical symbol of {s}?",
    "Which chemical symbol is used for {s}?",
    "Atomic number of {s}?",
]

ANSWER_ALT = {  # accepted alternatives, because these are all true
    "Washington, D.C.": ["washington", "washington dc", "washington d.c"],
    "Kuala Lumpur": ["kuala lumpur"],
    "Sucre": ["sucre"],
}


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ,.'-]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def correct(expected: str, got: str) -> bool:
    e, g = norm(expected), norm(got)
    if not g:
        return False
    cands = [e] + [norm(a) for a in ANSWER_ALT.get(expected, [])]
    return any(c and (c in g or g == c) for c in cands)


def build_questions(n: int, seed: int = 2024):
    """A fixed, mixed set of basic questions."""
    t = tables()
    rng = random.Random(seed)
    pool = []
    for country, city in t["capitals"].items():
        pool.append(("cap", country, city, EVAL_Q[seed % 4]))
    for country, cur in t["currencies"].items():
        pool.append(("cur", country, cur, EVAL_Q[4 + seed % 2]))
    for country, cont in t["continents"].items():
        pool.append(("cont", country, cont, EVAL_Q[6 + seed % 2]))
    for country, lang in t["languages"].items():
        pool.append(("lang", country, lang, EVAL_Q[8 + seed % 2]))
    for name, (sym, num) in t["elements"].items():
        pool.append(("elem", name, sym, EVAL_Q[10 + seed % 2]))
        pool.append(("enum", name, str(num), EVAL_Q[12]))
    for q, a in t["misc"]:
        pool.append(("misc", q, a, "{s}"))
    for q, a in t["code"]:
        pool.append(("code", q, a, "{s}"))

    rng2 = random.Random(seed)
    rng2.shuffle(pool)
    # round-robin over kinds so one table cannot dominate 50 questions
    by_kind: dict[str, list] = {}
    for row in pool:
        by_kind.setdefault(row[0], []).append(row)
    out, i = [], 0
    kinds = sorted(by_kind)
    while len(out) < n:
        k = kinds[i % len(kinds)]
        if by_kind[k]:
            out.append(by_kind[k].pop(0))
        i += 1
    return out


def ask(model, tok, question: str, max_tokens: int = 24) -> str:
    import torch

    prompt = f"Q: {question}\nA:"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.0, top_k=1,
                         eos_id=tok.special_id("<|endoftext|>"))
    text = tok.decode(out[0].tolist())[len(prompt):]
    text = text.split("<|endoftext|>")[0].split("\n")[0]
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_latest.pt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="reports/qa_eval.json")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[qa-eval] no checkpoint at {args.ckpt}")
        return
    model, tok, step = load(args.ckpt)

    held = set()
    meta_path = os.path.join(ROOT, "reports", "qa_corpus.json")
    if os.path.exists(meta_path):
        held = set(json.load(open(meta_path)).get("heldout_facts", []))

    rows = build_questions(args.n, args.seed)
    seen_hit = seen_n = held_hit = held_n = 0
    samples = []
    for kind, subject, expected, tpl in rows:
        q = tpl.format(s=subject) if "{s}" in tpl else subject
        got = ask(model, tok, q)
        ok = correct(expected, got)
        fid = f"{kind}:{subject}"
        is_held = any(h.endswith(":" + subject) for h in held)
        if is_held:
            held_n += 1
            held_hit += ok
        else:
            seen_n += 1
            seen_hit += ok
        if len(samples) < 12:
            samples.append({"q": q, "expected": expected, "got": got, "ok": ok,
                            "held_out_fact": is_held})

    result = {
        "step": step,
        "n": len(rows),
        "accuracy": round((seen_hit + held_hit) / max(1, len(rows)), 3),
        "seen_n": seen_n, "seen_accuracy": round(seen_hit / max(1, seen_n), 3),
        "heldout_n": held_n, "heldout_accuracy": round(held_hit / max(1, held_n), 3),
        "samples": samples,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    for s in samples[:8]:
        print(f"  {'OK ' if s['ok'] else 'MISS'} {s['q']}  ->  {s['got']!r} (want {s['expected']!r})")


if __name__ == "__main__":
    main()
