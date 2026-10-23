#!/usr/bin/env python3
"""Turn the knowledge base into question/answer training text.

Format (plain text, no new special tokens, so every existing checkpoint stays
compatible):

    Q: What is the capital of France?
    A: Paris

Each fact is asked several ways and the template used rotates, so the model
learns to answer questions rather than to complete one fixed string. Answers
are short on purpose: a 4.25M model can be reliable on short answers and will
never be reliable on paragraphs.

Splits: 90% of facts train, 10% are held out entirely. The held-out set is what
`scripts/qa_eval.py --split heldout` measures, and it is why the headline
accuracy is not just memorisation of a fixed pair list.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from array import array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.kb_basic import tables  # noqa: E402
from neurobrain.tokenizer import BPETokenizer  # noqa: E402

# The phrasing space is deliberately dense: a 4.25M model does not extrapolate
# from two templates to a third, so every question it could be asked about a
# fact is present with several wordings. What is held out is *facts*, not
# syntax -- the held-out split in qa_eval measures recall of unseen facts.
Q_TPL = [
    "What is the capital of {s}?",
    "Which city is the capital of {s}?",
    "Name the capital of {s}.",
    "Can you tell me the capital of {s}?",
    "The capital of {s} is what?",
    "What is {s}'s capital city?",
    "Which city serves as the capital of {s}?",
    "Do you know the capital of {s}?",
    "Capital of {s}?",
    "What city is the capital of {s}?",
]
CUR_TPL = [
    "What is the currency of {s}?",
    "Which currency is used in {s}?",
    "Currency used in {s}?",
    "What money is used in {s}?",
    "What currency does {s} use?",
]
CONT_TPL = [
    "Which continent is {s} in?",
    "On which continent is {s}?",
    "Which continent does {s} belong to?",
    "What is the continent of {s}?",
    "Where is {s} located by continent?",
]
LANG_TPL = [
    "What language is spoken in {s}?",
    "Which language do people speak in {s}?",
    "Which language is spoken by people in {s}?",
    "The language of {s} is what?",
    "What language do they speak in {s}?",
]
ELEM_SYM = ["What is the chemical symbol for {s}?", "Which symbol stands for {s}?",
            "Chemical symbol of {s}?", "Which chemical symbol is used for {s}?",
            "What is the symbol of the element {s}?"]
ELEM_NUM = ["What is the atomic number of {s}?", "Atomic number of {s}?",
            "How many protons does {s} have?"]
PLANET_TPL = ["What is {s}?", "Tell me about {s}.", "Describe {s} in one line."]

# simple paraphrases so the answer text is not one frozen string
def _answer_variants(ans: str) -> list[str]:
    variants = [ans]
    if not ans.endswith("."):
        variants.append(ans + ".")
    if ans and ans[0].isupper():
        variants.append(ans[0].lower() + ans[1:])
    return variants


def facts() -> list[tuple[str, str, str]]:
    """(question, answer, fact_id) - fact_id is what the held-out split keys on."""
    t = tables()
    out: list[tuple[str, str, str]] = []
    for country, city in t["capitals"].items():
        country = country.strip()
        for q in Q_TPL:
            out.append((q.format(s=country), city, f"cap:{country}"))
    for country, cur in t["currencies"].items():
        for q in CUR_TPL:
            out.append((q.format(s=country), cur, f"cur:{country}"))
    for country, cont in t["continents"].items():
        for q in CONT_TPL:
            out.append((q.format(s=country), cont, f"cont:{country}"))
    for country, lang in t["languages"].items():
        for q in LANG_TPL:
            out.append((q.format(s=country), lang, f"lang:{country}"))
    for name, (symbol, number) in t["elements"].items():
        for q in ELEM_SYM:
            out.append((q.format(s=name), symbol, f"elem:{name}"))
        for q in ELEM_NUM:
            out.append((q.format(s=name), str(number), f"enum:{name}"))
    for name, desc in t["planets"]:
        # one short factual question per planet instead of a "describe X" prompt
        out.append((f"Which planet is {name}?", desc, f"planet:{name}"))
        out.append((f"What do we know about {name}?", desc, f"planet:{name}"))
    for q, a in t["misc"]:
        out.append((q, a, f"misc:{abs(hash(q)) % 10**8}"))
    for q, a in t["code"]:
        out.append((q, a, f"code:{abs(hash(q)) % 10**8}"))
    return out


def split_facts(all_facts: list[tuple[str, str, str]], holdout: float, seed: int):
    ids = sorted({f[2] for f in all_facts})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_hold = max(1, int(len(ids) * holdout))
    held = set(ids[:n_hold])
    train = [f for f in all_facts if f[2] not in held]
    test = [f for f in all_facts if f[2] in held]
    return train, test, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--repeat", type=int, default=6, help="epochs over train facts")
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(args.data, "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]
    all_facts = facts()
    train, test, held = split_facts(all_facts, args.holdout, args.seed)
    print(f"[qa] {len(all_facts)} pairs from {len({f[2] for f in all_facts})} facts; "
          f"train {len(train)} / held-out {len(test)} ({len(held)} facts)")

    rng = random.Random(args.seed + 1)
    rows = []
    for _ in range(args.repeat):
        block = list(train)
        rng.shuffle(block)
        rows.extend(block)

    def write(name: str, pairs: list[tuple[str, str, str]]) -> int:
        buf = array("H")
        total = 0
        path = os.path.join(args.data, f"{name}.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for q, a, _ in pairs:
                for variant in _answer_variants(a):
                    text = f"Q: {q}\nA: {variant}\n"
                    ids = tok.encode(text) + [eot]
                    buf.extend(ids)
                    total += len(ids)
                if len(buf) > 4_000_000:
                    f.write(buf.tobytes())
                    buf = array("H")
            if buf:
                f.write(buf.tobytes())
        os.replace(tmp, path)
        return total

    n_train = write("qa_train", rows)
    n_val = write("qa_val", test)
    meta = {
        "train_pairs": len(rows), "train_tokens": n_train,
        "heldout_pairs": len(test), "heldout_tokens": n_val,
        "heldout_facts": sorted(held),
        "facts_total": len({f[2] for f in all_facts}),
    }
    with open(os.path.join(ROOT, "reports", "qa_corpus.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[qa] qa_train {n_train/1e6:.2f}M tokens, qa_val {n_val/1e6:.2f}M tokens")
    print(f"[qa] held-out facts: {len(held)} (see reports/qa_corpus.json)")


if __name__ == "__main__":
    main()
