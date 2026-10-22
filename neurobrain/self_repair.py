"""Self-repair (deliverable 5): generate -> execute -> feed back the first
error -> regenerate -> execute.

The repair context mirrors the drive code-episode protocol exactly (see
neurobrain/drive.py), because that is the distribution the model is trained on:

    <|task|>make the tests pass for fn()
    <|try|>1<|code|>{prompt}{attempt-1}
    <|run|>pytest -q<|err|>{first error}
    <|retry|>attempt 1 failed, patch the logic and rerun
    <|try|>2<|code|>{regenerated function}

Attempt 2 is generated from that context; the candidate is whatever follows the
last code marker -- a full def, or a body that we glue onto the prompt.

Headline numbers: repair_rate = fixed-after-error / failed-first, plus three
full transcripts (the deliverable asks for the receipts, including failures).

    python3 -m neurobrain.self_repair --ckpt checkpoints/lm_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .codebench import extract_completion, load_tasks, peak_rss_mb, run_candidate, sha256_file
from .sample import load

# markers built by concatenation on purpose: this sandbox's edit channels have
# been observed mangling full special-token literals in transit (see fleet
# agents/04-coder/NOTES.md, cycle 2).
TM = "<|" + "task" + "|>"
TRY = "<|" + "try" + "|>"
RUN = "<|" + "run" + "|>"
ERR = "<|" + "err" + "|>"
RETRY = "<|" + "retry" + "|>"
DRIVE = "<|" + "dri" + "ve" + "|>"
CTX_BUDGET = 160          # tokens of repair context, leaves ~90 for generation
GEN1_TOKENS = 96
GEN2_TOKENS = 88


def code_marker(tok) -> str:
    return tok.specials[1]


def _mk(word: str) -> str:
    return "<|" + word + "|>"


# drive-protocol markers that terminate a code generation even though they are
# not in codebench.STOP_MARKERS: a repair attempt that starts narrating the
# episode ("try 3", "run pytest") is no longer writing code.
CUTS = tuple(_mk(w) for w in ("try", "run", "err", "retry", "think", "ok",
                              "check", "done", "task", "drive", "ui"))


def clean_gen(text: str) -> str:
    cut = len(text)
    for m in CUTS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def build_repair_ctx(tok, prompt: str, comp1: str, err: str, fn: str,
                     budget: int = CTX_BUDGET) -> list[int]:
    """Token ids of the drive-protocol repair context, trimmed to budget.

    Trimming drops attempt-1 tokens from the FRONT (the earliest mistake is the
    least informative part); head and error feedback are never dropped.
    """
    cm = code_marker(tok)
    head = TM + "make the tests pass for " + fn + "()" + TRY + "1" + cm
    tail = (RUN + "pytest -q" + ERR + err[:80] +
            RETRY + "attempt 1 failed, patch the logic and rerun" + TRY + "2" + cm)
    head_ids = tok.encode(head)
    tail_ids = tok.encode(tail)
    body_ids = tok.encode(prompt + comp1)
    room = budget - len(head_ids) - len(tail_ids)
    if room < 8:                      # pathological: keep the very end of attempt 1
        body_ids = body_ids[-max(0, room):]
    elif len(body_ids) > room:
        body_ids = body_ids[-room:]
    return head_ids + body_ids + tail_ids


def assemble(prompt: str, gen: str) -> str:
    """Candidate source from a repair generation: whole def, or body on prompt."""
    g = extract_completion(clean_gen(gen), "")
    if g.lstrip().startswith("def "):
        return g if g.endswith(chr(10)) else g + chr(10)
    return prompt + g


def repair_task(model, tok, task: dict, temperature: float = 0.8, top_k: int = 40,
                max_ctx: int | None = None) -> dict:
    eos = tok.special_id(tok.specials[0])
    max_ctx = max_ctx or model.cfg.ctx
    prompt, tests, fn = task["prompt"], task["tests"], task["fn"]
    timeout_s = task.get("timeout_s", 5)

    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out1 = model.generate(ids, max_new_tokens=GEN1_TOKENS, temperature=temperature,
                          top_k=top_k, eos_id=eos)
    full1 = clean_gen(tok.decode(out1[0].tolist()))
    comp1 = extract_completion(full1, prompt)
    r1 = run_candidate(prompt, comp1, tests, fn, timeout_s)

    transcript = {"id": task["id"], "attempt1": comp1[:300], "passed1": r1["passed"],
                  "error": r1["why"], "attempt2": "", "passed2": False, "why2": ""}
    if r1["passed"]:
        return transcript

    err = r1["why"] or "test failed"
    ctx_ids = build_repair_ctx(tok, prompt, comp1, err, fn)
    ctx = torch.tensor([ctx_ids], dtype=torch.long)[:, -max_ctx:]
    out2 = model.generate(ctx, max_new_tokens=GEN2_TOKENS, temperature=temperature,
                          top_k=top_k, eos_id=eos)
    gen2 = tok.decode(out2[0].tolist()[ctx.shape[1]:])
    cand = assemble(prompt, gen2)
    r2 = run_candidate("", cand, tests, fn, timeout_s)
    transcript.update({"attempt2": cand[:400], "passed2": r2["passed"], "why2": r2["why"]})
    return transcript


def main() -> None:
    ap = argparse.ArgumentParser(description="self-repair on codebench")
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "lm_best.pt"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("reports", "self_repair.json"))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model, tok, step = load(args.ckpt, tok_path=os.path.join(args.data, "tokenizer.json"))
    tasks = load_tasks()
    if args.limit:
        tasks = tasks[:args.limit]

    t0 = time.time()
    rows = [repair_task(model, tok, t, args.temperature, args.top_k) for t in tasks]
    dt = time.time() - t0

    n = len(rows)
    first_pass = sum(r["passed1"] for r in rows)
    failed_first = [r for r in rows if not r["passed1"]]
    repaired = sum(r["passed2"] for r in failed_first)
    result = {
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "ckpt": args.ckpt, "ckpt_sha256": sha256_file(args.ckpt), "step": step,
        "decode": {"temperature": args.temperature, "top_k": args.top_k, "seed": args.seed},
        "n_tasks": n,
        "first_pass": first_pass,
        "failed_first": len(failed_first),
        "repaired": repaired,
        "repair_rate": round(repaired / len(failed_first), 4) if failed_first else None,
        "final_pass": round((first_pass + repaired) / n, 4),
        "seconds": round(dt, 1),
        "peak_rss_mb": peak_rss_mb(),
        "transcripts": failed_first[:3] if failed_first else rows[:3],
        "all_rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("transcripts", "all_rows")},
                     indent=1))
    print("[repair] wrote", args.out)


if __name__ == "__main__":
    main()
