"""Expert iteration: bias the model toward work that actually completes.

This is the honest version of "train it to never stop". A model cannot be
argued into persistence; it can be *selected* for it. The loop:

  1. ask the current model for N continuations per task (temperature > 1)
  2. execute every candidate: parse it, run it, score it
  3. keep only the candidates that pass (plus one failing attempt per episode,
     so the model still learns what failure looks like)
  4. fine-tune on the kept set, with a low LR, for a few hundred steps
  5. re-measure pass@1; keep the new weights only if the number went up

Same procedure applies to desktop episodes later: a "pass" is the task's
success check in the eval suite. The reward is always executable.

    python -m neurobrain.expert_iter --steps 400 --rounds 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap

import torch

from .model import GPT, GPTConfig
from .sample import load
from .tokenizer import BPETokenizer

# ------------------------------------------------------------------ task bank
# Each task: prompt, a hidden test, and the function name under test.
TASKS = [
    ("def add(a, b):\n    \"\"\"return the sum\"\"\"\n", "assert add(2, 3) == 5\nassert add(-1, 1) == 0", "add"),
    ("def mean(xs):\n    \"\"\"return the arithmetic mean of a non-empty list\"\"\"\n",
     "assert mean([2, 4, 6]) == 4\nassert abs(mean([1, 2]) - 1.5) < 1e-9", "mean"),
    ("def unique(xs):\n    \"\"\"return the list with duplicates removed, order preserved\"\"\"\n",
     "assert unique([1, 2, 2, 3, 1]) == [1, 2, 3]", "unique"),
    ("def word_count(s):\n    \"\"\"return how many words are in the string\"\"\"\n",
     "assert word_count('a b c') == 3\nassert word_count('') == 0", "word_count"),
    ("def reverse(s):\n    \"\"\"return the string backwards\"\"\"\n",
     "assert reverse('abc') == 'cba'\nassert reverse('') == ''", "reverse"),
    ("def even_only(xs):\n    \"\"\"return only the even numbers\"\"\"\n",
     "assert even_only([1, 2, 3, 4]) == [2, 4]", "even_only"),
    ("def max_of(xs):\n    \"\"\"return the largest element\"\"\"\n",
     "assert max_of([3, 9, 2]) == 9", "max_of"),
    ("def count_char(s, ch):\n    \"\"\"count occurrences of ch in s\"\"\"\n",
     "assert count_char('banana', 'a') == 3", "count_char"),
    ("def factorial(n):\n    \"\"\"return n! for n >= 0\"\"\"\n",
     "assert factorial(0) == 1\nassert factorial(5) == 120", "factorial"),
    ("def is_palindrome(s):\n    \"\"\"true if s reads the same both ways\"\"\"\n",
     "assert is_palindrome('racecar')\nassert not is_palindrome('abc')", "is_palindrome"),
    ("def flatten(nested):\n    \"\"\"flatten one level of nested lists\"\"\"\n",
     "assert flatten([[1, 2], [3]]) == [1, 2, 3]", "flatten"),
    ("def fizz(n):\n    \"\"\"return 'fizz' if n divisible by 3, else str(n)\"\"\"\n",
     "assert fizz(9) == 'fizz'\nassert fizz(4) == '4'", "fizz"),
]


def bench_tasks() -> list[tuple[str, str, str]]:
    """The codebench v1 bank (49 tasks) as (prompt, tests, fn) triples.

    Expert iteration should select on the same distribution codebench measures;
    the builtin 12-task list above stays as the quick smoke bank.
    """
    from .codebench import load_tasks
    return [(t["prompt"], t["tests"], t["fn"]) for t in load_tasks()]


def run_candidate(prompt: str, completion: str, test: str, fn: str) -> tuple[bool, str]:
    """Execute a candidate in a subprocess: syntax, then the hidden test."""
    body = prompt + completion
    try:
        ast.parse(body)
    except SyntaxError as e:
        return False, f"syntax: {e.msg}"
    source = textwrap.dedent(body) + "\n\n" + textwrap.dedent(test) + f"\nprint('OK', {fn})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=3)
        ok = proc.returncode == 0 and "OK" in proc.stdout
        return ok, (proc.stderr.strip().splitlines() or [""])[-1][:160]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def parses_and_defines(prompt: str, completion: str, fn: str) -> bool:
    """Tier-B reward: prompt+completion is real Python and defines the right name."""
    try:
        tree = ast.parse(prompt + completion)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn
               for n in tree.body)


def sample_completions(model, tok, prompt: str, n: int, max_tokens: int = 96) -> list[str]:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = []
    for _ in range(n):
        res = model.generate(ids, max_new_tokens=max_tokens, temperature=1.0, top_k=40)
        text = tok.decode(res[0].tolist())[len(prompt):]
        text = text.split("<|endoftext|>")[0]
        if "\ndef " in text or "\nclass " in text:      # stop at the next definition
            text = text.split("\ndef ")[0].split("\nclass ")[0].rstrip() + "\n"
        if not text.endswith("\n"):
            text += "\n"
        out.append(text)
    return out


def evaluate(model, tok, n_per_task: int = 4, tasks: list | None = None) -> dict:
    tasks = TASKS if tasks is None else tasks
    passed = total = 0
    detail = []
    for prompt, test, fn in tasks:
        p = 0
        for comp in sample_completions(model, tok, prompt, n_per_task, max_tokens=80):
            ok, why = run_candidate(prompt, comp, test, fn)
            p += ok
            total += 1
            if not ok:
                detail.append({"fn": fn, "why": why})
        passed += p
    return {"pass_at_1": round(passed / total, 3), "n": total, "failures": detail[:8]}


def finetune(model, tok, winners: list[str], steps: int, lr: float = 2e-4, ctx: int = 256):
    """SFT on the retained episodes (successes + the failing attempt that preceded them)."""
    cfg = model.cfg
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = random.Random(0)
    model.train()
    for step in range(steps):
        batch = [rng.choice(winners) for _ in range(4)]
        xs, ys = [], []
        for text in batch:
            ids = tok.encode(text)[: ctx + 1]
            if len(ids) < 8:
                ids = ids + [0] * (8 - len(ids))
            ids = ids + [tok.special_id("<|pad|>")] * (ctx + 1 - len(ids))
            xs.append(ids[:-1])
            ys.append([t if t != tok.special_id("<|pad|>") else -1 for t in ids[1:]])
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"[ei] sft step {step+1}/{steps} loss {loss.item():.4f}", flush=True)
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_best.pt")
    ap.add_argument("--out", default="checkpoints/lm_ei.pt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=12, help="candidates per task per round")
    ap.add_argument("--steps", type=int, default=200, help="fine-tune steps per round")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bench", default="codebench", choices=["codebench", "builtin"],
                    help="task bank: codebench v1 (49) or the builtin smoke list (12)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[ei] no checkpoint at {args.ckpt}")
        return
    model, tok, step = load(args.ckpt)
    torch.set_num_threads(2)
    tasks = bench_tasks() if args.bench == "codebench" else TASKS
    print(f"[ei] bench={args.bench}: {len(tasks)} tasks")

    base = evaluate(model, tok, tasks=tasks)
    print("[ei] baseline:", json.dumps(base))
    history = [{"round": 0, **base}]

    for rnd in range(1, args.rounds + 1):
        winners: list[str] = []
        soft: list[str] = []
        kept = tried = 0
        for prompt, test, fn in tasks:
            comps = sample_completions(model, tok, prompt, args.samples)
            first_failure = None
            for comp in comps:
                ok, _ = run_candidate(prompt, comp, test, fn)
                tried += 1
                if ok:
                    winners.append(prompt + comp + "<|endoftext|>")
                    winners.append(prompt + comp + "<|endoftext|>")
                    kept += 1
                elif parses_and_defines(prompt, comp, fn):
                    # tier B: valid Python, wrong logic. Weighted below real
                    # winners, and paired with the retry tag so the episode
                    # still teaches "fail -> look again -> keep going".
                    soft.append(prompt + comp + "<|retry|>test failed, patch the logic\n<|endoftext|>")
                elif first_failure is None:
                    first_failure = comp
            # keep one failure per task so the model still sees recovery shape
            if first_failure:
                winners.append(prompt + first_failure + "\n<|retry|>patch the logic and rerun\n"
                               + "<|endoftext|>")
        print(f"[ei] round {rnd}: {kept} pass / {len(soft)} parse-only / {tried} tried")
        if kept == 0 and not soft:
            print("[ei] nothing retainable; the base model cannot yet write this task class.")
            break
        # a passing episode is worth four parse-only ones
        train_set = winners * 4 + soft
        finetune(model, tok, train_set or winners, args.steps, args.lr)
        metrics = evaluate(model, tok)
        print(f"[ei] round {rnd}:", json.dumps(metrics))
        history.append({"round": rnd, "kept": kept, "tried": tried, **metrics})
        torch.save({"model": model.state_dict(), "step": step, "cfg": model.cfg.to_dict()},
                   args.out)

    os.makedirs("reports", exist_ok=True)
    with open("reports/expert_iter.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[ei] wrote reports/expert_iter.json")


if __name__ == "__main__":
    main()
