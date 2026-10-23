#!/usr/bin/env python3
"""One command, both bars, nothing hidden.

    python3 scripts/demo_brain.py            # 10 questions + 3 desktop tasks
    python3 scripts/demo_brain.py --n 50 --tasks 10

This is the morning demo. It boots the checkpoint, asks it basic questions, then
puts it in the mock desktop and makes it do the research-and-report job. Every
desktop action is executed and verified by the runtime, and the transcript is
printed exactly as it was produced -- no tidying, including the retries.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brainbox.host import BrainHost  # noqa: E402
from brainbox.lm_drive import LMDriver  # noqa: E402
from neurobrain.kb_basic import tables  # noqa: E402
from neurobrain.web_world import WebTask, WebWorld  # noqa: E402


def ask(host, topics, questions) -> tuple[int, int]:
    ok = 0
    for q, want in questions:
        out = host.generate(q, max_tokens=24, temperature=0.0)
        got = (out.get("completion") or "").strip().split("<|")[0].strip()
        good = got.rstrip(".").lower() == want.rstrip(".").lower()
        ok += good
        print(f"  {'ok ' if good else 'MISS'}  {q}\n        -> {got!r}"
              + ("" if good else f"   (want {want!r})"))
    return ok, len(questions)


def build_questions(n: int, seed: int = 11) -> list[tuple[str, str]]:
    t = tables()
    pool = []
    for country, capital in t["capitals"].items():
        pool.append((f"What is the capital of {country}?", capital))
    for country, cur in t["currencies"].items():
        pool.append((f"What currency is used in {country}?", cur))
    for el, sym in t["elements"].items():
        pool.append((f"What is the chemical symbol for {el}?", sym))
    for country, cont in t["continents"].items():
        pool.append((f"Which continent is {country} on?", cont))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def desktop(host, k: int, seed: int = 5) -> tuple[int, int]:
    rng = random.Random(seed)
    host_ok = 0
    for i in range(k):
        task = WebTask(rng)
        world = WebWorld(task)
        obj = task.objective()
        print(f"\n  task {i + 1}: {obj}")
        res = LMDriver(host, max_steps=14).run(world, obj)
        solved = world.saved and world.doc_text == task.answer
        host_ok += solved
        for line in res["transcript"]:
            print("      " + line.strip()[:220].replace("<|scan|>", "\n        scan: "))
        print(f"    -> {'SOLVED' if solved else 'MISS'}  document={world.doc_text!r} "
              f"saved={world.saved} steps={res['executed_steps']}")
    return host_ok, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_latest.pt")
    ap.add_argument("--n", type=int, default=10, help="how many questions")
    ap.add_argument("--tasks", type=int, default=3, help="how many desktop tasks")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    host = BrainHost(checkpoint=args.ckpt, threads=args.threads)
    host.boot()
    print(f"brain: {getattr(host.info, 'step', '?')} steps, "
          f"{os.path.getsize(args.ckpt) / 1e6:.1f} MB on disk\n")

    print(f"== questions ({args.n}) ==")
    q_ok, q_n = ask(host, None, build_questions(args.n))

    print(f"\n== desktop: research in chrome, report in docs ({args.tasks}) ==")
    d_ok, d_n = desktop(host, args.tasks)

    print(f"\nquestions {q_ok}/{q_n}  ({q_ok / max(1, q_n):.0%})")
    print(f"desktop   {d_ok}/{d_n}  ({d_ok / max(1, d_n):.0%})")


if __name__ == "__main__":
    main()
