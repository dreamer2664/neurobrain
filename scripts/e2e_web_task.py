#!/usr/bin/env python3
"""End to end: the LM runs the research-and-report task on the mock desktop.

    python3 scripts/e2e_web_task.py --ckpt checkpoints/lm_latest.pt --n 40

A task counts as solved only when the Google Docs document contains the answer
that was on the article page *and* it was saved. The runtime does the
verification; nothing the model says about itself is counted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brainbox.host import BrainHost  # noqa: E402
from brainbox.lm_drive import LMDriver  # noqa: E402
from neurobrain.web_world import WebTask, WebWorld  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lm_latest.pt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--out", default="reports/web_eval.json")
    ap.add_argument("--dump", default="reports/web_eval_transcripts.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    host = BrainHost(checkpoint=args.ckpt)
    host.boot()
    print(f"[web] checkpoint step {getattr(host.info, 'step', '?')}")
    driver = LMDriver(host, max_steps=args.max_steps, max_tokens=32, seed=13, temperature=0.0, top_k=1)

    rows, transcripts = [], []
    for i in range(args.n):
        task = WebTask(rng)
        world = WebWorld(task)
        row = driver.run(world, task.objective(), feasible=True)
        ok = world.saved and world.doc_text == task.answer
        useful = sum(1 for e in world.log if e["useful"])
        rows.append({"topic": task.topic, "answer": task.answer, "solved": ok,
                     "doc": world.doc_text, "saved": world.saved,
                     "steps": row["executed_steps"], "overclaim": row["overclaim"],
                     "actions": len(world.log), "useful_actions": useful})
        transcripts.append(row)
        print(f"[web] {i+1:>2}/{args.n}: {'OK  ' if ok else 'MISS'} "
              f"topic={task.topic[:28]:<28} doc={world.doc_text[:24]!r} "
              f"steps={row['executed_steps']}", flush=True)

    solved = sum(r["solved"] for r in rows)
    result = {
        "step": getattr(host.info, "step", None),
        "tasks": args.n,
        "solved": solved,
        "accuracy": round(solved / max(1, args.n), 3),
        "mean_steps": round(sum(r["steps"] for r in rows) / max(1, args.n), 2),
        "action_precision": round(sum(r["useful_actions"] for r in rows)
                                  / max(1, sum(r["actions"] for r in rows)), 3),
        "overclaim": round(sum(r["overclaim"] for r in rows) / max(1, args.n), 3),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.dump, "w") as f:
        json.dump(transcripts, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
