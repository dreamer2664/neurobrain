#!/usr/bin/env python3
"""The mock desktop, driven by an oracle instead of the model.

    python3 scripts/web_oracle.py --n 50

Purpose: prove the environment and its verifier independently of the policy.
If the oracle cannot get 100% here, the world is broken and no model score on it
means anything. The oracle picks elements by label from the same scan the model
is given -- it knows the answer, the model has to find it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brainbox.lm_drive import scan_from_obs  # noqa: E402
from neurobrain.web_world import WebTask, WebWorld  # noqa: E402


def click_label(world, label: str) -> dict:
    el = next((e for e in world.view().elements if e.label == label), None)
    if el is None:
        return {"ok": False, "detail": f"{label} is not on screen"}
    return world.act({"type": "click"}, (el.cx, el.cy))


def run_one(task: WebTask) -> dict:
    world = WebWorld(task)
    steps = 0

    def do(label):
        nonlocal steps
        steps += 1
        return click_label(world, label)

    do("chrome")
    do(task.topic)
    do("copy")
    do("docs")
    do("paste")
    do("save")
    return {"solved": world.saved and world.doc_text == task.answer,
            "doc": world.doc_text, "answer": task.answer, "steps": steps,
            "last_scan": scan_from_obs(world.observe())[:120]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="reports/web_oracle.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [run_one(WebTask(rng)) for _ in range(args.n)]
    solved = sum(r["solved"] for r in rows)
    result = {"n": args.n, "solved": solved, "accuracy": round(solved / args.n, 3),
              "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for r in rows[:3]:
        print(r["answer"], "->", r["doc"], "|", r["solved"])


if __name__ == "__main__":
    main()
