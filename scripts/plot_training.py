#!/usr/bin/env python3
"""Render training curves from reports/*.jsonl into samples/training_curves.png."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    lm = read(os.path.join(ROOT, "reports", "lm_log.jsonl"))
    grip = read(os.path.join(ROOT, "reports", "grip_log.jsonl"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#12141a")
    for ax in axes:
        ax.set_facecolor("#1a1d26")
        ax.tick_params(colors="#c9cdd6")
        for s in ax.spines.values():
            s.set_color("#3a3f4d")
        ax.grid(alpha=0.15)

    steps = [r["step"] for r in lm if "loss" in r]
    loss = [r["loss"] for r in lm if "loss" in r]
    axes[0].plot(steps, loss, color="#5ad2ff", lw=1.4)
    axes[0].set_title("language brain: training loss", color="#e8eaf0")
    axes[0].set_xlabel("step", color="#c9cdd6")
    axes[0].set_ylabel("cross-entropy", color="#c9cdd6")

    for key, color in [("code", "#7ef2a7"), ("ui", "#ffc857")]:
        pts = [(r["step"], r["eval"][key]) for r in lm if "eval" in r and key in r["eval"]]
        if pts:
            xs, ys = zip(*pts)
            axes[1].plot(xs, ys, marker="o", ms=3, color=color, label=key)
    if axes[1].lines:
        axes[1].legend(facecolor="#1a1d26", labelcolor="#c9cdd6", framealpha=0.4)
    axes[1].set_title("language brain: held-out loss", color="#e8eaf0")
    axes[1].set_xlabel("step", color="#c9cdd6")

    pts = [(r["step"], r["eval"]["hit_widget"]) for r in grip if "eval" in r]
    if pts:
        xs, ys = zip(*pts)
        axes[2].plot(xs, ys, marker="o", ms=3, color="#ff7b9c")
    axes[2].axhline(0.15, ls="--", color="#8892a6", lw=1, label="random baseline")
    if axes[2].lines:
        axes[2].legend(facecolor="#1a1d26", labelcolor="#c9cdd6", framealpha=0.4)
    axes[2].set_title("gripnet: click lands on the right widget", color="#e8eaf0")
    axes[2].set_xlabel("step", color="#c9cdd6")
    axes[2].set_ylabel("accuracy", color="#c9cdd6")

    fig.tight_layout()
    out = os.path.join(ROOT, "samples", "training_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
