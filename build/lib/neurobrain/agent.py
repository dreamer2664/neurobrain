"""A tiny desktop agent built from the two brains.

GripNet looks at the pixels and points. The language brain reads the element
tree as text and speaks the action. Neither model is pretrained; both are
trained by the scripts in this repo.

    python -m neurobrain.agent --n 6 --out samples/agent_demo.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from .gripnet import GripNet
from .sample import load as load_lm
from .ui_world import CANVAS, CHAR2ID, SCALE, render, sample_example, to_protocol

ACTION_RE = re.compile(r"(click|type|toggle|drag)\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def grip_click(model: GripNet, layout, task: str):
    img = render(layout)
    x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    t = re.sub(r"[^a-z0-9 -]", "", task.lower())[:48]
    ids = [CHAR2ID.get(c, 0) for c in t] + [0] * (48 - len(t))
    with torch.no_grad():
        scores, offsets, _ = model(x, torch.tensor([ids]))
        px, py = model.click_xy(scores, offsets, CANVAS)[0].tolist()
    return px, py


def lm_action(model, tok, layout, task: str, tokens: int = 16):
    full = to_protocol(layout, task, "click(0,0)")
    prompt = full.split("<|action|>")[0] + "<|action|>"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=tokens, temperature=0.2, top_k=20,
                         eos_id=tok.special_id("<|endoftext|>"))
    gen = tok.decode(out[0].tolist())[len(prompt):]
    m = ACTION_RE.search(gen)
    if not m:
        return None, gen.strip()
    return (m.group(1), int(m.group(2)), int(m.group(3))), gen.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", default="checkpoints/grip_latest.pt")
    ap.add_argument("--lm", default="checkpoints/lm_best.pt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="samples/agent_demo.png")
    ap.add_argument("--log", default="samples/agent_demo.txt")
    args = ap.parse_args()

    grip = GripNet()
    ck = torch.load(args.grip, map_location="cpu", weights_only=False)
    grip.load_state_dict(ck["model"])
    grip.eval()
    lm, tok, lm_step = load_lm(args.lm) if os.path.exists(args.lm) else (None, None, -1)

    rng = random.Random(args.seed)
    rows, tiles = [], []
    grip_hits = lm_hits = 0
    for _ in range(args.n):
        ex = sample_example(rng)
        gx, gy = grip_click(grip, ex.layout, ex.task)
        grip_hit = ex.target.contains(gx, gy)
        grip_hits += grip_hit
        entry = {"task": ex.task, "target": [ex.target.x, ex.target.y, ex.target.w, ex.target.h],
                 "gripnet_px": [round(gx, 1), round(gy, 1)], "gripnet_hit": bool(grip_hit)}
        if lm is not None:
            act, raw = lm_action(lm, tok, ex.layout, ex.task)
            if act:
                verb, ax, ay = act
                lm_hit = ex.target.contains(ax / SCALE, ay / SCALE, margin=0.5)
                entry.update({"lm_action": f"{verb}({ax},{ay})", "lm_hit": bool(lm_hit)})
                lm_hits += lm_hit
            else:
                entry.update({"lm_action": raw[:40], "lm_hit": False})
        rows.append(entry)

        tile = Image.fromarray(render(ex.layout)).resize((256, 256), Image.NEAREST)
        d = ImageDraw.Draw(tile)
        s = 2
        d.rectangle([ex.target.x * s, ex.target.y * s, (ex.target.x + ex.target.w) * s,
                     (ex.target.y + ex.target.h) * s], outline=(0, 220, 80), width=2)
        d.ellipse([gx * s - 6, gy * s - 6, gx * s + 6, gy * s + 6], outline=(255, 60, 60), width=2)
        if entry.get("lm_action") and lm is not None and "(" in entry["lm_action"]:
            m = ACTION_RE.search(entry["lm_action"])
            if m:
                lx, ly = int(m.group(2)) / SCALE * s, int(m.group(3)) / SCALE * s
                d.line([lx - 6, ly - 6, lx + 6, ly + 6], fill=(90, 170, 255), width=2)
                d.line([lx - 6, ly + 6, lx + 6, ly - 6], fill=(90, 170, 255), width=2)
        d.rectangle([0, 232, 256, 256], fill=(0, 0, 0))
        d.text((4, 234), ex.task[:27], fill=(255, 255, 255))
        tiles.append(tile)

    cols = 3
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows_n * 256), (18, 18, 22))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    with open(args.log, "w") as f:
        f.write(f"language brain step {lm_step}\n")
        f.write(f"gripnet: {grip_hits}/{args.n} clicks on target\n")
        if lm is not None:
            f.write(f"language brain: {lm_hits}/{args.n} clicks on target\n\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[agent] grip {grip_hits}/{args.n}  lm {lm_hits}/{args.n}  -> {args.out}")


if __name__ == "__main__":
    main()
