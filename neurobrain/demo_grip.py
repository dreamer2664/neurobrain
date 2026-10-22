"""Render GripNet predictions: target widget in green, predicted click in red."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from .gripnet import GripNet
from .ui_world import CANVAS, render, sample_example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grip_latest.pt")
    ap.add_argument("--out", default="samples/grip_predictions.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    model = GripNet()
    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        step = ck.get("step", 0)
    except FileNotFoundError:
        print(f"[demo] no checkpoint at {args.ckpt}")
        return
    model.eval()

    rng = random.Random(args.seed)
    tiles = []
    hits = 0
    with torch.no_grad():
        for _ in range(args.n):
            ex = sample_example(rng)
            img = render(ex.layout)
            x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
            import re
            from .ui_world import CHAR2ID
            t = re.sub(r"[^a-z0-9 -]", "", ex.task.lower())[:48]
            ids = [CHAR2ID.get(c, 0) for c in t] + [0] * (48 - len(t))
            scores, act_logits = model(x, torch.tensor([ids]))
            px, py = model.click_xy(scores, CANVAS)[0].tolist()
            hit = ex.target.contains(px, py)
            hits += hit
            tile = Image.fromarray(img).resize((256, 256), Image.NEAREST)
            d = ImageDraw.Draw(tile)
            s = 2
            d.rectangle([ex.target.x * s, ex.target.y * s, (ex.target.x + ex.target.w) * s,
                         (ex.target.y + ex.target.h) * s], outline=(0, 220, 80), width=2)
            cx, cy = px * s, py * s
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 40, 40), width=3)
            d.line([cx - 14, cy, cx + 14, cy], fill=(255, 40, 40), width=1)
            d.line([cx, cy - 14, cx, cy + 14], fill=(255, 40, 40), width=1)
            d.rectangle([0, 232, 256, 256], fill=(0, 0, 0))
            d.text((4, 234), f"{ex.task[:26]}", fill=(255, 255, 255))
            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows * 256), (20, 20, 24))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"[demo] step {step}: {hits}/{len(tiles)} correct -> {args.out}")


if __name__ == "__main__":
    main()
