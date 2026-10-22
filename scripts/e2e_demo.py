#!/usr/bin/env python3
"""End-to-end loop: capture a real screen -> ground with GripNet -> click -> verify.

This is the first honest composition of the pieces: our own renderer draws a
desktop into a real X window, the pixels come back through a real screen grab,
the grounder picks the point, and deskhands presses the button. Nothing is
hand-fed to the model.

Runs headless:
    xvfb-run -a -s "-screen 0 1280x1024x24" python3 scripts/e2e_demo.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageGrab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.gripnet import GripNet  # noqa: E402
from neurobrain.ui_world import CANVAS, CHAR2ID, render, sample_example  # noqa: E402

DISPLAY_PX = 512          # window size on the real screen
SCALE_UP = DISPLAY_PX // CANVAS   # 4


def draw_layout(canvas, layout, scale: int, w: int, h: int) -> None:
    """Render the synthetic desktop as real canvas widgets, not a bitmap."""
    from neurobrain.ui_world import ROLE_COLOR, label_code
    canvas.delete("all")
    light = layout.theme == "light"
    canvas.configure(bg="#ecedf1" if light else "#202228")
    wx, wy, ww, wh = layout.window
    canvas.create_rectangle(wx * scale, wy * scale, (wx + ww) * scale, (wy + wh) * scale,
                            fill="#fafbfd" if light else "#2c2e36", outline="#969aa4")
    canvas.create_rectangle(wx * scale, wy * scale, (wx + ww) * scale, (wy + 10) * scale,
                            fill="#464c60", outline="")
    for el in layout.elements:
        x0, y0 = el.x * scale, el.y * scale
        x1, y1 = (el.x + el.w) * scale, (el.y + el.h) * scale
        c = "#%02x%02x%02x" % ROLE_COLOR[el.role]
        canvas.create_rectangle(x0, y0, x1, y1, fill=c, outline="#60646e")
        code = label_code(el.label)
        cell = 6 * scale // 4
        gx, gy = x0 + 2 * scale, y0 + (el.h * scale - 2 * (cell + 1)) // 2
        for i, bit in enumerate(code):
            r, col = divmod(i, 3)
            bx, by = gx + col * (cell + 1), gy + r * (cell + 1)
            canvas.create_rectangle(bx, by, bx + cell - 1, by + cell - 1,
                                    fill="#fcfcfc" if bit else "#141418", outline="")


def ground(model: GripNet, frame: np.ndarray, task: str) -> tuple[float, float]:
    import re
    small = Image.fromarray(frame).resize((CANVAS, CANVAS), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1)[None]
    t = re.sub(r"[^a-z0-9 -]", "", task.lower())[:48]
    ids = [CHAR2ID.get(c, 0) for c in t] + [0] * (48 - len(t))
    with torch.no_grad():
        scores, offsets, _ = model(x, torch.tensor([ids]))
        px, py = model.click_xy(scores, offsets, CANVAS)[0].tolist()
    return px, py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grip_best.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="reports/e2e_demo.json")
    ap.add_argument("--strip", default="samples/e2e_strip.png")
    args = ap.parse_args()

    import tkinter as tk
    from deskhands.core import DesktopHands

    torch.set_num_threads(2)
    model = GripNet()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    hands = DesktopHands()

    rng = random.Random(args.seed)
    FRAME_PATH = "/tmp/brainbox_frame.png"
    state = {"layout": None}

    def on_click(event):
        """The world reacts: a click inside a widget paints a focus ring.

        That is the verification signal the drive loop needs -- a visible
        change that only happens when the action actually landed.
        """
        layout = state["layout"]
        if layout is None:
            return
        gx, gy = event.x / SCALE_UP, event.y / SCALE_UP
        for el in layout.elements:
            if el.contains(gx, gy):
                canvas.create_rectangle(el.x * SCALE_UP - 2, el.y * SCALE_UP - 2,
                                        (el.x + el.w) * SCALE_UP + 2,
                                        (el.y + el.h) * SCALE_UP + 2,
                                        outline="#00ff66", width=3)
                return

    root = tk.Tk()
    root.geometry(f"{DISPLAY_PX}x{DISPLAY_PX}+0+0")
    root.overrideredirect(True)
    canvas = tk.Canvas(root, width=DISPLAY_PX, height=DISPLAY_PX,
                       highlightthickness=0, bd=0)
    canvas.pack()
    canvas.bind("<Button-1>", on_click)

    hits = changed = 0
    rows, tiles = [], []
    for i in range(args.n):
        ex = sample_example(rng)
        state["layout"] = ex.layout
        big = Image.fromarray(render(ex.layout)).resize((DISPLAY_PX, DISPLAY_PX), Image.NEAREST)
        big.save(FRAME_PATH)
        frame_photo = tk.PhotoImage(file=FRAME_PATH)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=frame_photo)
        canvas.photo_ref = frame_photo
        root.update_idletasks()
        root.update()
        time.sleep(0.02)

        before = np.asarray(ImageGrab.grab(bbox=(0, 0, DISPLAY_PX, DISPLAY_PX)).convert("RGB"))
        px, py = ground(model, before, ex.task)
        sx, sy = px * SCALE_UP, py * SCALE_UP
        res = hands.perform({"type": "click", "x": int(sx), "y": int(sy)})
        root.update()          # let Tk process the synthetic button event
        time.sleep(0.05)
        root.update()
        after = np.asarray(ImageGrab.grab(bbox=(0, 0, DISPLAY_PX, DISPLAY_PX)).convert("RGB"))

        # verification: the world only changes if the click hit a widget
        hit = ex.target.contains(px, py)
        screen_change = float(np.abs(after.astype(np.int16) - before.astype(np.int16)).mean())
        hits += hit
        changed += screen_change > 0.01
        rows.append({"task": ex.task, "pred_128": [round(px, 1), round(py, 1)],
                     "screen": [int(sx), int(sy)], "hit": bool(hit),
                     "click_ok": res["ok"], "screen_delta": round(screen_change, 3)})
        if i < 6:
            t = Image.fromarray(before).copy()
            from PIL import ImageDraw
            d = ImageDraw.Draw(t)
            d.rectangle([ex.target.x * SCALE_UP, ex.target.y * SCALE_UP,
                         (ex.target.x + ex.target.w) * SCALE_UP, (ex.target.y + ex.target.h) * SCALE_UP],
                        outline=(0, 220, 80), width=2)
            d.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], outline=(255, 50, 50), width=3)
            tiles.append(t)

    root.destroy()
    hands.close()

    result = {
        "grip_step": ck.get("step"),
        "n": args.n,
        "hit_rate": round(hits / args.n, 3),
        "verified_rate": round(changed / args.n, 3),
        "display_px": DISPLAY_PX,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if tiles:
        cols = 3
        rows_n = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * DISPLAY_PX // 2, rows_n * DISPLAY_PX // 2), (15, 15, 18))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((DISPLAY_PX // 2, DISPLAY_PX // 2)),
                        ((i % cols) * DISPLAY_PX // 2, (i // cols) * DISPLAY_PX // 2))
        os.makedirs(os.path.dirname(args.strip) or ".", exist_ok=True)
        sheet.save(args.strip)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
