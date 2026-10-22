"""A tiny synthetic desktop world.

One generator, two views of the same screen:

1. **Pixels** -- a 128x128 rendered desktop (window chrome, buttons, fields,
   checkboxes, tabs, icons, sliders) used to teach a from-scratch CNN to look
   at a screen and put a finger on the right widget.
2. **Text protocol** -- the same layout serialized as an element tree, used to
   teach the language brain how desktop-control transcripts look.

Widget labels are drawn as small 3x2 binary "glyph chips" instead of real text
because at 128x128 real font glyphs are sub-pixel. Each label owns a unique
6-bit code, so the grounding task is genuine visual search: instruction word ->
glyph pattern -> locate it -> click it. Layouts, themes, sizes and distractor
sets are randomized, so the model has to generalize over positions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

CANVAS = 128          # rendered screen size (px)
SCALE = 15            # 128 * 15 = 1920, i.e. coordinates shown in the text protocol
ROLES = ["button", "field", "checkbox", "tab", "icon", "menu", "slider"]
LABELS = [
    "submit", "cancel", "search", "settings", "profile", "inbox", "upload",
    "download", "login", "logout", "refresh", "delete", "save", "share",
    "play", "pause", "help", "home", "cart", "filter", "export", "print",
    "rename", "archive",
]
ROLE_COLOR = {
    "button": (56, 132, 255),
    "field": (245, 246, 250),
    "checkbox": (120, 200, 140),
    "tab": (160, 160, 220),
    "icon": (255, 186, 84),
    "menu": (230, 230, 235),
    "slider": (200, 110, 170),
}
TASKS = {
    "button": ["click the {label} button", "press {label}", "hit the {label} button"],
    "field": ["type into the {label} field", "click the {label} input box", "focus {label}"],
    "checkbox": ["toggle the {label} checkbox", "check {label}", "turn on {label}"],
    "tab": ["open the {label} tab", "switch to {label}", "click the {label} tab"],
    "icon": ["click the {label} icon", "open {label}", "select the {label} icon"],
    "menu": ["choose {label} from the menu", "click the {label} menu item", "pick {label}"],
    "slider": ["drag the {label} slider", "move the {label} slider", "adjust {label}"],
}
ACTION_OF_ROLE = {
    "button": "click", "field": "type", "checkbox": "toggle",
    "tab": "click", "icon": "click", "menu": "click", "slider": "drag",
}


def label_code(label: str) -> list[int]:
    """Unique stable 6-bit glyph code per label (1 = white cell, 0 = black)."""
    idx = LABELS.index(label)
    return [(idx >> b) & 1 for b in range(6)]


@dataclass
class Element:
    role: str
    label: str
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (self.x - margin <= px <= self.x + self.w + margin and
                self.y - margin <= py <= self.y + self.h + margin)


@dataclass
class Layout:
    elements: list[Element]
    theme: str
    window: tuple[int, int, int, int]
    title: str = "untitled"
    extra: dict = field(default_factory=dict)


def make_layout(rng: random.Random, n: int | None = None) -> Layout:
    n = n or rng.randint(4, 8)
    roles = rng.sample(ROLES, k=min(len(ROLES), max(2, n // 2)))
    labels = rng.sample(LABELS, k=min(len(LABELS), n))

    wx, wy, ww, wh = 10, 8, 108, 106
    title_h = 10
    # coarse placement grid inside the window body
    rows, cols = 4, 3
    cell_w = (ww - 12) / cols
    cell_h = (wh - title_h - 10) / rows
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)

    elements: list[Element] = []
    for i in range(n):
        r, c = cells[i]
        role = rng.choice(roles)
        label = labels[i]
        if role == "slider":
            w, h = int(cell_w - 8), 12
        elif role == "menu":
            w, h = int(cell_w - 8), 9
        elif role == "tab":
            w, h = int(cell_w * 0.8), 10
        elif role == "field":
            w, h = int(cell_w - 8), 14
        elif role == "checkbox":
            w, h = 12, 12
        elif role == "icon":
            w, h = 18, 18
        else:
            w, h = int(cell_w - 10), 14
        max_x = wx + ww - 6 - w
        max_y = wy + wh - 6 - h
        x = int(wx + 6 + c * cell_w + rng.randint(0, max(1, int(cell_w - w - 2))))
        y = int(wy + title_h + 4 + r * cell_h + rng.randint(0, max(1, int(cell_h - h - 2))))
        x = min(max(x, wx + 4), max_x)
        y = min(max(y, wy + title_h + 2), max_y)
        elements.append(Element(role, label, x, y, w, h))

    return Layout(
        elements=elements,
        theme=rng.choice(["light", "dark"]),
        window=(wx, wy, ww, wh),
        title=rng.choice(["Files", "Settings", "Browser", "Editor", "Mail", "Photos"]),
    )


# ---------------------------------------------------------------- rendering
def _draw_glyph(d: ImageDraw.ImageDraw, x: int, y: int, label: str, cell: int = 3) -> None:
    code = label_code(label)
    for i, bit in enumerate(code):
        r, c = divmod(i, 3)
        cx, cy = x + c * cell, y + r * cell
        col = (250, 250, 250) if bit else (18, 18, 22)
        d.rectangle([cx, cy, cx + cell - 2, cy + cell - 2], fill=col)


def render(layout: Layout, size: int = CANVAS) -> np.ndarray:
    bg = (236, 238, 243) if layout.theme == "light" else (32, 34, 40)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    wx, wy, ww, wh = layout.window
    win_fill = (250, 251, 253) if layout.theme == "light" else (44, 46, 54)
    d.rectangle([wx, wy, wx + ww, wy + wh], fill=win_fill, outline=(150, 152, 160))
    d.rectangle([wx, wy, wx + ww, wy + 10], fill=(70, 76, 96))
    for i, col in enumerate([(235, 90, 80), (240, 190, 70), (110, 200, 110)]):
        d.ellipse([wx + ww - 22 + i * 7, wy + 3, wx + ww - 18 + i * 7, wy + 7], fill=col)

    for el in layout.elements:
        x, y, w, h = el.x, el.y, el.w, el.h
        c = ROLE_COLOR[el.role]
        if el.role == "button":
            d.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=c)
            _draw_glyph(d, x + 3, y + (h - 5) // 2 + 1, el.label, 2)
        elif el.role == "field":
            d.rectangle([x, y, x + w, y + h], fill=c, outline=(160, 164, 176))
            _draw_glyph(d, x + 3, y + (h - 5) // 2 + 1, el.label, 2)
        elif el.role == "checkbox":
            d.rectangle([x, y, x + w, y + h], fill=c, outline=(60, 120, 80))
            _draw_glyph(d, x + 2, y + 2, el.label, 2)
        elif el.role == "tab":
            d.rectangle([x, y, x + w, y + h], fill=c)
            _draw_glyph(d, x + 3, y + 2, el.label, 2)
        elif el.role == "icon":
            d.rectangle([x, y, x + w, y + h], fill=c)
            _draw_glyph(d, x + 3, y + 3, el.label, 3)
        elif el.role == "menu":
            d.rectangle([x, y, x + w, y + h], fill=c, outline=(200, 200, 206))
            _draw_glyph(d, x + 2, y + 1, el.label, 2)
        elif el.role == "slider":
            d.rectangle([x, y, x + w, y + h], fill=(236, 238, 243) if layout.theme == "light"
                        else (44, 46, 54), outline=c)
            _draw_glyph(d, x + 2, y + 2, el.label, 2)
            d.line([x + 10, y + h // 2, x + w - 2, y + h // 2], fill=c, width=2)
            kx = x + 10 + int((w - 12) * 0.6)
            d.ellipse([kx - 2, y + h // 2 - 2, kx + 3, y + h // 2 + 3], fill=c)
    return np.asarray(img, dtype=np.uint8)


# ----------------------------------------------------------------- protocol
def to_protocol(layout: Layout, task: str, action: str) -> str:
    els = "".join(
        f"<el>{e.role}[{e.x*SCALE},{e.y*SCALE},{e.w*SCALE},{e.h*SCALE}]{{{e.label}}}"
        for e in layout.elements
    )
    return (f"<|ui|><|screen|>{CANVAS*SCALE}x{CANVAS*SCALE}<|window|>{layout.title}"
            f"{els}<|task|>{task}<|action|>{action}<|endoftext|>")


@dataclass
class Example:
    layout: Layout
    task: str
    target: Element
    action: str
    click: tuple[int, int]


def sample_example(rng: random.Random) -> Example:
    layout = make_layout(rng)
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    action = ACTION_OF_ROLE[target.role]
    # a click lands somewhere inside the widget
    px = int(rng.uniform(target.x + target.w * 0.2, target.x + target.w * 0.8))
    py = int(rng.uniform(target.y + target.h * 0.2, target.y + target.h * 0.8))
    return Example(layout, task, target, action, (px, py))


def protocol_corpus(n: int, seed: int = 0) -> Iterable[str]:
    rng = random.Random(seed)
    for _ in range(n):
        ex = sample_example(rng)
        x, y = ex.click[0] * SCALE, ex.click[1] * SCALE
        yield to_protocol(ex.layout, ex.task, f"{ex.action}({x},{y})")


# ------------------------------------------------------------------ gripnet
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 -"
CHAR2ID = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}   # 0 = pad


def encode_task(task: str, max_len: int = 48) -> list[int]:
    task = re.sub(r"[^a-z0-9 -]", "", task.lower())[:max_len]
    ids = [CHAR2ID.get(c, 0) for c in task]
    return ids + [0] * (max_len - len(ids))


def target_heatmap(ex: Example, grid: int = 8) -> np.ndarray:
    """Soft gaussian target over the coarse grid, centred on the widget centre."""
    hm = np.zeros((grid, grid), dtype=np.float32)
    cell = CANVAS / grid
    gx, gy = ex.target.cx / cell - 0.5, ex.target.cy / cell - 0.5
    for i in range(grid):
        for j in range(grid):
            d2 = (i - gy) ** 2 + (j - gx) ** 2
            hm[i, j] = np.exp(-d2 / (2 * 0.9 ** 2))
    return hm / hm.sum()


ACTION_CLASSES = ["click", "type", "toggle", "drag"]


def target_offset(ex: Example, grid: int = 8) -> tuple[int, list[float]]:
    """Index of the cell holding the widget centre + offset inside that cell."""
    cell = CANVAS / grid
    cx, cy = ex.target.cx / cell, ex.target.cy / cell
    ix, iy = int(cx), int(cy)
    ix, iy = min(ix, grid - 1), min(iy, grid - 1)
    dx = (cx - (ix + 0.5)) / 0.45
    dy = (cy - (iy + 0.5)) / 0.45
    return iy * grid + ix, [max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy))]


def batch(n: int, rng: random.Random, grid: int = 8):
    xs = np.empty((n, CANVAS, CANVAS, 3), dtype=np.float32)
    tasks = np.empty((n, 48), dtype=np.int64)
    targets = np.empty((n, grid * grid), dtype=np.float32)
    offs = np.empty((n, 2), dtype=np.float32)
    actions = np.empty((n,), dtype=np.int64)
    examples = []
    for i in range(n):
        ex = sample_example(rng)
        img = render(ex.layout)
        xs[i] = img.astype(np.float32) / 127.5 - 1.0
        tasks[i] = np.array(encode_task(ex.task), dtype=np.int64)
        targets[i] = target_heatmap(ex, grid).reshape(-1)
        offs[i] = np.array(target_offset(ex, grid)[1], dtype=np.float32)
        actions[i] = ACTION_CLASSES.index(ex.action)
        examples.append(ex)
    return xs, tasks, targets, offs, actions, examples
