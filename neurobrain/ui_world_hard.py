"""Hard UI World v1 (`ui_world_hard`): realistic PIL-rendered desktop UIs for visual grounding.

Features (per agents/03-eyes/PROMPT.md & specs/data_formats.md):
  - Native 256x256 desktop rendering with real TrueType fonts at 8-14 px (DPI 100%-200%)
  - 5 desktop layout archetypes: toolbar_workspace, tabbed_dialog, file_manager,
    settings_panel, editor_split (toolbars, tabs, sidebars, modal dialogs, list views)
  - Widget states: normal, focused, selected, disabled
  - Seen themes (light, dark, high_contrast) + strictly disjoint OOD themes & OOD fonts
  - Realistic pointer cursor in frame + deterministic JPEG compression artifacts
  - Difficulty tiers: easy (4-6 widgets), medium (7-10), crowded (11-16), mixed
  - Ablation mode: label_mode="text" (default real text) vs "glyph" (6-bit code chip)
  - Same ROLES, LABELS, ACTION_CLASSES, CHAR_VOCAB taxonomy as ui_world.py
  - Streaming .npz shard generator + SHA-256 manifest verifier per specs/data_formats.md
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ui_world import (
    ACTION_CLASSES,
    ACTION_OF_ROLE,
    CANVAS as COARSE_CANVAS,
    LABELS,
    ROLES,
    TASKS,
    encode_task,
    label_code,
)

HARD_CANVAS = 256  # native screen resolution where 8-14px text is crisp & authentic

ARCHETYPES = [
    "toolbar_workspace",
    "tabbed_dialog",
    "file_manager",
    "settings_panel",
    "editor_split",
]

STATES = ["normal", "focused", "selected", "disabled"]

# Strictly disjoint Seen vs OOD font families for honest generalisation tables
_CANDIDATE_SEEN_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
_CANDIDATE_OOD_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]

SEEN_FONT_PATHS = [p for p in _CANDIDATE_SEEN_FONTS if os.path.exists(p)] or _CANDIDATE_SEEN_FONTS[:1]
OOD_FONT_PATHS = [p for p in _CANDIDATE_OOD_FONTS if os.path.exists(p)] or ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"]

THEME_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    # Seen themes
    "light": {
        "desk": (226, 230, 238),
        "win": (248, 249, 252),
        "panel": (236, 239, 245),
        "chrome": (58, 66, 86),
        "border": (176, 182, 194),
        "text": (24, 28, 36),
        "muted": (142, 148, 160),
        "btn": (48, 120, 240),
        "btn_text": (250, 252, 255),
        "field": (255, 255, 255),
        "tab": (218, 224, 236),
        "tab_sel": (248, 249, 252),
        "accent": (30, 108, 235),
        "icon_bg": (232, 238, 248),
    },
    "dark": {
        "desk": (22, 24, 30),
        "win": (36, 39, 48),
        "panel": (29, 32, 40),
        "chrome": (20, 22, 28),
        "border": (74, 80, 96),
        "text": (232, 236, 244),
        "muted": (106, 112, 126),
        "btn": (64, 132, 245),
        "btn_text": (250, 252, 255),
        "field": (24, 26, 33),
        "tab": (46, 50, 62),
        "tab_sel": (62, 68, 86),
        "accent": (92, 164, 255),
        "icon_bg": (48, 53, 66),
    },
    "high_contrast": {
        "desk": (0, 0, 0),
        "win": (8, 8, 12),
        "panel": (14, 14, 20),
        "chrome": (0, 0, 0),
        "border": (255, 255, 255),
        "text": (255, 255, 255),
        "muted": (130, 130, 130),
        "btn": (0, 90, 200),
        "btn_text": (255, 255, 0),
        "field": (0, 0, 0),
        "tab": (24, 24, 32),
        "tab_sel": (0, 120, 215),
        "accent": (0, 255, 255),
        "icon_bg": (20, 20, 28),
    },
    # Unseen OOD themes (held out strictly for generalisation evaluation)
    "solarized_light": {
        "desk": (238, 232, 213),
        "win": (253, 246, 227),
        "panel": (238, 232, 213),
        "chrome": (88, 110, 117),
        "border": (147, 161, 161),
        "text": (55, 71, 79),
        "muted": (158, 167, 165),
        "btn": (38, 139, 210),
        "btn_text": (253, 246, 227),
        "field": (255, 251, 240),
        "tab": (229, 222, 201),
        "tab_sel": (253, 246, 227),
        "accent": (42, 161, 152),
        "icon_bg": (234, 227, 206),
    },
    "nord_slate": {
        "desk": (36, 41, 51),
        "win": (46, 52, 64),
        "panel": (59, 66, 82),
        "chrome": (36, 41, 51),
        "border": (76, 86, 106),
        "text": (229, 233, 240),
        "muted": (118, 129, 150),
        "btn": (94, 129, 172),
        "btn_text": (236, 239, 244),
        "field": (40, 45, 56),
        "tab": (59, 66, 82),
        "tab_sel": (76, 86, 106),
        "accent": (136, 192, 208),
        "icon_bg": (67, 76, 94),
    },
    "high_contrast_amber": {
        "desk": (10, 8, 4),
        "win": (18, 14, 8),
        "panel": (26, 20, 10),
        "chrome": (12, 10, 4),
        "border": (235, 170, 30),
        "text": (255, 196, 56),
        "muted": (128, 96, 36),
        "btn": (85, 55, 10),
        "btn_text": (255, 215, 90),
        "field": (12, 10, 5),
        "tab": (34, 24, 10),
        "tab_sel": (110, 72, 12),
        "accent": (255, 210, 60),
        "icon_bg": (38, 28, 12),
    },
}

SEEN_THEMES = ["light", "dark", "high_contrast"]
OOD_THEMES = ["solarized_light", "nord_slate", "high_contrast_amber"]

TASKS_HARD = {
    "button": TASKS["button"] + ["select {label}", "tap {label}", "click {label}"],
    "field": TASKS["field"] + ["enter text in {label}", "select the {label} box", "click {label} field"],
    "checkbox": TASKS["checkbox"] + ["enable {label}", "tick {label}", "click {label} checkbox"],
    "tab": TASKS["tab"] + ["go to {label} tab", "select {label} tab", "activate {label}"],
    "icon": TASKS["icon"] + ["tap the {label} icon", "launch {label}", "press {label} icon"],
    "menu": TASKS["menu"] + ["select {label} item", "open {label} option", "click {label} in list"],
    "slider": TASKS["slider"] + ["set the {label} slider", "slide {label}", "change {label} level"],
}


@functools.lru_cache(maxsize=64)
def _get_font(path: str, size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size_px)
    except Exception:
        return ImageFont.load_default()


@dataclass
class HardElement:
    role: str
    label: str
    x: int
    y: int
    w: int
    h: int
    state: str = "normal"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.h + margin
        )


@dataclass
class HardLayout:
    elements: list[HardElement]
    theme: str
    archetype: str
    window: tuple[int, int, int, int]
    title: str
    font_path: str
    font_px: int
    dpi_scale: float
    cursor: tuple[int, int] | None = None
    jpeg_quality: int | None = None
    label_mode: str = "text"
    tier: str = "mixed"
    panels: list[tuple[str, int, int, int, int]] = field(default_factory=list)


@dataclass
class HardExample:
    layout: HardLayout
    task: str
    target: HardElement
    action: str
    click: tuple[int, int]


def _pick_state(rng: random.Random, allow_disabled: bool = True) -> str:
    r = rng.random()
    if r < 0.62:
        return "normal"
    if r < 0.76:
        return "focused"
    if r < 0.90:
        return "selected"
    return "disabled" if allow_disabled else "normal"


def _pack_slots(
    slots: list[tuple[str, int, int, int, int]],
    labels: list[str],
    rng: random.Random,
) -> list[HardElement]:
    els: list[HardElement] = []
    for (role, x, y, w, h), lbl in zip(slots, labels):
        x = max(2, min(x, HARD_CANVAS - 18))
        y = max(2, min(y, HARD_CANVAS - 14))
        w = max(16, min(w, HARD_CANVAS - x - 2))
        h = max(12, min(h, HARD_CANVAS - y - 2))
        els.append(HardElement(role=role, label=lbl, x=x, y=y, w=w, h=h, state=_pick_state(rng)))
    return els


def make_hard_layout(
    rng: random.Random,
    tier: str = "mixed",
    split: str = "seen",
    label_mode: str = "text",
    size_override: str | None = None,
) -> HardLayout:
    if tier == "mixed":
        r = rng.random()
        eff_tier = "easy" if r < 0.25 else ("medium" if r < 0.65 else "crowded")
    else:
        eff_tier = tier

    themes = SEEN_THEMES if split == "seen" else OOD_THEMES
    fonts = SEEN_FONT_PATHS if split == "seen" else OOD_FONT_PATHS
    theme = rng.choice(themes)
    font_path = rng.choice(fonts)

    # DPI 100%-200% -> font_px in 8..14
    if size_override == "compact":
        dpi_scale = rng.uniform(1.0, 1.15)
        font_px = rng.choice([8, 9])
    elif size_override == "large":
        dpi_scale = rng.uniform(1.65, 2.0)
        font_px = rng.choice([13, 14])
    else:
        if eff_tier == "easy":
            dpi_scale = rng.uniform(1.35, 1.9)
            font_px = rng.randint(11, 14)
        elif eff_tier == "medium":
            dpi_scale = rng.uniform(1.15, 1.65)
            font_px = rng.randint(9, 13)
        else:
            dpi_scale = rng.uniform(1.0, 1.4)
            font_px = rng.randint(8, 11)

    archetype = rng.choice(ARCHETYPES)
    pad = rng.randint(2, 10) if eff_tier != "crowded" else rng.randint(2, 5)
    wx, wy = pad, pad
    ww = HARD_CANVAS - 2 * pad
    wh = HARD_CANVAS - 2 * pad
    title_h = 16

    panels: list[tuple[str, int, int, int, int]] = []
    slots: list[tuple[str, int, int, int, int]] = []

    if eff_tier == "easy":
        target_n = rng.randint(4, 6)
    elif eff_tier == "medium":
        target_n = rng.randint(7, 10)
    else:
        target_n = rng.randint(11, 16)

    bx, by = wx + 4, wy + title_h + 3
    bw, bh = ww - 8, wh - title_h - 6

    if archetype == "toolbar_workspace":
        tb_h = 24
        sb_w = 64
        panels.append(("toolbar", bx, by, bw, tb_h))
        panels.append(("sidebar", bx, by + tb_h + 2, sb_w, bh - tb_h - 2))
        # Toolbar icons/buttons (3-5)
        n_tb = 3 if eff_tier == "easy" else (4 if eff_tier == "medium" else 5)
        tw = (bw - 8) // n_tb
        for i in range(n_tb):
            role = "icon" if i % 2 == 0 else "button"
            slots.append((role, bx + 3 + i * tw, by + 3, tw - 4, tb_h - 6))
        # Sidebar menu items
        n_sb = max(2, min(5, (target_n - n_tb) // 2))
        sh = min(22, (bh - tb_h - 10) // max(1, n_sb))
        for i in range(n_sb):
            slots.append(("menu", bx + 3, by + tb_h + 6 + i * (sh + 2), sb_w - 6, sh))
        # Main workspace controls
        rem = max(1, target_n - len(slots))
        mx, my = bx + sb_w + 6, by + tb_h + 6
        mw, mh = bw - sb_w - 8, bh - tb_h - 8
        cols = 2 if rem >= 4 else 1
        rows = (rem + cols - 1) // cols
        cw, ch = mw // cols, min(26, mh // max(1, rows))
        ws_roles = ["field", "checkbox", "slider", "button"]
        for i in range(rem):
            r, c = divmod(i, cols)
            slots.append((rng.choice(ws_roles), mx + c * cw + 2, my + r * ch + 2, cw - 6, max(14, ch - 4)))

    elif archetype == "tabbed_dialog":
        # Modal dialog centered inside background window
        dx = wx + (16 if eff_tier != "crowded" else 8)
        dy = wy + title_h + (12 if eff_tier != "crowded" else 6)
        dw = ww - 2 * (dx - wx)
        dh = wh - title_h - (dy - wy - title_h) - 6
        panels.append(("dialog", dx, dy, dw, dh))
        # Top tab strip (3-4 tabs)
        n_tabs = 3 if eff_tier != "crowded" else 4
        tw = (dw - 8) // n_tabs
        for i in range(n_tabs):
            slots.append(("tab", dx + 4 + i * tw, dy + 4, tw - 3, 18))
        # Body rows (fields, checkboxes, sliders)
        n_bottom = 2 if eff_tier == "easy" else 3
        rem = max(1, target_n - n_tabs - n_bottom)
        cols = 2 if rem >= 5 else 1
        rows = (rem + cols - 1) // cols
        body_y = dy + 26
        body_h = dh - 54
        cw, ch = (dw - 12) // cols, min(24, max(15, body_h // max(1, rows)))
        form_roles = ["field", "checkbox", "slider"]
        for i in range(rem):
            r, c = divmod(i, cols)
            slots.append((rng.choice(form_roles), dx + 6 + c * cw, body_y + r * ch, cw - 6, max(13, ch - 3)))
        # Bottom action buttons (Save / Cancel / Submit etc.)
        bw_btn = min(68, (dw - 16) // n_bottom)
        for i in range(n_bottom):
            slots.append(("button", dx + dw - (n_bottom - i) * (bw_btn + 4) - 4, dy + dh - 23, bw_btn, 18))

    elif archetype == "file_manager":
        # Top search/toolbar + left places sidebar + file list/icon grid
        panels.append(("toolbar", bx, by, bw, 22))
        sb_w = 62
        panels.append(("sidebar", bx, by + 24, sb_w, bh - 24))
        # Top bar: search field + 2-3 action icons/buttons
        slots.append(("field", bx + 3, by + 3, bw // 2 - 4, 16))
        n_act = 2 if eff_tier == "easy" else 3
        aw = (bw // 2 - 6) // n_act
        for i in range(n_act):
            slots.append((rng.choice(["icon", "button"]), bx + bw // 2 + 2 + i * aw, by + 3, aw - 3, 16))
        # Sidebar items
        n_sb = 2 if eff_tier == "easy" else (3 if eff_tier == "medium" else 4)
        for i in range(n_sb):
            slots.append(("menu", bx + 3, by + 28 + i * 21, sb_w - 6, 18))
        # Main list/grid items
        rem = max(1, target_n - len(slots))
        lx, ly = bx + sb_w + 5, by + 28
        lw, lh = bw - sb_w - 7, bh - 30
        cols = 2 if rem >= 5 else 1
        rows = (rem + cols - 1) // cols
        cw, ch = lw // cols, min(22, max(14, lh // max(1, rows)))
        for i in range(rem):
            r, c = divmod(i, cols)
            role = rng.choice(["menu", "icon", "checkbox", "button"])
            slots.append((role, lx + c * cw + 2, ly + r * ch + 1, cw - 5, max(13, ch - 3)))

    elif archetype == "settings_panel":
        sb_w = 66
        panels.append(("sidebar", bx, by, sb_w, bh))
        n_sb = 2 if eff_tier == "easy" else (3 if eff_tier == "medium" else 5)
        for i in range(n_sb):
            slots.append(("tab" if i % 2 == 0 else "menu", bx + 3, by + 4 + i * 21, sb_w - 6, 18))
        rem = max(2, target_n - n_sb)
        rx, ry = bx + sb_w + 6, by + 4
        rw, rh = bw - sb_w - 8, bh - 8
        cols = 2 if rem >= 6 else 1
        rows = (rem + cols - 1) // cols
        cw, ch = rw // cols, min(24, max(14, rh // max(1, rows)))
        s_roles = ["checkbox", "slider", "field", "button"]
        for i in range(rem):
            r, c = divmod(i, cols)
            slots.append((s_roles[i % len(s_roles)], rx + c * cw + 2, ry + r * ch + 1, cw - 6, max(13, ch - 3)))

    else:  # editor_split
        panels.append(("toolbar", bx, by, bw, 20))
        # Top tabs + toolbar icons
        n_top = 3 if eff_tier == "easy" else (4 if eff_tier == "medium" else 5)
        tw = (bw - 6) // n_top
        for i in range(n_top):
            role = "tab" if i < (n_top + 1) // 2 else "icon"
            slots.append((role, bx + 3 + i * tw, by + 2, tw - 3, 16))
        # Bottom/split panes
        rem = max(1, target_n - n_top)
        py0 = by + 24
        ph = bh - 26
        panels.append(("split", bx, py0 + ph // 2, bw, ph - ph // 2))
        cols = 2 if rem >= 4 else 1
        rows = (rem + cols - 1) // cols
        cw, ch = (bw - 6) // cols, min(24, max(14, ph // max(1, rows)))
        e_roles = ["field", "button", "checkbox", "slider", "menu"]
        for i in range(rem):
            r, c = divmod(i, cols)
            slots.append((rng.choice(e_roles), bx + 3 + c * cw, py0 + 2 + r * ch, cw - 5, max(13, ch - 3)))

    slots = slots[: min(len(LABELS), max(4, target_n))]
    chosen_labels = rng.sample(LABELS, k=len(slots))
    elements = _pack_slots(slots, chosen_labels, rng)

    # Ensure at least 2 elements are non-disabled so we always have valid targets
    enabled = [e for e in elements if e.state != "disabled"]
    if len(enabled) < 2:
        for e in elements[:2]:
            e.state = "normal"

    # Optional mouse cursor in frame (80% probability)
    cursor = None
    if rng.random() < 0.80:
        cursor = (rng.randint(12, HARD_CANVAS - 18), rng.randint(12, HARD_CANVAS - 18))

    # JPEG compression quality (75% probability; lower quality on crowded tier)
    jpeg_q = None
    if rng.random() < 0.75:
        jpeg_q = rng.randint(55, 82) if eff_tier == "crowded" else rng.randint(72, 92)

    return HardLayout(
        elements=elements,
        theme=theme,
        archetype=archetype,
        window=(wx, wy, ww, wh),
        title=rng.choice(["Files", "Settings", "Editor", "Workspace", "Terminal", "Package Manager"]),
        font_path=font_path,
        font_px=font_px,
        dpi_scale=round(dpi_scale, 2),
        cursor=cursor,
        jpeg_quality=jpeg_q,
        label_mode=label_mode,
        tier=eff_tier,
        panels=panels,
    )


def _draw_mini_icon(d: ImageDraw.ImageDraw, x: int, y: int, s: int, label: str, col: tuple[int, int, int]) -> None:
    """Deterministic tiny vector icon cue + text so icon buttons look like real desktop icons."""
    idx = LABELS.index(label) % 6
    if idx == 0:
        d.ellipse([x, y, x + s, y + s], outline=col, width=1)
    elif idx == 1:
        d.rectangle([x, y + 1, x + s, y + s - 1], outline=col, width=1)
    elif idx == 2:
        d.polygon([(x, y), (x + s, y + s // 2), (x, y + s)], outline=col)
    elif idx == 3:
        d.line([x, y + s // 2, x + s, y + s // 2], fill=col, width=2)
        d.line([x + s // 2, y, x + s // 2, y + s], fill=col, width=2)
    elif idx == 4:
        d.polygon([(x + s // 2, y), (x + s, y + s), (x, y + s)], outline=col)
    else:
        d.line([x, y + 2, x + s, y + 2], fill=col, width=1)
        d.line([x, y + s // 2, x + s, y + s // 2], fill=col, width=1)
        d.line([x, y + s - 2, x + s, y + s - 2], fill=col, width=1)


def _draw_glyph_chip(d: ImageDraw.ImageDraw, x: int, y: int, label: str, cell: int = 4) -> None:
    code = label_code(label)
    for i, bit in enumerate(code):
        r, c = divmod(i, 3)
        cx, cy = x + c * (cell + 1), y + r * (cell + 1)
        col = (248, 248, 250) if bit else (22, 24, 30)
        d.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], fill=col)


def _draw_label(
    d: ImageDraw.ImageDraw,
    el: HardElement,
    tx: int,
    ty: int,
    max_w: int,
    fg: tuple[int, int, int],
    font: ImageFont.ImageFont,
    font_px: int,
    label_mode: str,
) -> None:
    if label_mode == "glyph":
        _draw_glyph_chip(d, tx, max(el.y + 1, el.y + (el.h - 9) // 2), el.label, cell=4)
        return
    # Fit label cleanly inside widget width; dynamically step down font by 1-2px only if tight
    text = el.label
    bb = font.getbbox(text)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    if tw > max_w and max_w > 12:
        # Abbreviate only if narrower than text, keeping distinctive prefix
        max_chars = max(3, int(len(text) * max_w / max(1, tw)))
        text = text[:max_chars]
        bb = font.getbbox(text)
        th = bb[3] - bb[1]
    yy = max(el.y, ty - bb[1] + max(0, (el.h - th) // 2 - (ty - el.y)))
    d.text((tx, yy), text, fill=fg, font=font)


def _draw_cursor(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    pts = [
        (cx, cy),
        (cx, cy + 11),
        (cx + 3, cy + 8),
        (cx + 6, cy + 13),
        (cx + 8, cy + 12),
        (cx + 5, cy + 7),
        (cx + 9, cy + 7),
    ]
    d.polygon(pts, fill=(252, 252, 254), outline=(18, 18, 22))


def render_hard(layout: HardLayout, size: int = HARD_CANVAS) -> np.ndarray:
    pal = THEME_PALETTES[layout.theme]
    img = Image.new("RGB", (HARD_CANVAS, HARD_CANVAS), pal["desk"])
    d = ImageDraw.Draw(img)

    wx, wy, ww, wh = layout.window
    # Main window & titlebar
    d.rectangle([wx + 2, wy + 2, wx + ww + 2, wy + wh + 2], fill=(12, 14, 18))
    d.rectangle([wx, wy, wx + ww, wy + wh], fill=pal["win"], outline=pal["border"], width=1)
    d.rectangle([wx, wy, wx + ww, wy + 16], fill=pal["chrome"])
    title_font = _get_font(layout.font_path, 9)
    d.text((wx + 6, wy + 3), layout.title, fill=(235, 238, 245), font=title_font)
    for i, col in enumerate([(230, 88, 78), (238, 188, 68), (98, 196, 104)]):
        d.ellipse([wx + ww - 26 + i * 8, wy + 5, wx + ww - 20 + i * 8, wy + 11], fill=col)

    # Structural panels (toolbars, sidebars, modal dialogs, split panes)
    for kind, px, py, pw, ph in layout.panels:
        if kind == "dialog":
            d.rectangle([px + 3, py + 3, px + pw + 3, py + ph + 3], fill=(10, 12, 16))
            d.rectangle([px, py, px + pw, py + ph], fill=pal["win"], outline=pal["accent"], width=1)
        else:
            d.rectangle([px, py, px + pw, py + ph], fill=pal["panel"], outline=pal["border"], width=1)

    font = _get_font(layout.font_path, layout.font_px)

    for el in layout.elements:
        x, y, w, h = el.x, el.y, el.w, el.h
        disabled = el.state == "disabled"
        focused = el.state == "focused"
        selected = el.state == "selected"

        fg = pal["muted"] if disabled else pal["text"]
        border = pal["accent"] if (focused or selected) else pal["border"]
        bw_px = 2 if focused else 1

        if el.role == "button":
            fill = pal["panel"] if disabled else (pal["accent"] if selected else pal["btn"])
            bfg = pal["muted"] if disabled else pal["btn_text"]
            d.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=fill, outline=border, width=bw_px)
            _draw_label(d, el, x + 4, y + 2, w - 8, bfg, font, layout.font_px, layout.label_mode)

        elif el.role == "field":
            d.rectangle([x, y, x + w, y + h], fill=pal["field"], outline=border, width=bw_px)
            _draw_label(d, el, x + 4, y + 2, w - 10, fg, font, layout.font_px, layout.label_mode)
            if focused:
                d.line([x + w - 5, y + 3, x + w - 5, y + h - 3], fill=pal["accent"], width=1)

        elif el.role == "checkbox":
            box_s = min(11, h - 2)
            by0 = y + (h - box_s) // 2
            d.rectangle([x + 2, by0, x + 2 + box_s, by0 + box_s], fill=pal["field"], outline=border, width=bw_px)
            if selected:
                d.line([x + 4, by0 + box_s // 2, x + 2 + box_s // 2, by0 + box_s - 2], fill=pal["accent"], width=2)
                d.line([x + 2 + box_s // 2, by0 + box_s - 2, x + box_s, by0 + 2], fill=pal["accent"], width=2)
            _draw_label(d, el, x + box_s + 6, y + 2, w - box_s - 8, fg, font, layout.font_px, layout.label_mode)

        elif el.role == "tab":
            fill = pal["tab_sel"] if selected else pal["tab"]
            d.rectangle([x, y, x + w, y + h], fill=fill, outline=border, width=1)
            if selected or focused:
                d.line([x + 1, y + h - 1, x + w - 1, y + h - 1], fill=pal["accent"], width=2)
            _draw_label(d, el, x + 4, y + 2, w - 8, fg, font, layout.font_px, layout.label_mode)

        elif el.role == "icon":
            fill = pal["tab_sel"] if selected else pal["icon_bg"]
            d.rounded_rectangle([x, y, x + w, y + h], radius=2, fill=fill, outline=border, width=bw_px)
            ico_s = min(8, h - 4)
            _draw_mini_icon(d, x + 3, y + (h - ico_s) // 2, ico_s, el.label, fg)
            _draw_label(d, el, x + ico_s + 5, y + 2, w - ico_s - 7, fg, font, max(8, layout.font_px - 1), layout.label_mode)

        elif el.role == "menu":
            fill = pal["tab_sel"] if selected else pal["win"]
            d.rectangle([x, y, x + w, y + h], fill=fill, outline=border if focused else pal["panel"], width=1)
            if selected:
                d.rectangle([x, y + 2, x + 2, y + h - 2], fill=pal["accent"])
            _draw_label(d, el, x + 5, y + 2, w - 8, fg, font, layout.font_px, layout.label_mode)

        elif el.role == "slider":
            d.rectangle([x, y, x + w, y + h], fill=pal["win"], outline=border if focused else pal["panel"], width=1)
            lbl_w = min(w // 2, 42)
            _draw_label(d, el, x + 3, y + 2, lbl_w - 4, fg, font, max(8, layout.font_px - 1), layout.label_mode)
            tx0, tx1 = x + lbl_w + 2, x + w - 5
            ty_mid = y + h // 2
            if tx1 > tx0 + 4:
                d.line([tx0, ty_mid, tx1, ty_mid], fill=pal["border"], width=2)
                frac = 0.7 if selected else 0.45
                kx = int(tx0 + (tx1 - tx0) * frac)
                d.line([tx0, ty_mid, kx, ty_mid], fill=pal["accent"], width=2)
                d.ellipse([kx - 3, ty_mid - 3, kx + 3, ty_mid + 3], fill=pal["accent"])

    if layout.cursor is not None:
        _draw_cursor(d, layout.cursor[0], layout.cursor[1])

    if layout.jpeg_quality is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=layout.jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    if size != HARD_CANVAS:
        img = img.resize((size, size), Image.BILINEAR)

    return np.asarray(img, dtype=np.uint8)


def sample_hard_example(
    rng: random.Random,
    tier: str = "mixed",
    split: str = "seen",
    label_mode: str = "text",
    size_override: str | None = None,
) -> HardExample:
    layout = make_hard_layout(rng, tier=tier, split=split, label_mode=label_mode, size_override=size_override)
    candidates = [e for e in layout.elements if e.state != "disabled"] or layout.elements
    target = rng.choice(candidates)
    task = rng.choice(TASKS_HARD[target.role]).format(label=target.label)
    action = ACTION_OF_ROLE[target.role]
    px = int(rng.uniform(target.x + target.w * 0.25, target.x + target.w * 0.75))
    py = int(rng.uniform(target.y + target.h * 0.25, target.y + target.h * 0.75))
    return HardExample(layout=layout, task=task, target=target, action=action, click=(px, py))


def make_hard_val_split(
    n: int = 1000,
    seed: int = 202604,
    tier: str = "mixed",
    split: str = "seen",
    label_mode: str = "text",
    size_override: str | None = None,
) -> list[HardExample]:
    rng = random.Random(seed)
    return [
        sample_hard_example(rng, tier=tier, split=split, label_mode=label_mode, size_override=size_override)
        for _ in range(n)
    ]


# Vectorized grid coordinates for 50x faster heatmap generation
_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def target_heatmap_fast(cx: float, cy: float, canvas: float = HARD_CANVAS, grid: int = 32) -> np.ndarray:
    if grid not in _GRID_CACHE:
        ys, xs = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="ij")
        _GRID_CACHE[grid] = (ys, xs)
    ys, xs = _GRID_CACHE[grid]
    cell = canvas / float(grid)
    gx, gy = cx / cell - 0.5, cy / cell - 0.5
    d2 = (ys - gy) ** 2 + (xs - gx) ** 2
    hm = np.exp(-d2 / (2.0 * 0.9 * 0.9))
    return hm / hm.sum()


def target_offset_fast(cx: float, cy: float, canvas: float = HARD_CANVAS, grid: int = 32) -> tuple[int, tuple[float, float]]:
    cell = canvas / float(grid)
    gcx, gcy = cx / cell, cy / cell
    ix, iy = min(int(gcx), grid - 1), min(int(gcy), grid - 1)
    dx = max(-1.0, min(1.0, (gcx - (ix + 0.5)) / 0.45))
    dy = max(-1.0, min(1.0, (gcy - (iy + 0.5)) / 0.45))
    return iy * grid + ix, (dx, dy)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_worker(args: tuple[int, int, str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed, count, tier, label_mode = args
    rng = random.Random(seed)
    images = np.empty((count, COARSE_CANVAS, COARSE_CANVAS, 3), dtype=np.uint8)
    tasks = np.empty((count, 48), dtype=np.int64)
    heat = np.empty((count, 32, 32), dtype=np.float16)
    offsets = np.empty((count, 2), dtype=np.float32)
    actions = np.empty((count,), dtype=np.int64)

    for i in range(count):
        ex = sample_hard_example(rng, tier=tier, split="seen", label_mode=label_mode)
        images[i] = render_hard(ex.layout, size=COARSE_CANVAS)
        tasks[i] = np.asarray(encode_task(ex.task), dtype=np.int64)
        heat[i] = target_heatmap_fast(ex.target.cx, ex.target.cy, HARD_CANVAS, 32).astype(np.float16)
        _, (dx, dy) = target_offset_fast(ex.target.cx, ex.target.cy, HARD_CANVAS, 32)
        offsets[i, 0] = dx
        offsets[i, 1] = dy
        actions[i] = ACTION_CLASSES.index(ex.action)
    return images, tasks, heat, offsets, actions


def build_hard_shards(
    out_dir: str,
    total_samples: int = 50000,
    shard_size: int = 5000,
    seed: int = 1000,
    tier: str = "mixed",
    label_mode: str = "text",
    workers: int = 2,
) -> dict[str, Any]:
    """Stream deterministic 5k-sample .npz shards to disk per specs/data_formats.md."""
    os.makedirs(out_dir, exist_ok=True)
    n_shards = (total_samples + shard_size - 1) // shard_size
    shard_records = []
    written = 0

    pool = mp.Pool(processes=workers) if workers > 1 else None
    try:
        for s_idx in range(n_shards):
            cur_n = min(shard_size, total_samples - written)
            shard_seed = seed + s_idx * 100003
            if pool is not None and cur_n >= 2 * workers:
                # Split single shard deterministically across workers so RSS stays tiny
                half = cur_n // 2
                parts = pool.map(
                    _render_worker,
                    [
                        (shard_seed, half, tier, label_mode),
                        (shard_seed + 1, cur_n - half, tier, label_mode),
                    ],
                )
                images = np.concatenate([p[0] for p in parts], axis=0)
                tasks = np.concatenate([p[1] for p in parts], axis=0)
                heat = np.concatenate([p[2] for p in parts], axis=0)
                offsets = np.concatenate([p[3] for p in parts], axis=0)
                actions = np.concatenate([p[4] for p in parts], axis=0)
                del parts
            else:
                images, tasks, heat, offsets, actions = _render_worker((shard_seed, cur_n, tier, label_mode))

            fname = f"shard-{s_idx:04d}.npz"
            fpath = os.path.join(out_dir, fname)
            np.savez_compressed(
                fpath,
                images=images,
                tasks=tasks,
                heat=heat,
                offsets=offsets,
                actions=actions,
            )
            sha = _sha256_file(fpath)
            shard_records.append({
                "file": fname,
                "n": int(cur_n),
                "sha256": sha,
                "bytes": os.path.getsize(fpath),
            })
            written += cur_n
            del images, tasks, heat, offsets, actions
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    manifest = {
        "version": "v1",
        "generated_by": "agent-03",
        "world": "ui_world_hard",
        "tier": tier,
        "label_mode": label_mode,
        "seed": seed,
        "n": int(written),
        "shard_size": int(shard_size),
        "shards": shard_records,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_hard_shards(shard_dir: str) -> tuple[bool, str]:
    mpath = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(mpath):
        return False, f"missing manifest.json in {shard_dir}"
    with open(mpath) as f:
        manifest = json.load(f)
    total = 0
    for rec in manifest.get("shards", []):
        fpath = os.path.join(shard_dir, rec["file"])
        if not os.path.exists(fpath):
            return False, f"missing shard {rec['file']}"
        actual = _sha256_file(fpath)
        if actual != rec["sha256"]:
            return False, f"sha256 mismatch on {rec['file']}: expected {rec['sha256']}, got {actual}"
        total += int(rec["n"])
    if total != int(manifest.get("n", -1)):
        return False, f"sample count mismatch: sum={total} != manifest.n={manifest.get('n')}"
    return True, f"verified {len(manifest.get('shards', []))} shards, n={total}"


def render_showcase(out_path: str, seed: int = 77) -> None:
    """Render an 8-tile showcase across archetypes, themes, states, and OOD styles."""
    rng = random.Random(seed)
    tiles = []
    configs = [
        ("easy", "seen", "text"),
        ("medium", "seen", "text"),
        ("medium", "seen", "text"),
        ("crowded", "seen", "text"),
        ("crowded", "seen", "text"),
        ("crowded", "seen", "text"),
        ("medium", "ood", "text"),
        ("crowded", "ood", "text"),
    ]
    cap_font = _get_font(SEEN_FONT_PATHS[0], 10)
    for tier, split, lmode in configs:
        ex = sample_hard_example(rng, tier=tier, split=split, label_mode=lmode)
        arr = render_hard(ex.layout, size=HARD_CANVAS)
        tile = Image.new("RGB", (HARD_CANVAS, HARD_CANVAS + 28), (16, 18, 22))
        tile.paste(Image.fromarray(arr), (0, 0))
        d = ImageDraw.Draw(tile)
        t = ex.target
        d.rectangle([t.x, t.y, t.x + t.w, t.y + t.h], outline=(0, 235, 95), width=2)
        d.text((4, HARD_CANVAS + 2), f"[{split}/{ex.layout.tier}] {ex.task[:28]}", fill=(238, 240, 246), font=cap_font)
        d.text(
            (4, HARD_CANVAS + 14),
            f"{ex.layout.archetype} | {ex.layout.theme} | {ex.layout.font_px}px | n={len(ex.layout.elements)}",
            fill=(155, 205, 255),
            font=cap_font,
        )
        tiles.append(tile)

    cols = 4
    rows = 2
    tw, th = HARD_CANVAS, HARD_CANVAS + 28
    sheet = Image.new("RGB", (cols * tw, rows * th), (12, 12, 16))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.quantize(colors=256, method=Image.MEDIANCUT).save(out_path, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ui_hard_v1")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--showcase", default="")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        ok, detail = verify_hard_shards(args.out)
        print(json.dumps({"ok": ok, "detail": detail}))
        raise SystemExit(0 if ok else 1)

    t0 = __import__("time").time()
    manifest = build_hard_shards(
        out_dir=args.out,
        total_samples=args.n,
        shard_size=args.shard_size,
        seed=args.seed,
        workers=args.workers,
    )
    ok, detail = verify_hard_shards(args.out)
    elapsed = round(__import__("time").time() - t0, 1)
    if args.showcase:
        render_showcase(args.showcase)
    print(json.dumps({
        "ok": ok,
        "detail": detail,
        "elapsed_s": elapsed,
        "n": manifest["n"],
        "shards": len(manifest["shards"]),
        "total_mb": round(sum(s["bytes"] for s in manifest["shards"]) / (1024 * 1024), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
