"""The hard UI world: a desktop that looks like a desktop.

The seed world (``ui_world.py``) drew 6-bit "glyph chips" on a 4x3 grid because
at 128x128 real glyphs are sub-pixel. That made grounding a search over 72 of
1024 cells with 2.3 bits of entropy -- an easy problem. This world is the other
end: real antialiased text, window chrome, menu bars, toolbars, tab strips,
sidebars, list views, settings grids, modal dialogs, disabled/focused/selected/
hover states, three themes, a cursor in frame, JPEG noise and DPI 100-200%.

Same label taxonomy as the seed world (24 labels, 7 roles) so checkpoints
transfer; same 128x128 output, 48-char instruction encoding, four action
classes. What changes is the distribution:

    * widgets are placed by a layout grammar with real section bands
      (title / menu / toolbar / tabs / body: sidebar+content / status bar),
      not on a 4x3 grid -- all 1024 cells are reachable
    * text is drawn with DejaVu at 2x supersampling, box-filtered down, so a
      7 px label is antialiased the way a scaled screenshot is
    * offsets are scaled by half a cell, so they only clip at the true edge
    * tier 0/1/2 turns crowding, dialog frequency and noise up; DPI scales the
      chrome so 200% carries fewer, bigger widgets

Everything is deterministic: all choices come from the ``random.Random`` passed
in and nothing reads the clock.

Format note: this module produces ``HardElement`` objects with ``cx/cy`` so the
shard writer can reuse the seed world's heatmap and offset code unchanged.
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .ui_world import ACTION_OF_ROLE, CANVAS, LABELS, ROLES, TASKS

SS = 2  # supersample: draw at 256x256, filter down to 128x128

# --------------------------------------------------------------------- themes
THEMES = {
    "light": {
        "desktop": (198, 203, 212), "window": (252, 253, 255), "chrome": (236, 239, 244),
        "titlebar": (222, 226, 234), "title_text": (38, 42, 50), "text": (30, 34, 42),
        "muted": (112, 118, 130), "border": (168, 174, 186), "accent": (36, 104, 236),
        "accent_text": (255, 255, 255), "disabled": (172, 176, 184), "disabled_bg": (228, 231, 236),
        "field": (255, 255, 255), "row_alt": (244, 246, 250), "danger": (200, 52, 52),
    },
    "dark": {
        "desktop": (22, 24, 30), "window": (40, 43, 51), "chrome": (33, 36, 43),
        "titlebar": (29, 31, 37), "title_text": (228, 232, 240), "text": (232, 236, 244),
        "muted": (146, 152, 166), "border": (74, 80, 94), "accent": (84, 152, 255),
        "accent_text": (10, 12, 16), "disabled": (98, 102, 112), "disabled_bg": (50, 53, 60),
        "field": (52, 56, 64), "row_alt": (45, 48, 56), "danger": (238, 104, 96),
    },
    "contrast": {
        "desktop": (0, 0, 0), "window": (14, 14, 14), "chrome": (6, 6, 6),
        "titlebar": (248, 216, 0), "title_text": (0, 0, 0), "text": (255, 255, 255),
        "muted": (248, 216, 0), "border": (248, 216, 0), "accent": (248, 216, 0),
        "accent_text": (0, 0, 0), "disabled": (118, 118, 118), "disabled_bg": (30, 30, 30),
        "field": (0, 0, 0), "row_alt": (22, 22, 22), "danger": (255, 72, 72),
    },
}
THEME_NAMES = ["light", "dark", "contrast"]

TITLE_WORDS = ["Files", "Settings", "Mail", "Editor", "Photos", "Store", "Terminal",
               "Backup", "Notes", "Calendar", "Vault", "Studio", "Player", "Sync"]
MENU_WORDS = ["File", "Edit", "View", "Insert", "Format", "Tools", "Window", "Help"]
LIST_HEADERS = ["Name", "Size", "Modified", "Kind", "Owner"]
LIST_STEMS = ["report", "budget", "notes", "draft", "export", "scan", "photo", "log",
              "backup", "invoice", "roster", "plan", "sheet", "chart", "audio", "clip"]
LIST_EXTS = ["txt", "py", "png", "csv", "log", "pdf", "zip"]
STATUS_WORDS = ["3 items selected", "Ready", "Saved 2 min ago", "12.4 GB free",
                "Connected", "Syncing 4 of 9", "UTF-8", "354 ms", "Offline"]


@lru_cache(maxsize=1)
def _font_dir() -> str:
    for p in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"):
        if os.path.isdir(p):
            return p
    try:
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=512)
def _font(px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    d = _font_dir()
    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return ImageFont.truetype(p, max(5, int(round(px * SS))))
    return ImageFont.load_default()


def text_w(s: str, px: float, bold: bool = False) -> float:
    return _font(px, bold).getlength(s) / SS


def truncate(s: str, px: float, max_w: float) -> str:
    if max_w <= 0:
        return ""
    if text_w(s, px) <= max_w:
        return s
    while s and text_w(s + "..", px) > max_w:
        s = s[:-1]
    return (s + "..") if s else ""


# -------------------------------------------------------------------- layout
@dataclass
class HardElement:
    role: str
    label: str
    x: float
    y: float
    w: float
    h: float
    section: str = "content"          # toolbar tabs sidebar content dialog
    state: str = "normal"             # normal disabled focused selected hover
    font_px: float = 7.0
    z: int = 0                        # 1 = dialog, drawn above everything
    show_text: bool = True            # icons/checkboxes draw a glyph, not a label

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
class HardLayout:
    elements: list[HardElement]
    theme: str
    title: str
    tier: int
    dpi: int
    cursor: tuple[int, int]
    window: tuple[float, float, float, float]
    bands: dict[str, float]                      # title/menu/toolbar/tabs/status heights
    menu_items: list[str]
    status: str
    sidebar_w: float = 0.0
    dialog: dict | None = None
    list_rows: list[list[str]] = field(default_factory=list)
    list_headers: list[str] = field(default_factory=list)
    noise: int = 0
    blur: float = 0.0


def _free_spot(els: list["HardElement"], x0: float, y0: float, x1: float, y1: float,
               w: float, h: float, rng: random.Random, tries: int = 16):
    """A position inside the box that does not overlap an existing widget."""
    for _ in range(tries):
        x = rng.uniform(x0, max(x0, x1 - w))
        y = rng.uniform(y0, max(y0, y1 - h))
        if not any(not (x + w < e.x or x > e.x + e.w or y + h < e.y or y > e.y + e.h)
                   for e in els):
            return x, y
    return None


def _pick_label(rng: random.Random, used: set[str]) -> str:
    pool = [l for l in LABELS if l not in used] or list(LABELS)
    lab = rng.choice(pool)
    used.add(lab)
    return lab


def make_layout(rng: random.Random, tier: int | None = None) -> HardLayout:
    if tier is None:
        tier = rng.choices([0, 1, 2], weights=[0.20, 0.45, 0.35])[0]
    theme = rng.choice(THEME_NAMES) if tier >= 1 else rng.choice(["light", "dark"])
    dpi = rng.choice([100, 125, 150, 200]) if tier >= 1 else rng.choice([100, 125])
    dfac = 0.90 + (dpi - 100) / 250.0          # 100 -> 0.90, 200 -> 1.30

    # Label size in the 128 px output: 5.6-7.5 px. On a 1920 px source screen
    # that is a 8-14 px font that has been downsampled; at 128 px it is the
    # smallest size whose glyphs stay distinct after antialiasing. DPI scales
    # the chrome, so 200% carries fewer, bigger widgets (the real DPI failure
    # mode), and tier raises crowding instead of shrinking text below legibility.
    base = {0: 7.2, 1: 6.5, 2: 5.9}[tier] * dfac
    base = max(5.2, min(7.8, base))

    wx = float(rng.randint(2, 7))
    wy = float(rng.randint(2, 7))
    ww = CANVAS - wx - rng.randint(2, 7)
    wh = CANVAS - wy - rng.randint(2, 7)

    bands = {"title": round(base * 1.35, 2)}
    if tier >= 1 or rng.random() < 0.45:
        bands["menu"] = round(base * 1.05, 2)
    if rng.random() < 0.85:
        bands["toolbar"] = round(base * 1.45, 2)
    if rng.random() < 0.55:
        bands["tabs"] = round(base * 1.2, 2)
    bands["status"] = round(base * 1.05, 2)

    used: set[str] = set()
    els: list[HardElement] = []
    menu_items = rng.sample(MENU_WORDS, k=rng.randint(3, 5))

    # ---- toolbar --------------------------------------------------------
    ty = wy + bands["title"] + bands.get("menu", 0.0)
    if "toolbar" in bands:
        bx = wx + 2.0
        xmax = wx + ww - 2.0
        n = {0: rng.randint(2, 3), 1: rng.randint(3, 5), 2: rng.randint(4, 6)}[tier]
        for _ in range(n):
            lab = _pick_label(rng, used)
            role = rng.choices(["button", "icon", "menu"], weights=[0.45, 0.35, 0.20])[0]
            if role == "icon":
                w = bands["toolbar"] - 2.0
            else:
                w = text_w(lab.capitalize(), base) + 4.0
            if bx + w > xmax:
                break
            state = "disabled" if rng.random() < 0.10 else rng.choices(
                ["normal", "hover", "focused"], weights=[0.75, 0.15, 0.10])[0]
            els.append(HardElement(role, lab, bx, ty + 1.0, w, bands["toolbar"] - 2.0,
                                   section="toolbar", state=state, font_px=base,
                                   show_text=(role != "icon")))
            bx += w + rng.uniform(1.0, 2.5)
        if rng.random() < 0.55:
            lab = _pick_label(rng, used)
            fw = min(max(14.0, ww * 0.26), 26.0)
            fx = wx + ww - fw - 2.0
            if fx > bx + 1.0:
                els.append(HardElement("field", lab, fx, ty + 1.2, fw, bands["toolbar"] - 2.4,
                                       section="toolbar", font_px=base * 0.9,
                                       state="normal"))

    # ---- tab strip ------------------------------------------------------
    ky = ty + bands.get("toolbar", 0.0)
    if "tabs" in bands:
        tx = wx + 1.5
        xmax = wx + ww - 1.5
        for i in range(rng.randint(2, 4)):
            lab = _pick_label(rng, used)
            tw = text_w(lab.capitalize(), base) + 4.5
            if tx + tw > xmax:
                break
            state = "selected" if (i == 0 and rng.random() < 0.7) else "normal"
            els.append(HardElement("tab", lab, tx, ky + 0.5, tw, bands["tabs"] - 1.0,
                                   section="tabs", state=state, font_px=base))
            tx += tw + 0.7

    # ---- body -----------------------------------------------------------
    body_top = ky + bands.get("tabs", 0.0) + 1.0
    body_bottom = wy + wh - bands["status"] - 0.5
    body_h = body_bottom - body_top
    sidebar_w = 0.0
    if rng.random() < (0.5 if tier >= 1 else 0.2):
        sidebar_w = min(max(18.0, ww * rng.uniform(0.24, 0.32)), 30.0)
        sy = body_top + 0.5
        item_h = max(6.5, base * 1.35)
        i = 0
        while sy + item_h <= body_bottom and i < 7:
            lab = _pick_label(rng, used)
            state = "selected" if rng.random() < 0.22 else (
                "disabled" if rng.random() < 0.10 else "normal")
            els.append(HardElement("menu", lab, wx + 1.0, sy, sidebar_w - 2.2, item_h - 0.7,
                                   section="sidebar", state=state, font_px=base * 0.95))
            sy += item_h
            i += 1

    cx0 = wx + sidebar_w + 1.5
    cw = wx + ww - 1.5 - cx0
    cy0 = body_top
    ch = body_h

    list_rows: list[list[str]] = []
    list_headers: list[str] = []
    kind = rng.choices(["list", "settings", "form"], weights=[0.40, 0.35, 0.25])[0]

    if kind == "list":
        ncols = rng.randint(1, 3)
        list_headers = rng.sample(LIST_HEADERS, k=ncols)
        hdr_h = max(5.0, base * 0.95)
        row_h = max(6.0, base * 1.25)
        nrows = max(1, int((ch - hdr_h) // row_h))
        role = rng.choices(["checkbox", "icon", "button"], weights=[0.45, 0.35, 0.20])[0]
        for r in range(nrows):
            row = [f"{rng.choice(LIST_STEMS)}.{rng.choice(LIST_EXTS)}",
                   f"{rng.randint(1, 900)} KB",
                   f"{rng.randint(1, 28)} d"][:ncols]
            list_rows.append(row)
            if rng.random() < 0.8:
                lab = _pick_label(rng, used)
                state = "disabled" if rng.random() < 0.12 else (
                    "selected" if rng.random() < 0.18 else "normal")
                ry = cy0 + hdr_h + r * row_h + 0.5
                if role == "button":
                    ew = min(text_w(lab.capitalize(), base * 0.9) + 3.5, cw * 0.42)
                    ex = cx0 + cw - ew - 0.5
                else:
                    ew = max(4.5, row_h - 1.6)
                    ex = cx0 + 0.8
                els.append(HardElement(role, lab, ex, ry, ew, row_h - 1.2, section="content",
                                       state=state, font_px=base * 0.9,
                                       show_text=(role == "button")))
    elif kind == "settings":
        rows = max(2, int(ch // (base * 1.9)))
        rh = ch / rows
        for r in range(min(rows, 7)):
            lab = _pick_label(rng, used)
            control = rng.choices(["checkbox", "field", "slider", "button"],
                                  weights=[0.35, 0.28, 0.17, 0.20])[0]
            cwid = {"checkbox": max(4.5, base * 0.75), "field": min(cw * 0.42, 26.0),
                    "slider": min(cw * 0.44, 26.0),
                    "button": min(text_w(lab.capitalize(), base * 0.9) + 4.0, cw * 0.45)}[control]
            els.append(HardElement(control, lab, cx0 + cw - cwid - 1.0,
                                   cy0 + r * rh + (rh - max(5.0, base)) / 2,
                                   cwid, max(5.0, base), section="content",
                                   state="disabled" if rng.random() < 0.10 else "normal",
                                   font_px=base * 0.9, show_text=(control == "button")))
    else:  # form
        nfields = rng.randint(1, 3)
        fh = max(6.0, base * 1.1)
        for r in range(nfields):
            lab = _pick_label(rng, used)
            els.append(HardElement("field", lab, cx0 + 0.8,
                                   cy0 + r * (fh + 2.2), cw - 1.6, fh, section="content",
                                   font_px=base * 0.9,
                                   state="focused" if (r == 0 and rng.random() < 0.3) else "normal"))
        by = cy0 + nfields * (fh + 2.2) + 1.5
        bw_total = 0.0
        for _ in range(rng.randint(1, 2)):
            lab = _pick_label(rng, used)
            bw = min(text_w(lab.capitalize(), base * 0.9) + 4.0, cw * 0.42)
            if by + fh > body_bottom or bw_total + bw > cw:
                break
            els.append(HardElement("button", lab, cx0 + bw_total + 0.8, by, bw, fh,
                                   section="content", font_px=base * 0.9))
            bw_total += bw + 1.5

    # ---- guaranteed slider: drag is a rare action class otherwise ---------
    if kind == "settings" or rng.random() < 0.72:
        lab = _pick_label(rng, used)
        sw, sh = min(cw * 0.4, 24.0), max(5.0, base * 0.8)
        spot = _free_spot(els, cx0 + 0.8, cy0 + 0.5, cx0 + cw - 1.0, body_bottom - sh - 0.5,
                          sw, sh, rng)
        if spot:
            els.append(HardElement("slider", lab, spot[0], spot[1], sw, sh,
                                   section="content", font_px=base * 0.9))

    # ---- dialog ---------------------------------------------------------
    dialog = None
    if rng.random() < (0.12 if tier == 0 else 0.42 if tier == 1 else 0.55):
        dw = min(ww - 6.0, ww * rng.uniform(0.68, 0.86))
        dh = min(wh - 8.0, wh * rng.uniform(0.40, 0.54))
        dx = wx + (ww - dw) / 2 + rng.uniform(-1.5, 1.5)
        dy = wy + (wh - dh) / 2 + rng.uniform(-1.5, 1.5)
        dx = min(max(dx, wx + 2.0), wx + ww - dw - 2.0)
        dy = min(max(dy, wy + bands["title"] + 1.0), wy + wh - dh - 1.5)
        title = rng.choice(["Confirm", "Unsaved changes", "Warning", "Delete file?",
                            "Overwrite?", "Save changes"])
        nb = rng.randint(2, 3)
        labs = [_pick_label(rng, used) for _ in range(nb)]
        dialog = {"rect": (dx, dy, dw, dh), "title": title, "buttons": labs}
        bh = max(6.0, base * 1.15)
        gap = 2.0
        bw = (dw - 4.0 - gap * (nb - 1)) / nb
        for i, lab in enumerate(labs):
            els.append(HardElement("button", lab, dx + 2.0 + i * (bw + gap),
                                   dy + dh - bh - 2.0, bw, bh, section="dialog",
                                   font_px=base * 0.92, z=1))

    # ---- cursor: never on top of the widgets it must not cover ----------
    cursor = (6, 6)
    for _ in range(30):
        cx = rng.randint(2, CANVAS - 8)
        cy = rng.randint(2, CANVAS - 10)
        if not any(e.contains(cx, cy, margin=2.5) for e in els):
            cursor = (cx, cy)
            break

    return HardLayout(
        elements=els, theme=theme, title=rng.choice(TITLE_WORDS), tier=tier, dpi=dpi,
        cursor=cursor, window=(wx, wy, ww, wh), bands=bands, menu_items=menu_items,
        status=rng.choice(STATUS_WORDS), sidebar_w=sidebar_w, dialog=dialog,
        list_rows=list_rows, list_headers=list_headers,
        noise={0: 0, 1: rng.randint(3, 8), 2: rng.randint(5, 14)}[tier],
        blur={0: 0.0, 1: 0.25, 2: rng.choice([0.0, 0.4])}[tier],
    )


# ------------------------------------------------------------------ rendering
def _s(v: float) -> int:
    return int(round(v * SS))


def render(layout: HardLayout, size: int = CANVAS) -> np.ndarray:
    th = THEMES[layout.theme]
    W = size * SS
    img = Image.new("RGB", (W, W), th["desktop"])
    d = ImageDraw.Draw(img)
    wx, wy, ww, wh = layout.window
    b = layout.bands
    f_title = _font(max(4.5, b["title"] * 0.62), True)
    f_menu = _font(max(4.0, b.get("menu", 6.0) * 0.62))

    # window shell
    d.rectangle([_s(wx), _s(wy), _s(wx + ww), _s(wy + wh)], fill=th["window"],
                outline=th["border"], width=SS)
    d.rectangle([_s(wx), _s(wy), _s(wx + ww), _s(wy + b["title"])], fill=th["titlebar"])
    d.text((_s(wx + 2.0), _s(wy + b["title"] * 0.16)),
           truncate(layout.title, b["title"] * 0.62, ww - 14.0), font=f_title,
           fill=th["title_text"])
    for i, col in enumerate([(236, 96, 84), (240, 190, 74), (110, 196, 112)]):
        cxp = wx + ww - 3.0 - i * 3.4
        d.ellipse([_s(cxp - 1.7), _s(wy + b["title"] * 0.3), _s(cxp),
                   _s(wy + b["title"] * 0.3 + 1.7)],
                  fill=col if layout.theme != "contrast" else th["border"])

    y = wy + b["title"]
    if "menu" in b:
        d.rectangle([_s(wx), _s(y), _s(wx + ww), _s(y + b["menu"])], fill=th["chrome"])
        mx = wx + 2.0
        for item in layout.menu_items:
            if mx + f_menu.getlength(item) / SS > wx + ww - 2:
                break
            d.text((_s(mx), _s(y + b["menu"] * 0.12)), item, font=f_menu, fill=th["text"])
            mx += f_menu.getlength(item) / SS + 3.0
        y += b["menu"]
    if "toolbar" in b:
        d.rectangle([_s(wx), _s(y), _s(wx + ww), _s(y + b["toolbar"])], fill=th["chrome"])
        y += b["toolbar"]
    if "tabs" in b:
        d.rectangle([_s(wx), _s(y), _s(wx + ww), _s(y + b["tabs"])], fill=th["chrome"])
        y += b["tabs"]

    body_top = y
    body_bottom = wy + wh - b["status"]

    # sidebar panel
    if layout.sidebar_w:
        d.rectangle([_s(wx + 0.5), _s(body_top + 0.5), _s(wx + layout.sidebar_w),
                     _s(body_bottom - 0.5)], fill=th["chrome"])
    # list rows (decoration behind the widgets)
    if layout.list_rows:
        hdr_h = max(5.0, 6.0)
        row_h = (body_bottom - body_top - hdr_h) / max(1, len(layout.list_rows))
        rx = wx + layout.sidebar_w + 1.0
        rw = wx + ww - 1.0 - rx
        hf = _font(max(4.5, row_h * 0.55), True)
        cf = _font(max(4.0, row_h * 0.5))
        col_w = max(14.0, rw * 0.42)
        for i, htxt in enumerate(layout.list_headers):
            d.text((_s(rx + 1.0 + i * col_w), _s(body_top + 0.4)), htxt, font=hf, fill=th["muted"])
        for r, row in enumerate(layout.list_rows):
            ry = body_top + hdr_h + r * row_h
            if r % 2 == 1:
                d.rectangle([_s(rx), _s(ry), _s(rx + rw), _s(ry + row_h)], fill=th["row_alt"])
            for i, cell in enumerate(row):
                d.text((_s(rx + 1.0 + i * col_w), _s(ry + row_h * 0.12)),
                       truncate(cell, row_h * 0.5, col_w - 1.5), font=cf, fill=th["muted"])

    # status bar
    d.rectangle([_s(wx), _s(body_bottom), _s(wx + ww), _s(wy + wh)], fill=th["chrome"])
    d.text((_s(wx + 1.5), _s(body_bottom + 0.3)), truncate(layout.status, 4.5, ww - 3),
           font=_font(max(4.0, b["status"] * 0.6)), fill=th["muted"])

    # widgets
    for el in layout.elements:
        if el.z == 0:
            _draw_element(d, el, th, layout)

    # modal dialog
    if layout.dialog:
        dx, dy, dw, dh = layout.dialog["rect"]
        overlay = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle([0, 0, W, W], fill=(0, 0, 0, 84))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([_s(dx), _s(dy), _s(dx + dw), _s(dy + dh)], fill=th["window"],
                    outline=th["border"], width=2 * SS)
        df = _font(max(4.5, b["title"] * 0.66), True)
        d.rectangle([_s(dx), _s(dy), _s(dx + dw), _s(dy + b["title"])], fill=th["chrome"])
        d.text((_s(dx + 2.0), _s(dy + b["title"] * 0.16)), layout.dialog["title"],
               font=df, fill=th["text"])
        body_f = _font(max(4.0, b["title"] * 0.52))
        msg = ("This cannot be undone." if "Delete" in layout.dialog["title"]
               else "Do you want to continue?")
        d.text((_s(dx + 2.5), _s(dy + b["title"] + 2.0)), msg, font=body_f, fill=th["muted"])
        for el in layout.elements:
            if el.z == 1:
                _draw_element(d, el, th, layout)

    # cursor
    cxp, cyp = layout.cursor
    pts = [(_s(cxp), _s(cyp)), (_s(cxp), _s(cyp + 7.0)), (_s(cxp + 1.7), _s(cyp + 5.1)),
           (_s(cxp + 2.8), _s(cyp + 7.4)), (_s(cxp + 3.9), _s(cyp + 6.8)),
           (_s(cxp + 2.8), _s(cyp + 4.6)), (_s(cxp + 4.8), _s(cyp + 4.6))]
    d.polygon(pts, fill=(255, 255, 255), outline=(16, 16, 16))

    # downsample + sensor degradation
    img = img.resize((size, size), Image.BOX if layout.tier == 2 else Image.LANCZOS)
    if layout.blur:
        img = img.filter(ImageFilter.GaussianBlur(layout.blur))
    if layout.noise >= 8:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(40, 96 - layout.noise * 3), subsampling=2)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(img, dtype=np.uint8)
        if layout.noise:
            rng = np.random.default_rng((cxp << 16) ^ (cyp << 3) ^ layout.noise
                                        ^ (layout.tier * 7919) ^ (layout.dpi * 31))
            noise = rng.integers(-layout.noise, layout.noise + 1, size=arr.shape[:2] + (1,))
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return arr


def _draw_element(d: ImageDraw.ImageDraw, el: HardElement, th: dict, layout: HardLayout) -> None:
    fg = th["text"]
    if el.state == "disabled":
        fg = th["disabled"]
    box = [_s(el.x), _s(el.y), _s(el.x + el.w), _s(el.y + el.h)]

    def label(text: str, px: float, color, bold=False, center=True):
        t = truncate(text, px, el.w - 1.2)
        f = _font(px, bold)
        tw = f.getlength(t) / SS
        tx = el.x + (el.w - tw) / 2 if center else el.x + 0.8
        ty = el.y + (el.h - px * 0.78) / 2
        d.text((_s(tx), _s(ty)), t, font=f, fill=color)

    if el.role == "button":
        danger = any(k in el.label for k in ("delete", "cancel"))
        fill = th["disabled_bg"] if el.state == "disabled" else (
            th["danger"] if danger else th["accent"])
        d.rounded_rectangle(box, radius=max(1, _s(0.8)), fill=fill, outline=th["border"], width=1)
        label(el.label, el.font_px, th["accent_text"] if el.state != "disabled" else th["disabled"])
        if el.state == "focused":
            d.rectangle([box[0] - SS // 2, box[1] - SS // 2, box[2] + SS // 2, box[3] + SS // 2],
                        outline=th["accent"], width=SS)
    elif el.role == "field":
        d.rectangle(box, fill=th["field"], outline=th["accent"] if el.state == "focused"
                    else th["border"], width=1)
        label(el.label, el.font_px, th["muted"], center=False)
    elif el.role == "checkbox":
        d.rectangle(box, fill=th["field"],
                    outline=th["accent"] if el.state != "disabled" else th["disabled"], width=1)
        if el.state == "selected":
            d.line([_s(el.x + el.w * 0.18), _s(el.y + el.h * 0.55),
                    _s(el.x + el.w * 0.42), _s(el.y + el.h * 0.82),
                    _s(el.x + el.w * 0.86), _s(el.y + el.h * 0.2)],
                   fill=th["accent"], width=SS + 1)
    elif el.role == "tab":
        fill = th["window"] if el.state == "selected" else th["chrome"]
        d.rectangle(box, fill=fill, outline=th["border"], width=1)
        if el.state == "selected":
            d.line([box[0], box[3] - 1, box[2], box[3] - 1], fill=th["accent"], width=2 * SS)
        label(el.label, el.font_px, fg)
    elif el.role == "icon":
        d.rectangle(box, fill=th["accent"] if el.state == "selected" else th["field"],
                    outline=th["border"], width=1)
        inset = min(el.w, el.h) * 0.24
        d.rectangle([_s(el.x + inset), _s(el.y + inset), _s(el.x + el.w - inset),
                     _s(el.y + el.h - inset)],
                    outline=th["accent_text"] if el.state == "selected" else
                    (th["text"] if el.state != "disabled" else th["disabled"]), width=SS)
    elif el.role == "menu":
        if el.state == "selected":
            d.rectangle(box, fill=th["accent"])
            fg = th["accent_text"]
        elif el.state == "hover":
            d.rectangle(box, fill=th["row_alt"])
        label(el.label.capitalize(), el.font_px, fg, center=False)
    elif el.role == "slider":
        ty = el.y + el.h / 2
        d.line([_s(el.x + 1.0), _s(ty), _s(el.x + el.w - 1.0), _s(ty)],
               fill=th["muted"] if el.state == "disabled" else th["accent"], width=max(1, SS // 2))
        kx = el.x + 1.0 + (el.w - 2.0) * 0.55
        d.ellipse([_s(kx - 1.6), _s(ty - 1.6), _s(kx + 1.6), _s(ty + 1.6)],
                  fill=th["accent"] if el.state != "disabled" else th["disabled"])


# --------------------------------------------------------------- task sampling
HARD_TASKS = {
    "button": ["click the {label} button", "press {label}", "hit the {label} button",
               "confirm with {label}"],
    "field": ["type into the {label} field", "click the {label} input box", "focus {label}"],
    "checkbox": ["toggle the {label} checkbox", "check {label}", "turn on {label}"],
    "tab": ["open the {label} tab", "switch to {label}", "click the {label} tab"],
    "icon": ["click the {label} icon", "open {label}", "select the {label} icon"],
    "menu": ["choose {label} from the menu", "click the {label} menu item", "pick {label}"],
    "slider": ["drag the {label} slider", "move the {label} slider", "adjust {label}"],
}
RELATIVE = [
    ("the {role} in the toolbar", "toolbar"),
    ("the {role} in the sidebar", "sidebar"),
    ("the {role} in the dialog", "dialog"),
    ("the {role} above the list", "toolbar"),
]


@dataclass
class HardExample:
    layout: HardLayout
    task: str
    target: HardElement
    action: str
    click: tuple[int, int]


def sample_example(rng: random.Random, tier: int | None = None) -> HardExample:
    for _ in range(8):
        layout = make_layout(rng, tier)
        live = [e for e in layout.elements if e.state != "disabled" and e.w >= 4 and e.h >= 4]
        if not live:
            continue
        dialog_live = [e for e in live if e.section == "dialog"]
        if dialog_live and rng.random() < 0.5:
            target = rng.choice(dialog_live)      # the modal is what you act on
        else:
            target = rng.choice(live)
        task = rng.choice(HARD_TASKS.get(target.role, ["click {label}"])).format(label=target.label)
        # a relative reference may only name a band that exists in this layout
        sections = {e.section for e in live}
        usable = [r for r in RELATIVE if r[1] in sections and r[1] == target.section]
        if usable and rng.random() < 0.20:
            task = f"{rng.choice(usable)[0]}, labelled {target.label}"
        if rng.random() < 0.10:
            task = task.lower()
        action = ACTION_OF_ROLE[target.role]
        px = int(round(rng.uniform(target.x + target.w * 0.3, target.x + target.w * 0.7)))
        py = int(round(rng.uniform(target.y + target.h * 0.3, target.y + target.h * 0.7)))
        return HardExample(layout, task, target, action,
                           (min(max(px, 0), CANVAS - 1), min(max(py, 0), CANVAS - 1)))
    raise RuntimeError("could not build a usable hard layout")


# ------------------------------------------------------------ shard targeting
def heatmap_vec(ex: HardExample, grid: int = 32, sigma: float = 0.9) -> np.ndarray:
    cell = CANVAS / grid
    gx, gy = ex.target.cx / cell - 0.5, ex.target.cy / cell - 0.5
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    hm = np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return (hm / hm.sum()).astype(np.float32)


def target_offset(ex: HardExample, grid: int = 32) -> tuple[int, list[float]]:
    """Cell index + offset scaled by HALF a cell, so |offset| > 1 only at the edge.

    The seed world divides by 0.45 instead, which saturates 52% of its samples;
    a regressor trained there learns to predict +-1 and stops learning sub-cell
    position. Here the full cell is used.
    """
    cell = CANVAS / grid
    cx, cy = ex.target.cx / cell, ex.target.cy / cell
    ix, iy = min(int(cx), grid - 1), min(int(cy), grid - 1)
    dx = (cx - (ix + 0.5)) / 0.5
    dy = (cy - (iy + 0.5)) / 0.5
    return iy * grid + ix, [max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy))]
