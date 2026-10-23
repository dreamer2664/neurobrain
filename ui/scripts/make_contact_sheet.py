#!/usr/bin/env python3
"""Build the Atelier release contact sheet from deterministic browser captures."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
FONT_PATH = ROOT / "assets" / "fonts" / "Geist-Variable.ttf"
W, H = 1920, 1010
im = Image.new("RGB", (W, H), "#f4f3f1")
d = ImageDraw.Draw(im)


def font(size, weight=400):
    face = ImageFont.truetype(str(FONT_PATH), size)
    try:
        face.set_variation_by_axes([weight])
    except (AttributeError, OSError, ValueError):
        pass
    return face


def text(x, y, value, size, fill="#20211f", weight=400, anchor=None):
    d.text((x, y), value, font=font(size, weight), fill=fill, anchor=anchor)


def rounded_paste(source, box, radius=6):
    source = ImageOps.fit(source.convert("RGB"), (box[2], box[3]), Image.Resampling.LANCZOS)
    mask = Image.new("L", source.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, source.width - 1, source.height - 1), radius=radius, fill=255)
    im.paste(source, (box[0], box[1]), mask)


logo = Image.open(ROOT / "assets" / "icons" / "icon-192.png").convert("RGBA").resize((46, 46), Image.Resampling.LANCZOS)
im.paste(logo, (56, 40), logo)
text(118, 37, "Neurobrain Atelier", 34, weight=600)
text(119, 83, "Four palettes. One responsive local workspace.", 17, "#62645f")
text(1864, 48, "DESIGN RELEASE 2.0.0", 14, "#20211f", 550, "ra")
text(1864, 78, "23 OCTOBER 2026", 13, "#62645f", 400, "ra")
d.line((56, 126, 1864, 126), fill="#d9d6d1", width=1)

panels = [
    {"name":"Linen", "description":"Stone and forest", "desktop":"01-linen-desktop.png", "mobile":"02-linen-mobile.png", "ground":"#f4f3f1", "ink":"#20211f", "accent":"#2f6f59", "border":"#dfdcd7"},
    {"name":"Mist", "description":"Blue-grey and cobalt", "desktop":"03-mist-desktop.png", "mobile":"04-mist-mobile.png", "ground":"#f2f4f7", "ink":"#182028", "accent":"#1d5fbf", "border":"#dce2e9"},
    {"name":"Rose", "description":"Neutral rose and raspberry", "desktop":"05-rose-desktop.png", "mobile":"06-rose-mobile.png", "ground":"#f6f3f4", "ink":"#282126", "accent":"#9c3f62", "border":"#e2d9de"},
    {"name":"Ink", "description":"Neutral dark and sage", "desktop":"07-ink-desktop.png", "mobile":"08-ink-mobile.png", "ground":"#101111", "ink":"#f5f5f2", "accent":"#8fb8a2", "border":"#303232"},
]

for index, panel in enumerate(panels):
    x, y, pw, ph = 56 + index * 456, 158, 440, 716
    d.rounded_rectangle((x, y, x + pw, y + ph), radius=10, fill="#ffffff", outline=panel["border"], width=1)
    d.rounded_rectangle((x, y, x + pw, y + 4), radius=2, fill=panel["accent"])
    text(x + 20, y + 25, panel["name"], 24, weight=600)
    text(x + 20, y + 61, panel["description"], 14, "#62645f")
    for swatch, colour in enumerate((panel["ground"], panel["ink"], panel["accent"])):
        sx = x + pw - 100 + swatch * 27
        d.rectangle((sx, y + 31, sx + 20, y + 51), fill=colour, outline=panel["border"])

    dx, dy, dw, dh = x + 18, y + 100, 404, 253
    rounded_paste(Image.open(SHOTS / panel["desktop"]), (dx, dy, dw, dh))
    d.rounded_rectangle((dx, dy, dx + dw, dy + dh), radius=6, outline=panel["border"], width=1)

    mx, my, mw, mh = x + 318, y + 205, 104, 225
    d.rounded_rectangle((mx - 3, my - 3, mx + mw + 3, my + mh + 3), radius=12, fill=panel["ink"])
    rounded_paste(Image.open(SHOTS / panel["mobile"]), (mx, my, mw, mh), 9)

    text(x + 20, y + 450, "Palette values", 13, "#62645f", 550)
    rows = (("Ground", panel["ground"]), ("Text", panel["ink"]), ("Data accent", panel["accent"]))
    for row, (label, colour) in enumerate(rows):
        ry = y + 480 + row * 34
        d.rounded_rectangle((x + 20, ry + 2, x + 36, ry + 18), radius=3, fill=colour, outline=panel["border"])
        text(x + 48, ry, label, 14)
        text(x + pw - 20, ry, colour, 14, "#62645f", 400, "ra")

    d.line((x + 20, y + 580, x + pw - 20, y + 580), fill=panel["border"], width=1)
    text(x + 20, y + 602, "Switch in Settings → Colour theme", 15, weight=550)
    text(x + 20, y + 632, "The choice persists on this device.", 14, "#62645f")
    text(x + 20, y + 676, "DESKTOP 1440 × 900   /   MOBILE 390 × 844", 11, "#62645f", 550)


d.line((56, 916, 1864, 916), fill="#d9d6d1", width=1)
facts = [
    (56, "12 HIGH-RESOLUTION CAPTURES"),
    (475, "99 BROWSER CHECKS"),
    (845, "ZERO EXTERNAL RUNTIME ASSETS"),
    (1310, "PWA + BRAINBOX ADAPTER"),
]
for x, label in facts:
    d.ellipse((x, 956, x + 7, 963), fill="#2f6f59")
    text(x + 19, 950, label, 13, "#62645f", 550)

output = SHOTS / "00-design-overview.jpg"
im.save(output, quality=94, subsampling=0, optimize=True)
print(output)
