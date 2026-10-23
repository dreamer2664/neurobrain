#!/usr/bin/env python3
"""Build the release contact sheet from the deterministic browser captures."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
FONTS = ROOT / "assets" / "fonts"
W, H = 1920, 1040

im = Image.new("RGB", (W, H), "#080a0f")
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
g.ellipse((-250, -420, 800, 420), fill=(114, 217, 247, 30))
g.ellipse((1250, -350, 2150, 380), fill=(195, 244, 123, 24))
g.ellipse((600, 750, 1400, 1300), fill=(165, 139, 245, 13))
glow = glow.filter(ImageFilter.GaussianBlur(150))
im = Image.alpha_composite(im.convert("RGBA"), glow)
d = ImageDraw.Draw(im)

def font(size, mono=False):
    path = FONTS / ("IBMPlexMono-Regular.ttf" if mono else "SpaceGrotesk-Variable.ttf")
    return ImageFont.truetype(str(path), size)

def text(x, y, value, size, fill, mono=False):
    d.text((x, y), value, font=font(size, mono), fill=fill)

def rounded_paste(source, box, radius):
    source = ImageOps.fit(source.convert("RGB"), (box[2], box[3]), Image.Resampling.LANCZOS)
    mask = Image.new("L", source.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, source.width - 1, source.height - 1), radius=radius, fill=255)
    im.paste(source, (box[0], box[1]), mask)

logo = Image.open(ROOT / "assets" / "icons" / "icon-192.png").resize((58, 58), Image.Resampling.LANCZOS)
im.paste(logo, (60, 44), logo if logo.mode == "RGBA" else None)
text(134, 42, "NEUROBRAIN", 43, "#f3f6f4")
text(136, 96, "THREE DIRECTIONS  /  ONE LOCAL BRAIN", 14, "#8d98a3", mono=True)

d.rounded_rectangle((1530, 52, 1858, 100), radius=12, fill="#10151b", outline="#273029", width=1)
d.ellipse((1550, 70, 1558, 78), fill="#c3f47b")
text(1572, 65, "UI DESIGN RELEASE 1.0.0", 13, "#c3f47b", mono=True)
text(60, 147, "A responsive command center for a brain built from scratch.", 19, "#a8b1b9")
d.text((1858, 151), "23 OCT 2026", font=font(12, mono=True), fill="#66717c", anchor="ra")
d.line((60, 181, 1860, 181), fill="#20262d", width=1)

panels = [
    ("01", "PULSE", "Cinematic neural command center", "01-pulse-desktop.png", "02-pulse-mobile.png", "#c3f47b"),
    ("02", "ATELIER", "Calm, warm cognitive workspace", "03-atelier-desktop.png", "04-atelier-mobile.png", "#a993ff"),
    ("03", "TERMINAL", "Explicit technical operator console", "05-terminal-desktop.png", "06-terminal-mobile.png", "#b6ff46"),
]

for i, (number, name, desc, desktop, mobile, accent) in enumerate(panels):
    x, y, pw, ph = 60 + i * 608, 211, 580, 688
    d.rounded_rectangle((x, y, x + pw, y + ph), radius=24, fill="#0e1218", outline="#242b32", width=1)
    d.rounded_rectangle((x + 22, y + 20, x + 63, y + 42), radius=6, fill="#1a2119" if i != 1 else "#211d31")
    text(x + 32, y + 23, number, 12, accent, mono=True)
    text(x + 22, y + 55, name, 32, "#f3f6f4")
    text(x + 22, y + 101, desc, 14, "#8d98a3")
    # Soft shadow behind the desktop frame.
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x + 24, y + 145, x + 560, y + 483), radius=14, fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(13))
    im = Image.alpha_composite(im, shadow)
    d = ImageDraw.Draw(im)
    rounded_paste(Image.open(SHOTS / desktop), (x + 20, y + 136, 540, 338), 12)
    d.rounded_rectangle((x + 20, y + 136, x + 560, y + 474), radius=12, outline="#34404a", width=1)
    # Floating phone capture.
    mx, my, mw, mh = x + 435, y + 337, 120, 260
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((mx - 4, my + 6, mx + mw + 8, my + mh + 12), radius=21, fill=(0, 0, 0, 155))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    im = Image.alpha_composite(im, shadow)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((mx - 4, my - 4, mx + mw + 4, my + mh + 4), radius=20, fill=accent)
    rounded_paste(Image.open(SHOTS / mobile), (mx, my, mw, mh), 15)
    text(x + 22, y + 614, "DESKTOP 1440×900  /  MOBILE 390×844", 11, "#707c86", mono=True)
    d.ellipse((x + 23, y + 653, x + 29, y + 659), fill=accent)
    text(x + 39, y + 647, "RESPONSIVE  ·  LOCAL-FIRST  ·  INTERACTIVE", 11, "#a8b1b9", mono=True)

# Footer is a compact release manifest, not marketing noise.
d.line((60, 934, 1860, 934), fill="#20262d", width=1)
chips = [("8 HI-RES SCREENSHOTS", 60, 315), ("26 BROWSER CHECKS", 390, 270), ("3 COMPLETE THEMES", 690, 270), ("ZERO EXTERNAL ASSETS", 990, 300), ("PWA + BRAINBOX ADAPTER", 1320, 335)]
for label, x, width in chips:
    d.rounded_rectangle((x, 965, x + width, 1006), radius=10, fill="#0f141a", outline="#232b32")
    d.ellipse((x + 15, 981, x + 21, 987), fill="#c3f47b")
    text(x + 31, 977, label, 11, "#9ca7af", mono=True)

output = SHOTS / "00-design-overview.jpg"
im.convert("RGB").save(output, quality=94, subsampling=0, optimize=True)
print(output)
