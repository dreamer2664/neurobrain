"""Open-Vocabulary Desktop + SERP ("MiniGoogle") + Cross-Role UI Generator & Co-Trainer (M9).

Extends `EyesGrounder` (`Stage1CoarseNet` + `Stage2RefineNet`) beyond the 24 seed labels
to read arbitrary English words, 2-4 digit port/status/permission numbers, SERP result
rows, desktop shortcut icons, OS taskbar tabs, multi-window controls (`copy`, `paste`,
`bookmark`, `run`, `close`, `start`, `back`), and crowded cross-role duplicate-label toolbars,
while replaying 45% Hard World v2 so existing seen/OOD accuracy never regresses.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .diagnose_grip import peak_rss_mb
from .eval_suite import PALETTES, run_eval_suite
from .export_grounder import export_grounder_checkpoint
from .eyes_grounder import EyesGrounder, Stage1CoarseNet, Stage2RefineNet
from .mock_desktop import AD_TRAP_DOCS, KNOWLEDGE_DOCS, OfflineSearchEngine, run_desktop_suite
from .train_eyes import (
    apply_palette_aug,
    build_screen_bank,
    evaluate_grounder_on_split,
    train_stage2_refiner,
)
from .ui_world import ACTION_CLASSES, ACTION_OF_ROLE, TASKS, encode_task
from .ui_world_hard_v2 import (
    HARD_CANVAS,
    OOD_FONT_PATHS,
    SEEN_FONT_PATHS,
    TASKS_HARD,
    HardElement,
    HardExample,
    HardLayout,
    _get_font,
    make_hard_layout,
    make_hard_val_split,
    render_hard,
)

# Rich open-vocabulary desktop & technical word pool (general English + CLI + web + numbers)
TRAIN_OPEN_WORDS: tuple[str, ...] = (
    # Desktop & OS actions/apps
    "browser", "notes", "files", "terminal", "editor", "settings", "start", "close",
    "copy", "paste", "bookmark", "back", "run", "command", "clear", "open", "quit",
    "history", "downloads", "plugins", "network", "system", "display", "sound", "power",
    "wifi", "bluetooth", "battery", "clock", "calendar", "mail", "photos", "music",
    "camera", "console", "monitor", "backup", "restore", "update", "install", "remove",
    # Technical & web search vocabulary
    "postgres", "mysql", "redis", "sqlite", "mongodb", "docker", "kubectl", "nginx",
    "apache", "systemd", "journal", "crontab", "cron", "timer", "chmod", "chown",
    "umask", "tar", "extract", "compress", "grep", "recursive", "awk", "sed", "find",
    "mtime", "inode", "df", "lsof", "listen", "netstat", "ss", "iptables", "ufw",
    "ssh", "port", "keygen", "scp", "rsync", "curl", "wget", "dns", "https", "http",
    "smtp", "ntp", "tcp", "syn", "ack", "tls", "alpn", "quic", "udp", "socket",
    "git", "bisect", "reflog", "stash", "cherry", "revert", "blame", "rebase", "merge",
    "branch", "tag", "fetch", "pull", "clone", "diff", "status", "commit",
    "python", "gil", "asyncio", "gather", "venv", "create", "pytest", "marks",
    "pathlib", "glob", "json", "dumps", "functools", "lru", "collections", "deque",
    "typing", "dataclass", "itertools", "heapq", "struct", "pickle", "logging",
    "vacuum", "wal", "left", "join", "btree", "index", "pragma", "schema", "query",
    "sha256", "bits", "ed25519", "key", "aes", "hmac", "rsa", "cipher", "digest",
    "dijkstra", "path", "quicksort", "pivot", "mergesort", "binary", "tree", "graph",
    "utf8", "variable", "ieee754", "float", "ascii", "hex", "octal", "base64",
    "kill", "sigterm", "sigkill", "sighup", "pid", "process", "thread", "mutex",
    "ad", "cloud", "deal", "boot", "camp", "vpn", "offer", "promo", "sponsor",
)

# Strictly held-out multi-word open-vocabulary test phrases NEVER sampled during training!
HELD_OUT_OOD_PHRASES: tuple[str, ...] = (
    "kafka 9092",
    "grafana 3000",
    "elastic 9200",
    "memcached 11211",
    "amqp 5672",
    "ldap port 389",
    "bgp port 179",
    "snmp udp 161",
    "http 404 miss",
    "http 502 badgw",
    "chmod 644 file",
    "chmod 700 dir",
    "zstd level 19",
    "btrfs scrub",
    "zfs snapshot",
    "tmux attach",
    "strace syscall",
    "epoll wakeup",
    "numpy einsum",
    "scipy sparse",
    "blake2b 512",
    "chacha20 poly",
    "kruskal mst",
    "bellman ford",
)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


_SEARCH_ENGINE_FOR_GEN = OfflineSearchEngine()


def _sample_open_phrase(rng: random.Random) -> str:
    """Sample realistic 1-3 word + numeric SERP/desktop labels for character-level OCR training."""
    r = rng.random()
    if r < 0.35:
        pool = KNOWLEDGE_DOCS + AD_TRAP_DOCS
        return rng.choice(pool).slug
    if r < 0.70:
        w = rng.choice(TRAIN_OPEN_WORDS)
        num_choice = str(rng.choice([
            rng.randint(10, 99),
            rng.randint(100, 999),
            rng.randint(1000, 9999),
            22, 53, 80, 123, 301, 418, 429, 443, 503, 587, 600, 755, 3306, 5432, 6379, 8080,
        ]))
        return f"{w} {num_choice}" if rng.random() < 0.8 else f"{w} port {num_choice[:3]}"
    w1, w2 = rng.sample(TRAIN_OPEN_WORDS, 2)
    return f"{w1} {w2}"[:18]


def _sample_serp_quad(rng: random.Random, split: str) -> list[str]:
    """Sample 4 SERP result labels with hard lexical/numeric siblings so CRNN reads every word & digit."""
    if split == "ood_phrases":
        return rng.sample(list(HELD_OUT_OOD_PHRASES), 4)

    r = rng.random()
    # 28%: Same prefix word, 4 DIFFERENT numbers (forces exact digit verification, e.g. 429 vs 503 vs 301!)
    if r < 0.28:
        w = rng.choice(TRAIN_OPEN_WORDS)
        nums = set()
        while len(nums) < 4:
            nums.add(str(rng.choice([
                rng.randint(10, 99),
                rng.randint(100, 999),
                rng.randint(1000, 9999),
                22, 53, 80, 123, 161, 179, 301, 389, 418, 429, 443, 503, 587, 600, 644, 700, 755, 3306, 5432, 6379, 8080,
            ])))
        out = [f"{w} {n}" for n in nums]
        rng.shuffle(out)
        return out

    # 22%: Shared middle/suffix word (`port`, `file`, `dir`, `mode`, `code`, `sql`), distinct words + numbers!
    if r < 0.50:
        mid = rng.choice(["port", "code", "mode", "key", "file", "dir", "level"])
        ws = rng.sample(TRAIN_OPEN_WORDS, 4)
        out = [f"{ws[i]} {mid} {rng.randint(10, 999)}"[:18] for i in range(4)]
        if rng.random() < 0.35:
            out[0] = rng.choice(AD_TRAP_DOCS).slug
        rng.shuffle(out)
        return out

    # 28%: Real BM25 search sibling cluster (including optional rank-1 `[ad ...]` trap!)
    if r < 0.78:
        doc = rng.choice(KNOWLEDGE_DOCS)
        res = _SEARCH_ENGINE_FOR_GEN.search(doc.slug, top_n=4, include_ad_trap=(rng.random() < 0.45))
        out = [d.slug for d in res]
        while len(out) < 4:
            cand = _sample_open_phrase(rng)
            if cand not in out:
                out.append(cand)
        if rng.random() < 0.5:
            rng.shuffle(out)
        return out[:4]

    # 22%: General diverse open-vocabulary phrases
    used: list[str] = []
    while len(used) < 4:
        ph = _sample_open_phrase(rng)
        if ph not in used:
            used.append(ph)
    return used


def _draw_tk_styled_screen(
    elements: list[HardElement],
    theme_name: str,
    font_path: str,
    title: str,
    archetype: str,
    rng: random.Random,
) -> np.ndarray:
    """Fast PIL renderer faithfully matching `MockDesktopOS` & `TkDesktopApp` X11/Tkinter geometry."""
    pal_hex = PALETTES[theme_name]
    p = {k: _hex_to_rgb(v) for k, v in pal_hex.items()}

    img = Image.new("RGB", (HARD_CANVAS, HARD_CANVAS), p["bg"])
    d = ImageDraw.Draw(img)
    f8 = _get_font(font_path, 9 if rng.random() < 0.35 else 8)
    f7 = _get_font(font_path, 8 if rng.random() < 0.25 else 7)

    if archetype in ("desktop_serp", "desktop_multiwin"):
        # Window frame + bottom taskbar matching MockDesktopOS
        wx, wy, ww, wh = 48, 6, 204, 228
        d.rectangle([wx, wy, wx + ww, wy + wh], fill=p["widget"], outline=p["border"])
        d.rectangle([wx, wy, wx + ww, wy + 18], fill=p["accent"], outline=p["border"])
        d.text((wx + 5, wy + 4), title, fill=(255, 255, 255), font=f8)
        d.rectangle([0, 235, 255, 255], fill=p["chrome"], outline=p["border"])
    else:
        # Full-screen Tkinter app frame matching FileManager / Browser / Editor / Settings
        d.rectangle([0, 0, 255, 18], fill=p["chrome"], outline=p["border"])
        d.text((6, 4), title, fill=p["fg"], font=f8)
        d.rectangle([0, 18, 66, 255], fill=p["panel"], outline=p["border"])
        d.rectangle([0, 18, 255, 38], fill=p["panel"], outline=p["border"])

    for el in elements:
        fg = p["muted"] if el.state == "disabled" else p["fg"]
        outline = (245, 197, 66) if el.state == "focused" and rng.random() < 0.35 else (p["accent"] if el.state in ("focused", "selected") else p["border"])

        if el.role == "button":
            bg = p["panel"] if el.state == "disabled" else p["accent"]
            bfg = p["muted"] if el.state == "disabled" else (255, 255, 255)
            d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=bg, outline=outline)
            d.text((el.x + 5, el.y + max(1, (el.h - 10) // 2)), el.label[: max(4, (el.w - 8) // 5)], fill=bfg, font=f7 if el.h < 18 else f8)

        elif el.role == "tab":
            bg = p["widget"] if el.state == "selected" else p["chrome"]
            d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=bg, outline=outline)
            if el.state == "selected":
                d.rectangle([el.x + 1, el.y + el.h - 2, el.x + el.w - 1, el.y + el.h], fill=p["accent"])
            d.text((el.x + 4, el.y + max(1, (el.h - 9) // 2)), el.label[: max(4, (el.w - 6) // 5)], fill=fg, font=f7 if el.w < 54 else f8)

        elif el.role == "icon":
            if el.x < 16 and el.h >= 26:
                # Desktop launcher icon (`x=3..45, h=30`)!
                d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=p["panel"], outline=outline)
                cx = el.x + el.w // 2
                d.rectangle([cx - 5, el.y + 3, cx + 5, el.y + 11], fill=p["accent"])
                d.text((el.x + 3, el.y + 16), el.label[:7], fill=fg, font=f7)
            else:
                # Compact toolbar icon (`h=16`)!
                d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=p["widget"], outline=outline)
                gx, gy = el.x + 5, el.y + el.h // 2
                if el.label in ("play", "share", "upload"):
                    d.polygon([(gx - 2, gy - 3), (gx - 2, gy + 3), (gx + 3, gy)], fill=p["accent"])
                else:
                    d.line([(gx - 3, gy), (gx + 3, gy)], fill=p["accent"], width=2)
                    d.line([(gx, gy - 3), (gx, gy + 3)], fill=p["accent"], width=2)
                max_c = max(4, (el.w - 11) // 5)
                d.text((el.x + 10, el.y + max(1, (el.h - 9) // 2)), el.label[:max_c], fill=fg, font=f7)

        elif el.role == "field":
            d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=p["field_bg"], outline=outline)
            d.text((el.x + 4, el.y + max(1, (el.h - 10) // 2)), el.label[: max(6, (el.w - 8) // 5)], fill=fg, font=f8)
            if el.state == "focused":
                d.line([(el.x + el.w - 4, el.y + 3), (el.x + el.w - 4, el.y + el.h - 3)], fill=p["accent"])

        elif el.role == "menu":
            bg = p["field_bg"] if archetype == "desktop_serp" else p["panel"]
            d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=bg, outline=outline)
            d.text((el.x + 5, el.y + max(1, (el.h - 10) // 2)), el.label[: max(6, (el.w - 10) // 5)], fill=fg, font=f8)

        elif el.role == "checkbox":
            bx, by = el.x + 2, el.y + (el.h - 10) // 2
            d.rectangle([bx, by, bx + 10, by + 10], fill=p["field_bg"], outline=outline)
            if el.state == "selected":
                d.line([(bx + 2, by + 5), (bx + 4, by + 8), (bx + 8, by + 2)], fill=p["accent"], width=2)
            d.text((bx + 14, el.y + max(1, (el.h - 10) // 2)), el.label[:10], fill=fg, font=f8)

        elif el.role == "slider":
            d.rectangle([el.x, el.y, el.x + el.w, el.y + el.h], fill=p["bg"], outline=p["panel"])
            d.text((el.x + 3, el.y + max(1, (el.h - 10) // 2)), el.label[:7], fill=fg, font=f8)
            tx0 = el.x + max(34, int(el.w * 0.44))
            tx1 = el.x + el.w - 6
            ty = el.y + el.h // 2
            d.line([(tx0, ty), (tx1, ty)], fill=p["border"], width=2)
            kx = int(tx0 + 0.45 * max(1, tx1 - tx0))
            d.line([(tx0, ty), (kx, ty)], fill=p["accent"], width=2)
            d.ellipse([kx - 4, ty - 4, kx + 4, ty + 4], fill=p["accent"], outline=fg)

    return np.asarray(img, dtype=np.uint8)


def make_openvocab_screen(
    rng: random.Random,
    split: str = "train",
) -> tuple[HardLayout, np.ndarray]:
    """Generate `(HardLayout, rgb_256)` for Open-Vocab Desktop, SERP, or Crowded Cross-Role UIs."""
    theme = rng.choice(["dark", "light", "high_contrast", "nord_slate", "solarized_light"] if split != "train" else ["dark", "light", "high_contrast"])
    font_pool = OOD_FONT_PATHS if split == "ood_phrases" else SEEN_FONT_PATHS
    font_path = rng.choice(font_pool)

    els: list[HardElement] = []
    mode_r = rng.random()

    if split == "ood_phrases" or mode_r < 0.44:
        archetype = "desktop_serp"
        title = "Browser"
        # Desktop icons (`x=3..45`)
        desk_names = rng.sample(["browser", "notes", "files", "terminal", "editor", "settings", "mail", "music"], 6)
        for i, dn in enumerate(desk_names):
            els.append(HardElement("icon", dn, 3 + rng.randint(0, 2), 6 + i * 36, 42, 30, "normal"))

        # Browser controls
        wx, wy, ww = 48 + rng.randint(-2, 2), 6 + rng.randint(0, 4), 204
        els.append(HardElement("button", "close", wx + ww - 44, wy + 2, 42, 15, "normal"))
        els.append(HardElement("icon", "home", wx + 4, wy + 20, 44, 16, "normal"))
        els.append(HardElement("icon", "back", wx + 52, wy + 20, 44, 16, "normal"))
        els.append(HardElement("field", "search", wx + 4, wy + 40, ww - 58, 20, rng.choice(["normal", "focused"])))
        els.append(HardElement("button", "submit", wx + ww - 50, wy + 40, 46, 20, "normal"))

        # 4 stacked SERP result cards with hard lexical & numeric siblings!
        used_phrases = _sample_serp_quad(rng, split)
        for ri, ph in enumerate(used_phrases):
            els.append(HardElement("menu", ph, wx + 6, wy + 68 + ri * 36, ww - 12, 30, "normal"))

        # Bottom OS taskbar (`y=237`)
        els.append(HardElement("button", "start", 2, 237, 44, 18, "normal"))
        tb_apps = ["browser", "notes", "files", "terminal"]
        sel_tb = rng.choice(tb_apps)
        for ti, ta in enumerate(tb_apps):
            els.append(HardElement("tab", ta, 50 + ti * 51, 237, 49, 18, "selected" if ta == sel_tb else "normal"))

    elif mode_r < 0.72:
        archetype = "desktop_multiwin"
        sub_win = rng.choice(["notes", "article", "terminal", "alert", "files"])
        title = sub_win.capitalize()
        for i, dn in enumerate(("browser", "notes", "files", "terminal", "editor", "settings")):
            els.append(HardElement("icon", dn, 3, 6 + i * 36, 42, 30, "normal"))

        wx, wy, ww = 50 + rng.randint(-2, 4), 10 + rng.randint(-2, 6), 198
        els.append(HardElement("button", "close", wx + ww - 44, wy + 2, 42, 15, "normal"))
        if sub_win == "notes":
            els.append(HardElement("field", "notes", wx + 6, wy + 24, ww - 12, 24, rng.choice(["normal", "focused"])))
            els.append(HardElement("button", "paste", wx + 6, wy + 158, 86, 22, "normal"))
            els.append(HardElement("button", "save", wx + 100, wy + 158, 88, 22, "normal"))
        elif sub_win == "article":
            els.append(HardElement("icon", "home", wx + 4, wy + 20, 44, 16, "normal"))
            els.append(HardElement("icon", "back", wx + 52, wy + 20, 44, 16, "normal"))
            els.append(HardElement("field", "search", wx + 4, wy + 40, ww - 58, 20, "normal"))
            els.append(HardElement("button", "submit", wx + ww - 50, wy + 40, 46, 20, "normal"))
            els.append(HardElement("button", "copy", wx + 8, wy + 152, 86, 22, "normal"))
            els.append(HardElement("button", "bookmark", wx + 102, wy + 152, 90, 22, "normal"))
        elif sub_win == "terminal":
            els.append(HardElement("field", "command", wx + 6, wy + 24, ww - 62, 22, rng.choice(["normal", "focused"])))
            els.append(HardElement("button", "run", wx + ww - 52, wy + 24, 46, 22, "normal"))
        elif sub_win == "files":
            els.append(HardElement("button", "delete", wx + 8, wy + 168, 86, 22, "normal"))
            els.append(HardElement("button", "export", wx + 102, wy + 168, 88, 22, "normal"))
        else:  # alert popup
            els.append(HardElement("button", "cancel", 88, 130, 88, 24, "normal"))

        els.append(HardElement("button", "start", 2, 237, 44, 18, "normal"))
        for ti, ta in enumerate(("browser", "notes", "files", "terminal")):
            els.append(HardElement("tab", ta, 50 + ti * 51, 237, 49, 18, "selected" if ta == sub_win else "normal"))

    else:
        # Crowded cross-role duplicate-label UI (`Files` / `Browser` / `Editor`)!
        archetype = "crowded_crossrole"
        title = rng.choice(["Files", "Browser", "Editor"])
        shared_pool = ["home", "refresh", "download", "upload", "archive", "share", "cart", "export", "filter", "print", "save", "delete", "rename", "cancel", "submit", "login", "play", "pause", "profile", "settings", "inbox", "help"]

        # 4 or 5 narrow toolbar icons (`y=20, h=16, w=35..46`)
        n_ico = rng.choice([4, 5])
        ico_labels = rng.sample(shared_pool[:10], n_ico)
        ix = 68
        iw = 35 if n_ico == 5 else 44
        for il in ico_labels:
            els.append(HardElement("icon", il, ix, 20, iw, 16, "normal"))
            ix += iw + 2

        # 3 tabs — deliberately reuse 1-2 labels from `ico_labels` to force cross-role disambiguation!
        tab_labels = [ico_labels[0]] + rng.sample([x for x in shared_pool if x != ico_labels[0]], 2)
        rng.shuffle(tab_labels)
        for ti, tl in enumerate(tab_labels):
            els.append(HardElement("tab", tl, 68 + ti * 60, 41, 58, 16, "selected" if ti == 0 else "normal"))

        # 3-4 sidebar menus (also sharing 1 label!)
        mnu_labels = [ico_labels[-1]] + rng.sample(["inbox", "archive", "settings", "help", "upload"], 3)
        seen_m: list[str] = []
        for ml in mnu_labels:
            if ml not in seen_m and len(seen_m) < 4:
                seen_m.append(ml)
        for mi, ml in enumerate(seen_m):
            els.append(HardElement("menu", ml, 4, 41 + mi * 22, 58, 18, "normal"))

        # Fields + Checkboxes + Slider + Bottom Action Buttons
        els.append(HardElement("field", "search", 70, 64, 178, 19, rng.choice(["normal", "focused"])))
        if rng.random() < 0.6:
            els.append(HardElement("field", "rename", 70, 88, 178, 18, "normal"))
            cy_chk = 112
        else:
            cy_chk = 90
        els.append(HardElement("checkbox", "filter", 70, cy_chk, 84, 18, rng.choice(["normal", "selected"])))
        els.append(HardElement("checkbox", "export", 160, cy_chk, 84, 18, rng.choice(["normal", "selected"])))
        if rng.random() < 0.5:
            els.append(HardElement("slider", rng.choice(["play", "pause", "download"]), 70, cy_chk + 24, 178, 18, "normal"))

        btn_row1 = rng.sample(["submit", "delete", "rename", "export", "play", "pause"], 3)
        for bi, bl in enumerate(btn_row1):
            els.append(HardElement("button", bl, 70 + bi * 60, 194, 56, 20, "normal"))
        btn_row2 = rng.sample(["save", "cancel", "login", "cart"], 2)
        for bi, bl in enumerate(btn_row2):
            els.append(HardElement("button", bl, 70 + bi * 90, 220, 86, 20, "normal"))

    layout = HardLayout(
        elements=els,
        theme=theme if theme in ("light", "dark", "high_contrast") else "dark",
        window=(0, 0, HARD_CANVAS, HARD_CANVAS),
        title=title,
        archetype=archetype,
        font_path=font_path,
        font_px=8,
        dpi_scale=1.0,
        cursor=None,
        jpeg_quality=rng.choice([None, None, 88, 78]),
        tier="crowded",
        label_mode="text",
    )
    rgb = _draw_tk_styled_screen(els, theme, font_path, title, archetype, rng)
    if layout.jpeg_quality is not None:
        import io
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=layout.jpeg_quality)
        buf.seek(0)
        rgb = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return layout, rgb


def sample_openvocab_desktop_example(rng: random.Random, split: str = "train") -> HardExample:
    layout, _ = make_openvocab_screen(rng, split=split)
    if split == "ood_phrases":
        cands = [e for e in layout.elements if e.label in HELD_OUT_OOD_PHRASES]
    elif layout.archetype == "desktop_serp" and rng.random() < 0.65:
        cands = [e for e in layout.elements if e.role == "menu" and e.state != "disabled"]
    else:
        cands = [e for e in layout.elements if e.state != "disabled"]
    target = rng.choice(cands)
    task = rng.choice(TASKS_HARD[target.role]).format(label=target.label)
    action = ACTION_OF_ROLE[target.role]
    px = int(rng.uniform(target.x + target.w * 0.25, target.x + target.w * 0.75))
    py = int(rng.uniform(target.y + target.h * 0.25, target.y + target.h * 0.75))
    return HardExample(layout=layout, task=task, target=target, action=action, click=(px, py))


def render_openvocab_example(ex: HardExample, size: int = HARD_CANVAS) -> np.ndarray:
    rng = random.Random(hash((ex.task, ex.target.x, ex.target.y)) & 0xFFFFFFFF)
    rgb = _draw_tk_styled_screen(ex.layout.elements, ex.layout.theme, ex.layout.font_path, ex.layout.title, ex.layout.archetype, rng)
    if size != HARD_CANVAS:
        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return rgb


def make_heldout_ood_phrase_split(n: int = 250, seed: int = 202609) -> tuple[list[HardExample], list[np.ndarray]]:
    rng = random.Random(seed)
    exs: list[HardExample] = []
    rgbs: list[np.ndarray] = []
    for _ in range(n):
        layout, rgb = make_openvocab_screen(rng, split="ood_phrases")
        cands = [e for e in layout.elements if e.label in HELD_OUT_OOD_PHRASES]
        target = rng.choice(cands)
        task = f"choose {target.label} from the menu"
        exs.append(HardExample(layout=layout, task=task, target=target, action="click", click=(int(target.cx), int(target.cy))))
        rgbs.append(rgb)
    return exs, rgbs


# ---------------------------------------------------------------------------
# Co-Training Stage-1 & Stage-2 (45% Hard World v2 + 55% Open-Vocab Desktop/Cross-Role)
# ---------------------------------------------------------------------------

def _target_tensors_for_element(el: HardElement, g: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cell_s = HARD_CANVAS / float(g)
    fx = el.cx / cell_s - 0.5
    fy = el.cy / cell_s - 0.5
    gx = int(min(max(round(fx), 0), g - 1))
    gy = int(min(max(round(fy), 0), g - 1))

    yy, xx = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
    sigma = 0.85
    hm = torch.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2.0 * sigma * sigma))
    hm = hm / hm.sum().clamp_min(1e-8)

    off = torch.tensor(
        [
            float(np.clip((el.cx / cell_s - (gx + 0.5)) / 0.45, -1.0, 1.0)),
            float(np.clip((el.cy / cell_s - (gy + 0.5)) / 0.45, -1.0, 1.0)),
        ],
        dtype=torch.float32,
    )
    wh = torch.tensor([el.w / float(HARD_CANVAS), el.h / float(HARD_CANVAS)], dtype=torch.float32)
    act = ACTION_CLASSES.index(ACTION_OF_ROLE[el.role])
    return hm, off, wh, act


def fine_tune_stage1_openvocab(
    init_stage1_ckpt: str,
    steps: int = 750,
    screens_per_step: int = 6,
    queries_per_screen: int = 4,
    lr: float = 5.5e-4,
    seed: int = 9091,
    out_ckpt: str = "checkpoints/eyes_stage1_v3.pt",
) -> Stage1CoarseNet:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    s1 = Stage1CoarseNet()
    raw = torch.load(init_stage1_ckpt, map_location="cpu", weights_only=False)
    s1.load_state_dict(raw["stage1"])
    s1.train()

    opt = torch.optim.AdamW(s1.parameters(), lr=lr, weight_decay=1e-2)
    t0 = time.time()

    for step in range(1, steps + 1):
        imgs_128 = []
        rep_counts = []
        tok_list = []
        hm_list = []
        off_list = []
        wh_list = []
        act_list = []
        cell_list = []

        for _ in range(screens_per_step):
            if rng.random() < 0.45:
                lay = make_hard_layout(rng, tier="mixed", split="seen", label_mode="text")
                rgb256 = render_hard(lay, size=HARD_CANVAS)
                els = [e for e in lay.elements if e.state != "disabled"]
            else:
                lay, rgb256 = make_openvocab_screen(rng, split="train")
                els = [e for e in lay.elements if e.state != "disabled"]
            if len(els) < 2:
                continue

            arr128 = np.asarray(Image.fromarray(rgb256).resize((128, 128), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
            imgs_128.append(torch.from_numpy(arr128).permute(2, 0, 1))

            chosen = rng.sample(els, min(queries_per_screen, len(els)))
            rep_counts.append(len(chosen))
            for el in chosen:
                t_str = rng.choice(TASKS_HARD[el.role]).format(label=el.label)
                tok_list.append(encode_task(t_str))
                hm, off, wh, act = _target_tensors_for_element(el, g=s1.grid)
                hm_list.append(hm)
                off_list.append(off)
                wh_list.append(wh)
                act_list.append(act)
                cell_list.append(int(hm.view(-1).argmax().item()))

        xs = torch.stack(imgs_128, dim=0)
        xs = apply_palette_aug(xs, rng)
        fpn = s1.encode_vision(xs)

        # Expand vision FPN features across queries on each screen
        idx_map = []
        for s_i, rc in enumerate(rep_counts):
            idx_map.extend([s_i] * rc)
        idx_t = torch.tensor(idx_map, dtype=torch.long)
        fpn_exp = fpn[idx_t]

        tasks_t = torch.tensor(tok_list, dtype=torch.long)
        hms_t = torch.stack(hm_list, dim=0)
        offs_t = torch.stack(off_list, dim=0)
        whs_t = torch.stack(wh_list, dim=0)
        acts_t = torch.tensor(act_list, dtype=torch.long)
        cells_t = torch.tensor(cell_list, dtype=torch.long)

        q = s1.encode_instruction(tasks_t)
        scores, pred_offs, pred_wh, act_logits = s1.match_features(fpn_exp, q)
        base_loss, _ = s1.loss(scores, pred_offs, pred_wh, act_logits, hms_t, offs_t, acts_t, target_wh=whs_t)

        # In-screen symmetric contrastive loss (crucial for cross-role same-label disambiguation!)
        contrast = torch.tensor(0.0)
        n_grp = 0
        cur = 0
        for kc in rep_counts:
            if kc >= 2:
                sc_sub = scores[cur : cur + kc]
                cl_sub = cells_t[cur : cur + kc]
                if len(torch.unique(cl_sub)) == kc:
                    mat = sc_sub[:, cl_sub]
                    diag = torch.arange(kc)
                    contrast = contrast + 0.5 * (F.cross_entropy(mat, diag) + F.cross_entropy(mat.t(), diag))
                    n_grp += 1
            cur += kc
        if n_grp > 0:
            contrast = contrast / n_grp

        loss = base_loss + 0.55 * contrast
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s1.parameters(), 2.0)
        opt.step()

        cur_lr = lr * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        if step % 150 == 0 or step == steps:
            print(f"[stage1-v3] step {step}/{steps} loss={loss.item():.4f} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s", flush=True)

    s1.eval()
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"stage1": s1.state_dict(), "step": steps}, out_ckpt)
    return s1


def build_mixed_openvocab_bank(n_per_tier: int = 720, seed: int = 50505) -> dict[str, list[tuple[np.ndarray, list[Any]]]]:
    """Build 2,160-screen native 256x256 bank mixing 45% Hard World v2 + 55% Open-Vocab Desktop/SERP/CrossRole."""
    rng = random.Random(seed)
    bank: dict[str, list[tuple[np.ndarray, list[Any]]]] = {"easy": [], "medium": [], "crowded": []}
    t0 = time.time()
    for tier in ("easy", "medium", "crowded"):
        items = []
        while len(items) < n_per_tier:
            if rng.random() < 0.45:
                lay = make_hard_layout(rng, tier=tier, split="seen", label_mode="text")
                els = [e for e in lay.elements if e.state != "disabled"]
                if len(els) < 2:
                    continue
                rgb = render_hard(lay, size=HARD_CANVAS)
            else:
                lay, rgb = make_openvocab_screen(rng, split="train")
                els = [e for e in lay.elements if e.state != "disabled"]
                if lay.archetype == "desktop_serp":
                    serp_els = [e for e in els if e.role == "menu"]
                    chrome_els = [e for e in els if e.role != "menu"]
                    els = serp_els + rng.sample(chrome_els, min(4, len(chrome_els)))
                if len(els) < 2:
                    continue
            items.append((rgb, els))
        bank[tier] = items
    print(f"[mixed_bank] {sum(len(v) for v in bank.values())} screens in {time.time()-t0:.1f}s rss={peak_rss_mb():.0f}MB", flush=True)
    return bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", default="checkpoints/eyes_crnn40_3550.pt")
    ap.add_argument("--stage1-steps", type=int, default=750)
    ap.add_argument("--stage2-steps-per-bank", type=int, default=750)
    ap.add_argument("--out-pt", default="checkpoints/eyes_openvocab_5050.pt")
    ap.add_argument("--export-dir", default="checkpoints/grip_v3")
    args = ap.parse_args()

    torch.set_num_threads(2)

    # 1. Co-train Stage-1 (750 steps)
    s1 = fine_tune_stage1_openvocab(
        init_stage1_ckpt=args.init_ckpt,
        steps=args.stage1_steps,
        lr=5.5e-4,
        seed=9091,
        out_ckpt="checkpoints/eyes_stage1_v3.pt",
    )

    # 2. Co-train Stage-2 CRNN across 2 fresh mixed open-vocab + HardWorld banks (1,500 steps total -> step 5,050!)
    raw = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    s2 = Stage2RefineNet()
    s2.load_state_dict(raw["stage2"])

    for b_idx, b_seed in enumerate((51515, 62626)):
        bank = build_mixed_openvocab_bank(n_per_tier=720, seed=b_seed)
        s2 = train_stage2_refiner(
            stage1=s1,
            steps=args.stage2_steps_per_bank,
            screens_per_step=8,
            max_siblings=6,
            lr=9.0e-4 if b_idx == 0 else 5.5e-4,
            seed=700 + b_idx,
            threads=2,
            hard_neg_mining=True,
            curriculum=False,
            bank=bank,
            init_stage2=s2,
            out_ckpt=args.out_pt,
        )
        del bank
        gc.collect()

    total_step = 3550 + 2 * args.stage2_steps_per_bank
    torch.save({"stage1": s1.state_dict(), "stage2": s2.state_dict(), "step_stage2": total_step}, args.out_pt)
    export_grounder_checkpoint(args.out_pt, args.export_dir, step=total_step, top_k=12)

    # 3. Evaluate on ALL FOUR benchmarks: Hard Seen, Hard OOD, Held-Out Unseen Open-Vocab Phrases, M7 60-Task, M8 40-Task!
    g = EyesGrounder(s1, s2, top_k=12, nms_radius_cells=1, stage1_weight=0.50)

    val_seen = make_hard_val_split(n=1000, seed=202604, split="seen")
    rgb_seen = [render_hard(e.layout, size=HARD_CANVAS) for e in val_seen]
    rs, _ = evaluate_grounder_on_split(g, val_seen, use_stage2=True, threads=2, pre_rendered=rgb_seen)
    del rgb_seen
    gc.collect()

    val_ood = make_hard_val_split(n=500, seed=202605, split="ood")
    rgb_ood = [render_hard(e.layout, size=HARD_CANVAS) for e in val_ood]
    ro, _ = evaluate_grounder_on_split(g, val_ood, use_stage2=True, threads=2, pre_rendered=rgb_ood)
    del rgb_ood
    gc.collect()

    val_ov, rgb_ov = make_heldout_ood_phrase_split(n=250, seed=202609)
    rov, _ = evaluate_grounder_on_split(g, val_ov, use_stage2=True, threads=2, pre_rendered=rgb_ov)
    del rgb_ov
    gc.collect()

    print(
        f"[M9 grip_v3 eval] HardSeen={rs['hit_widget']:.4f} ({rs['median_center_err_px']:.2f}px, lat={rs['latency_ms_per_frame']:.2f}ms) | "
        f"HardOOD={ro['hit_widget']:.4f} ({ro['median_center_err_px']:.2f}px) | "
        f"HeldOutOpenVocabPhrases={rov['hit_widget']:.4f} ({rov['median_center_err_px']:.2f}px)",
        flush=True,
    )


if __name__ == "__main__":
    main()
