"""60-task deterministic headless UI evaluation suite (`specs/eval_suite.md`).

Runs end-to-end on real X11 (`Xvfb` + `tkinter` apps + `deskhands.capture.Capture`
`XGetImage` + `deskhands.core.DesktopHands` `XTest` events + `EyesGrounder`).

Categories (12 deterministic tasks each = 60 tasks):
  - `file_manager` (`fm-001`..`fm-012`): real filesystem ops in temp workspace
  - `editor`       (`ed-001`..`ed-012`): multi-tab text editor + disk save/export/print
  - `settings`     (`st-001`..`st-012`): checkboxes, sliders (`drag`), fields + `settings.json`
  - `browser`      (`br-001`..`br-012`): crowded 14-16 widget browser/package UI + OOD themes/fonts
  - `recovery`     (`rc-001`..`rc-012`): disabled decoys, stale twin panes (`exclude_box`), modal confirms

Single-command CLI:
  python3 -m neurobrain.eval_suite --out /home/user/fleet/agents/03-eyes/scoreboard.json
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from .diagnose_grip import peak_rss_mb
from .export_grounder import load_exported_grounder

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#16181e",
        "chrome": "#21252e",
        "panel": "#1c1f27",
        "widget": "#2a2f3b",
        "accent": "#3465a4",
        "border": "#525b6e",
        "fg": "#ebedf2",
        "muted": "#687082",
        "field_bg": "#101217",
    },
    "light": {
        "bg": "#f2f4f8",
        "chrome": "#dfe4ec",
        "panel": "#e7ebf2",
        "widget": "#ffffff",
        "accent": "#2b6cb0",
        "border": "#8b95a5",
        "fg": "#181b20",
        "muted": "#8e97a4",
        "field_bg": "#ffffff",
    },
    "high_contrast": {
        "bg": "#000000",
        "chrome": "#121212",
        "panel": "#0a0a0a",
        "widget": "#161616",
        "accent": "#0066cc",
        "border": "#00e5ff",
        "fg": "#ffffff",
        "muted": "#606060",
        "field_bg": "#000000",
    },
    # Unseen OOD palettes matching Hard World OOD split
    "nord_slate": {
        "bg": "#2e3440",
        "chrome": "#3b4252",
        "panel": "#353c4a",
        "widget": "#434c5e",
        "accent": "#5e81ac",
        "border": "#88c0d0",
        "fg": "#eceff4",
        "muted": "#6c7689",
        "field_bg": "#242933",
    },
    "solarized_light": {
        "bg": "#fdf6e3",
        "chrome": "#eee8d5",
        "panel": "#f5efdc",
        "widget": "#fbf4e1",
        "accent": "#268bd2",
        "border": "#839496",
        "fg": "#073642",
        "muted": "#93a1a1",
        "field_bg": "#ffffff",
    },
}


@dataclass
class WidgetSpec:
    wid: str
    role: str  # button | checkbox | field | icon | menu | slider | tab
    label: str
    x: int
    y: int
    w: int
    h: int
    state: str = "normal"  # normal | focused | selected | disabled
    checked: bool = False
    slider_val: float = 0.35
    text_val: str = ""
    stale_decoy: bool = False  # if True, clicking does nothing (for recovery tasks)
    on_invoke: Callable[["TkDesktopApp", "WidgetSpec"], None] | None = None


class TkDesktopApp:
    """Real Tkinter 256x256 X11 window driven exclusively by X11 mouse/keyboard events."""

    def __init__(
        self,
        root: Any,
        title: str,
        theme: str = "dark",
        font_family: str = "DejaVu Sans",
        font_size: int = 8,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.title = title
        self.theme_name = theme
        self.pal = PALETTES[theme]
        self.font_family = font_family
        self.font_size = font_size

        for child in list(self.root.winfo_children()):
            child.destroy()

        self.canvas = tk.Canvas(
            self.root,
            width=256,
            height=256,
            bg=self.pal["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.place(x=0, y=0, width=256, height=256)

        self.widgets: list[WidgetSpec] = []
        self.focused_wid: str | None = None
        self.active_wid: str | None = None
        self.click_seq: int = 0
        self.last_hit_wid: str | None = None
        self.dragging_wid: str | None = None
        self.state_store: dict[str, Any] = {}

        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.focus_force()
        self.canvas.focus_set()

    def add(self, w: WidgetSpec) -> WidgetSpec:
        self.widgets.append(w)
        return w

    def get(self, wid: str) -> WidgetSpec:
        for w in self.widgets:
            if w.wid == wid:
                return w
        raise KeyError(wid)

    def widget_at(self, x: int, y: int) -> WidgetSpec | None:
        for w in reversed(self.widgets):
            if w.x <= x <= w.x + w.w and w.y <= y <= w.y + w.h:
                return w
        return None

    def _on_mouse_down(self, ev: Any) -> None:
        x, y = int(ev.x), int(ev.y)
        hit = self.widget_at(x, y)
        self.last_hit_wid = hit.wid if hit is not None else None
        if hit is None or hit.state == "disabled" or hit.stale_decoy:
            return
        self.active_wid = hit.wid
        self.click_seq += 1
        if hit.role == "field":
            self.focused_wid = hit.wid
            for w in self.widgets:
                if w.role == "field" and w.state != "disabled":
                    w.state = "focused" if w.wid == hit.wid else "normal"
            self.redraw()
            return
        if hit.role == "checkbox":
            hit.checked = not hit.checked
            if hit.on_invoke:
                hit.on_invoke(self, hit)
            self.redraw()
            return
        if hit.role == "slider":
            self.dragging_wid = hit.wid
            self._update_slider_from_x(hit, x)
            if hit.on_invoke:
                hit.on_invoke(self, hit)
            self.redraw()
            return
        if hit.role == "tab":
            for w in self.widgets:
                if w.role == "tab" and w.state != "disabled" and not w.stale_decoy:
                    w.state = "selected" if w.wid == hit.wid else "normal"
            if hit.on_invoke:
                hit.on_invoke(self, hit)
            self.redraw()
            return
        if hit.on_invoke:
            hit.on_invoke(self, hit)
        self.redraw()

    def _on_mouse_drag(self, ev: Any) -> None:
        if not self.dragging_wid:
            return
        w = self.get(self.dragging_wid)
        if w.state == "disabled" or w.stale_decoy:
            return
        self._update_slider_from_x(w, int(ev.x))
        if w.on_invoke:
            w.on_invoke(self, w)
        self.redraw()

    def _on_mouse_up(self, ev: Any) -> None:
        self.dragging_wid = None

    @staticmethod
    def _update_slider_from_x(w: WidgetSpec, x: int) -> None:
        tx0 = w.x + max(30, int(w.w * 0.42))
        tx1 = w.x + w.w - 6
        val = (x - tx0) / max(1.0, float(tx1 - tx0))
        w.slider_val = max(0.0, min(1.0, val))

    def _on_key_press(self, ev: Any) -> None:
        if not self.focused_wid:
            return
        w = self.get(self.focused_wid)
        if w.state == "disabled" or w.stale_decoy:
            return
        ks = getattr(ev, "keysym", "")
        ch = getattr(ev, "char", "")
        if ks == "BackSpace":
            w.text_val = w.text_val[:-1]
        elif ks in ("Return", "KP_Enter"):
            if w.on_invoke:
                w.on_invoke(self, w)
        elif ch and 32 <= ord(ch) < 127:
            w.text_val += ch
            if w.on_invoke:
                w.on_invoke(self, w)
        self.redraw()

    def redraw(self) -> None:
        c = self.canvas
        p = self.pal
        c.delete("all")
        fnt = (self.font_family, self.font_size)

        # Window chrome titlebar (0..18)
        c.create_rectangle(0, 0, 256, 18, fill=p["chrome"], outline=p["border"])
        c.create_text(6, 9, text=self.title, anchor="w", fill=p["fg"], font=fnt)
        for idx, col in enumerate(("#e06c75", "#e5c07b", "#98c379")):
            cx = 224 + idx * 10
            c.create_oval(cx - 3, 6, cx + 3, 12, fill=col, outline="")

        # Optional panel dividers
        if self.state_store.get("has_sidebar"):
            c.create_rectangle(0, 18, 66, 256, fill=p["panel"], outline=p["border"])
        if self.state_store.get("has_toolbar"):
            c.create_rectangle(0, 18, 256, 38, fill=p["panel"], outline=p["border"])

        for w in self.widgets:
            fg = p["muted"] if w.state == "disabled" else p["fg"]
            is_act = (w.wid == self.active_wid)
            outline = "#f5c542" if is_act else (p["accent"] if w.state in ("focused", "selected") else p["border"])

            if w.role == "button":
                bg = p["panel"] if w.state == "disabled" else p["accent"]
                btn_fg = p["muted"] if w.state == "disabled" else "#ffffff"
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=bg, outline=outline)
                c.create_text(w.x + 5, w.y + w.h // 2, text=w.label, anchor="w", fill=btn_fg, font=fnt)

            elif w.role == "tab":
                bg = p["widget"] if w.state == "selected" else p["chrome"]
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=bg, outline=outline)
                if w.state == "selected":
                    c.create_rectangle(w.x + 1, w.y + w.h - 2, w.x + w.w - 1, w.y + w.h, fill=p["accent"], outline="")
                c.create_text(w.x + 4, w.y + w.h // 2, text=w.label, anchor="w", fill=fg, font=fnt)

            elif w.role == "icon":
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=p["widget"], outline=outline)
                gx, gy = w.x + 5, w.y + w.h // 2
                # Distinct vector mini-icon (cross / arrow / triangle) rather than checkbox square
                if w.label in ("play", "share", "upload"):
                    c.create_polygon(gx - 2, gy - 3, gx - 2, gy + 3, gx + 3, gy, fill=p["accent"], outline="")
                else:
                    c.create_line(gx - 3, gy, gx + 3, gy, fill=p["accent"], width=2)
                    c.create_line(gx, gy - 3, gx, gy + 3, fill=p["accent"], width=2)
                ico_fnt = (self.font_family, max(7, self.font_size - 1))
                max_chars = max(4, (w.w - 11) // 5)
                c.create_text(w.x + 10, w.y + w.h // 2, text=w.label[:max_chars], anchor="w", fill=fg, font=ico_fnt)

            elif w.role == "menu":
                bg = p["accent"] if w.state == "selected" else p["panel"]
                mfg = "#ffffff" if bg == p["accent"] else fg
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=bg, outline=outline)
                c.create_text(w.x + 5, w.y + w.h // 2, text=w.label, anchor="w", fill=mfg, font=fnt)

            elif w.role == "checkbox":
                bx, by = w.x + 2, w.y + (w.h - 10) // 2
                c.create_rectangle(bx, by, bx + 10, by + 10, fill=p["field_bg"], outline=outline)
                if w.checked:
                    c.create_line(bx + 2, by + 5, bx + 4, by + 8, fill=p["accent"], width=2)
                    c.create_line(bx + 4, by + 8, bx + 8, by + 2, fill=p["accent"], width=2)
                c.create_text(bx + 14, w.y + w.h // 2, text=w.label, anchor="w", fill=fg, font=fnt)

            elif w.role == "field":
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=p["field_bg"], outline=outline)
                disp = f"{w.label}: {w.text_val}" if w.text_val else w.label
                c.create_text(w.x + 4, w.y + w.h // 2, text=disp[: max(6, (w.w - 8) // 6)], anchor="w", fill=fg, font=fnt)
                if w.state == "focused":
                    c.create_line(w.x + w.w - 5, w.y + 3, w.x + w.w - 5, w.y + w.h - 3, fill=p["accent"])

            elif w.role == "slider":
                c.create_rectangle(w.x, w.y, w.x + w.w, w.y + w.h, fill=p["bg"], outline=outline if is_act else p["panel"])
                c.create_text(w.x + 3, w.y + w.h // 2, text=w.label[:7], anchor="w", fill=fg, font=fnt)
                tx0 = w.x + max(34, int(w.w * 0.44))
                tx1 = w.x + w.w - 6
                ty = w.y + w.h // 2
                c.create_line(tx0, ty, tx1, ty, fill=p["border"], width=2)
                kx = int(round(tx0 + w.slider_val * max(1, tx1 - tx0)))
                c.create_line(tx0, ty, kx, ty, fill=p["accent"], width=2)
                c.create_oval(kx - 4, ty - 4, kx + 4, ty + 4, fill=p["accent"], outline=fg)

        # Tiny 4x4 activity pulse pixel in titlebar so repeated valid clicks on the same live widget remain observable
        if self.click_seq > 0:
            pulse_col = "#98c379" if (self.click_seq % 2 == 1) else "#61afef"
            c.create_rectangle(210, 7, 215, 12, fill=pulse_col, outline="")

    def pump(self, cycles: int = 8, delay_s: float = 0.004) -> None:
        for _ in range(cycles):
            self.root.update_idletasks()
            self.root.update()
            if delay_s > 0:
                time.sleep(delay_s)


# ---------------------------------------------------------------------------
# App builders + deterministic workspace seeding
# ---------------------------------------------------------------------------

def _seed_common_workspace(ws: str) -> None:
    for sub in ("Home", "Downloads", "Archive", "Inbox", "Uploads", "Trash", "docs", "browser"):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    for fn, content in [
        ("Downloads/ubuntu.iso", "ISO-IMAGE-1"),
        ("Downloads/rescue.iso", "ISO-IMAGE-2"),
        ("Downloads/readme.txt", "keep this file"),
        ("Home/draft.txt", "initial draft"),
        ("Home/notes.md", "# Notes\nhello"),
        ("Home/report.csv", "a,b\n1,2\n"),
        ("Inbox/msg01.txt", "urgent item"),
        ("Uploads/staged.pkg", "binary-pkg"),
        ("docs/main.py", "def main():\n    return 1\n"),
        ("docs/notes.txt", "todo: ship m7\n"),
        ("docs/help.md", "# Help\nusage guide\n"),
    ]:
        with open(os.path.join(ws, fn), "w") as f:
            f.write(content)
    with open(os.path.join(ws, "settings.json"), "w") as f:
        json.dump(
            {
                "active_tab": "settings",
                "share": False,
                "upload": False,
                "export": False,
                "refresh": True,
                "filter": False,
                "play_level": 0.25,
                "pause_level": 0.50,
                "download_limit": 0.20,
                "profile_name": "",
                "home_dir": "/home/user",
                "saved": False,
                "submitted": False,
                "logged_out": False,
            },
            f,
            indent=2,
        )


def mount_file_manager(root: Any, ws: str, setup: str, theme: str = "dark", font: str = "DejaVu Sans") -> TkDesktopApp:
    app = TkDesktopApp(root, "Files", theme=theme, font_family=font)
    app.state_store.update(
        {
            "has_sidebar": True,
            "has_toolbar": True,
            "folder": "Home",
            "pending_delete": setup == "pending_batch",
            "indexed_files": sorted(os.listdir(os.path.join(ws, "Home"))),
            "filtered_files": [],
        }
    )
    if setup == "external_arrival":
        with open(os.path.join(ws, "Home", "incoming.log"), "w") as f:
            f.write("arrived")

    # Toolbar icons (y=20..36)
    def do_refresh(a: TkDesktopApp, _: WidgetSpec) -> None:
        folder = a.state_store["folder"]
        a.state_store["indexed_files"] = sorted(os.listdir(os.path.join(ws, folder)))

    def do_upload(a: TkDesktopApp, _: WidgetSpec) -> None:
        with open(os.path.join(ws, "Uploads", "uploaded.done"), "w") as f:
            f.write(a.state_store["folder"])

    def do_archive(a: TkDesktopApp, _: WidgetSpec) -> None:
        src_dir = os.path.join(ws, a.state_store["folder"])
        chk_flt = a.get("chk_filter").checked
        for fn in list(os.listdir(src_dir)):
            p = os.path.join(src_dir, fn)
            if os.path.isfile(p) and (not chk_flt or fn.endswith(".md")):
                shutil.move(p, os.path.join(ws, "Archive", fn))

    def do_share(a: TkDesktopApp, _: WidgetSpec) -> None:
        with open(os.path.join(ws, "shared_links.txt"), "w") as f:
            f.write(f"shared:{a.state_store['folder']}\n")

    app.add(WidgetSpec("ico_refresh", "icon", "refresh", 70, 20, 44, 16, on_invoke=do_refresh))
    app.add(WidgetSpec("ico_upload", "icon", "upload", 116, 20, 44, 16, on_invoke=do_upload))
    app.add(WidgetSpec("ico_archive", "icon", "archive", 162, 20, 46, 16, on_invoke=do_archive))
    app.add(WidgetSpec("ico_share", "icon", "share", 210, 20, 42, 16, on_invoke=do_share))

    # Tabs (y=42..58)
    def switch_folder(folder_name: str):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            a.state_store["folder"] = folder_name
            a.state_store["indexed_files"] = sorted(os.listdir(os.path.join(ws, folder_name)))
        return _cb

    app.add(WidgetSpec("tab_home", "tab", "home", 70, 42, 56, 16, state="selected", on_invoke=switch_folder("Home")))
    app.add(WidgetSpec("tab_download", "tab", "download", 128, 42, 62, 16, on_invoke=switch_folder("Downloads")))
    app.add(WidgetSpec("tab_archive", "tab", "archive", 192, 42, 58, 16, on_invoke=switch_folder("Archive")))

    # Sidebar menu (x=4..62)
    app.add(WidgetSpec("mnu_inbox", "menu", "inbox", 4, 42, 58, 18, on_invoke=switch_folder("Inbox")))
    app.add(WidgetSpec("mnu_upload", "menu", "upload", 4, 64, 58, 18, on_invoke=switch_folder("Uploads")))
    app.add(WidgetSpec("mnu_help", "menu", "help", 4, 86, 58, 18))

    # Search & Rename input fields
    app.add(WidgetSpec("fld_search", "field", "search", 72, 66, 176, 18))
    app.add(WidgetSpec("fld_rename", "field", "rename", 72, 90, 176, 18))

    # Checkboxes
    def on_filter_toggle(a: TkDesktopApp, w: WidgetSpec) -> None:
        q = a.get("fld_search").text_val.strip()
        folder = a.state_store["folder"]
        files = sorted(os.listdir(os.path.join(ws, folder)))
        a.state_store["filtered_files"] = [fn for fn in files if q in fn] if w.checked else files

    app.add(WidgetSpec("chk_filter", "checkbox", "filter", 72, 116, 82, 18, on_invoke=on_filter_toggle))
    app.add(WidgetSpec("chk_export", "checkbox", "export", 162, 116, 84, 18))

    # Bottom action buttons
    def do_delete(a: TkDesktopApp, _: WidgetSpec) -> None:
        folder = a.state_store["folder"]
        fdir = os.path.join(ws, folder)
        if folder == "Downloads":
            for p in glob.glob(os.path.join(fdir, "*.iso")):
                shutil.move(p, os.path.join(ws, "Trash", os.path.basename(p)))
        else:
            for fn in sorted(os.listdir(fdir))[:1]:
                shutil.move(os.path.join(fdir, fn), os.path.join(ws, "Trash", fn))

    def do_rename(a: TkDesktopApp, _: WidgetSpec) -> None:
        new_name = a.get("fld_rename").text_val.strip() or "renamed.txt"
        fdir = os.path.join(ws, a.state_store["folder"])
        files = sorted(os.listdir(fdir))
        if files:
            os.rename(os.path.join(fdir, files[0]), os.path.join(fdir, new_name))

    def do_export(a: TkDesktopApp, _: WidgetSpec) -> None:
        folder = a.state_store["folder"]
        out_p = os.path.join(ws, folder, "manifest.json")
        with open(out_p, "w") as f:
            json.dump({"folder": folder, "include_hidden": a.get("chk_export").checked, "files": sorted(os.listdir(os.path.join(ws, folder)))}, f)

    def do_save(a: TkDesktopApp, _: WidgetSpec) -> None:
        folder = a.state_store["folder"]
        with open(os.path.join(ws, folder, "snapshot.json"), "w") as f:
            json.dump({"saved": True, "files": sorted(os.listdir(os.path.join(ws, folder)))}, f)

    def do_cancel(a: TkDesktopApp, _: WidgetSpec) -> None:
        a.state_store["pending_delete"] = False

    app.add(WidgetSpec("btn_delete", "button", "delete", 72, 198, 54, 20, on_invoke=do_delete))
    app.add(WidgetSpec("btn_rename", "button", "rename", 130, 198, 56, 20, on_invoke=do_rename))
    app.add(WidgetSpec("btn_export", "button", "export", 190, 198, 58, 20, on_invoke=do_export))
    app.add(WidgetSpec("btn_save", "button", "save", 72, 224, 82, 20, on_invoke=do_save))
    app.add(WidgetSpec("btn_cancel", "button", "cancel", 162, 224, 86, 20, on_invoke=do_cancel))

    app.redraw()
    app.pump()
    return app


def mount_editor(root: Any, ws: str, setup: str, theme: str = "light", font: str = "DejaVu Sans") -> TkDesktopApp:
    app = TkDesktopApp(root, "Editor", theme=theme, font_family=font)
    app.state_store.update({"has_sidebar": False, "has_toolbar": True, "doc": "notes.txt", "printed": False, "submitted": False, "canceled": False})

    def switch_doc(doc_name: str):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            a.state_store["doc"] = doc_name
        return _cb

    app.add(WidgetSpec("ico_print", "icon", "print", 8, 20, 52, 16, on_invoke=lambda a, _: open(os.path.join(ws, "docs", "print.spool"), "w").write(a.state_store["doc"])))
    app.add(WidgetSpec("ico_share", "icon", "share", 64, 20, 52, 16, on_invoke=lambda a, _: open(os.path.join(ws, "docs", "shared.doc"), "w").write(a.state_store["doc"])))
    app.add(WidgetSpec("ico_upload", "icon", "upload", 120, 20, 56, 16, on_invoke=lambda a, _: open(os.path.join(ws, "docs", "cloud.sync"), "w").write("synced")))
    app.add(WidgetSpec("ico_refresh", "icon", "refresh", 180, 20, 64, 16, on_invoke=lambda a, _: a.state_store.update(reloaded=True)))

    app.add(WidgetSpec("tab_home", "tab", "home", 8, 42, 56, 16, state="selected", on_invoke=switch_doc("notes.txt")))
    app.add(WidgetSpec("tab_profile", "tab", "profile", 68, 42, 58, 16, on_invoke=switch_doc("main.py")))
    app.add(WidgetSpec("tab_help", "tab", "help", 130, 42, 54, 16, on_invoke=switch_doc("help.md")))
    app.add(WidgetSpec("tab_archive", "tab", "archive", 188, 42, 60, 16, on_invoke=switch_doc("archive.md")))

    app.add(WidgetSpec("fld_inbox", "field", "inbox", 10, 66, 236, 22))
    app.add(WidgetSpec("fld_search", "field", "search", 10, 94, 114, 20))
    app.add(WidgetSpec("fld_rename", "field", "rename", 130, 94, 116, 20))

    app.add(WidgetSpec("chk_filter", "checkbox", "filter", 10, 122, 100, 18))
    app.add(WidgetSpec("chk_export", "checkbox", "export", 126, 122, 104, 18))

    def do_save(a: TkDesktopApp, _: WidgetSpec) -> None:
        doc = a.state_store["doc"]
        body = a.get("fld_inbox").text_val
        p = os.path.join(ws, "docs", doc)
        prev = open(p).read() if os.path.exists(p) else ""
        with open(p, "w") as f:
            f.write(prev + ("\n" if prev and not prev.endswith("\n") else "") + body)

    def do_replace_submit(a: TkDesktopApp, _: WidgetSpec) -> None:
        doc = a.state_store["doc"]
        p = os.path.join(ws, "docs", doc)
        needle = a.get("fld_search").text_val
        repl = a.get("fld_rename").text_val
        txt = open(p).read() if os.path.exists(p) else ""
        if needle:
            txt = txt.replace(needle, repl)
        with open(p, "w") as f:
            f.write(txt)
        a.state_store["submitted"] = True

    def do_export(a: TkDesktopApp, _: WidgetSpec) -> None:
        with open(os.path.join(ws, "docs", "exported.bundle"), "w") as f:
            f.write(f"doc={a.state_store['doc']};wrap={a.get('chk_filter').checked};pdf={a.get('chk_export').checked}")

    def do_delete(a: TkDesktopApp, _: WidgetSpec) -> None:
        p = os.path.join(ws, "docs", a.state_store["doc"])
        with open(p, "w") as f:
            f.write("")

    app.add(WidgetSpec("btn_save", "button", "save", 10, 198, 72, 20, on_invoke=do_save))
    app.add(WidgetSpec("btn_submit", "button", "submit", 88, 198, 76, 20, on_invoke=do_replace_submit))
    app.add(WidgetSpec("btn_export", "button", "export", 170, 198, 76, 20, on_invoke=do_export))
    app.add(WidgetSpec("btn_delete", "button", "delete", 10, 224, 110, 20, on_invoke=do_delete))
    app.add(WidgetSpec("btn_cancel", "button", "cancel", 130, 224, 116, 20, on_invoke=lambda a, _: a.state_store.update(canceled=True)))

    app.redraw()
    app.pump()
    return app


def mount_settings(root: Any, ws: str, setup: str, theme: str = "dark", font: str = "DejaVu Sans") -> TkDesktopApp:
    app = TkDesktopApp(root, "Settings", theme=theme, font_family=font)
    app.state_store.update({"has_sidebar": False, "has_toolbar": False, "active_tab": "settings"})

    def set_tab(tname: str):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            a.state_store["active_tab"] = tname
        return _cb

    app.add(WidgetSpec("tab_settings", "tab", "settings", 8, 22, 60, 16, state="selected", on_invoke=set_tab("settings")))
    app.add(WidgetSpec("tab_profile", "tab", "profile", 72, 22, 56, 16, on_invoke=set_tab("profile")))
    app.add(WidgetSpec("tab_login", "tab", "login", 132, 22, 54, 16, on_invoke=set_tab("login")))
    app.add(WidgetSpec("tab_help", "tab", "help", 190, 22, 56, 16, on_invoke=set_tab("help")))

    app.add(WidgetSpec("chk_share", "checkbox", "share", 10, 46, 108, 18, checked=False))
    app.add(WidgetSpec("chk_upload", "checkbox", "upload", 130, 46, 112, 18, checked=False))
    app.add(WidgetSpec("chk_export", "checkbox", "export", 10, 68, 108, 18, checked=False))
    app.add(WidgetSpec("chk_refresh", "checkbox", "refresh", 130, 68, 112, 18, checked=True))

    app.add(WidgetSpec("sld_play", "slider", "play", 10, 94, 234, 18, slider_val=0.25))
    app.add(WidgetSpec("sld_pause", "slider", "pause", 10, 116, 234, 18, slider_val=0.50))
    app.add(WidgetSpec("sld_download", "slider", "download", 10, 138, 234, 18, slider_val=0.20))

    app.add(WidgetSpec("fld_profile", "field", "profile", 10, 164, 112, 20))
    app.add(WidgetSpec("fld_home", "field", "home", 128, 164, 118, 20))

    def persist(saved_flag: bool = True, submit_flag: bool = False, logout_flag: bool = False):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            data = {
                "active_tab": a.state_store["active_tab"],
                "share": a.get("chk_share").checked,
                "upload": a.get("chk_upload").checked,
                "export": a.get("chk_export").checked,
                "refresh": a.get("chk_refresh").checked,
                "play_level": round(a.get("sld_play").slider_val, 3),
                "pause_level": round(a.get("sld_pause").slider_val, 3),
                "download_limit": round(a.get("sld_download").slider_val, 3),
                "profile_name": a.get("fld_profile").text_val,
                "home_dir": a.get("fld_home").text_val or "/home/user",
                "saved": saved_flag,
                "submitted": submit_flag,
                "logged_out": logout_flag,
            }
            with open(os.path.join(ws, "settings.json"), "w") as f:
                json.dump(data, f, indent=2)
        return _cb

    app.add(WidgetSpec("btn_save", "button", "save", 10, 196, 74, 20, on_invoke=persist(True, False, False)))
    app.add(WidgetSpec("btn_submit", "button", "submit", 90, 196, 76, 20, on_invoke=persist(True, True, False)))
    app.add(WidgetSpec("btn_cancel", "button", "cancel", 172, 196, 74, 20, on_invoke=lambda a, _: a.state_store.update(canceled=True)))
    app.add(WidgetSpec("btn_logout", "button", "logout", 10, 222, 236, 20, on_invoke=persist(False, False, True)))

    app.redraw()
    app.pump()
    return app


def mount_browser(root: Any, ws: str, setup: str, theme: str = "high_contrast", font: str = "DejaVu Sans") -> TkDesktopApp:
    """Crowded 15-widget browser / package-hub app testing dense toolbars + OOD themes/fonts."""
    app = TkDesktopApp(root, "Browser", theme=theme, font_family=font)
    app.state_store.update({"has_sidebar": True, "has_toolbar": True, "tab": "home", "playing": False, "paused": False})

    def write_ev(name: str, payload: Any = True):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            p = os.path.join(ws, "browser", "state.json")
            cur = json.load(open(p)) if os.path.exists(p) else {}
            cur[name] = payload() if callable(payload) else payload
            with open(p, "w") as f:
                json.dump(cur, f)
        return _cb

    # Crowded 5-icon top toolbar (y=20..36)
    app.add(WidgetSpec("ico_home", "icon", "home", 68, 20, 35, 16, on_invoke=write_ev("toolbar_home", True)))
    app.add(WidgetSpec("ico_refresh", "icon", "refresh", 105, 20, 36, 16, on_invoke=write_ev("refreshed", True)))
    app.add(WidgetSpec("ico_download", "icon", "download", 143, 20, 37, 16, on_invoke=lambda a, _: open(os.path.join(ws, "Downloads", "pkg.bin"), "w").write("OK")))
    app.add(WidgetSpec("ico_share", "icon", "share", 182, 20, 35, 16, on_invoke=write_ev("shared", True)))
    app.add(WidgetSpec("ico_cart", "icon", "cart", 219, 20, 34, 16, on_invoke=write_ev("cart_opened", True)))

    # 3 browser tabs
    app.add(WidgetSpec("tab_home", "tab", "home", 68, 40, 56, 16, state="selected", on_invoke=write_ev("tab", "home")))
    app.add(WidgetSpec("tab_cart", "tab", "cart", 126, 40, 58, 16, on_invoke=write_ev("tab", "cart")))
    app.add(WidgetSpec("tab_profile", "tab", "profile", 186, 40, 64, 16, on_invoke=write_ev("tab", "profile")))

    # Bookmarks sidebar
    app.add(WidgetSpec("mnu_inbox", "menu", "inbox", 4, 40, 58, 18, on_invoke=write_ev("bookmark", "inbox")))
    app.add(WidgetSpec("mnu_archive", "menu", "archive", 4, 62, 58, 18, on_invoke=write_ev("bookmark", "archive")))
    app.add(WidgetSpec("mnu_settings", "menu", "settings", 4, 84, 58, 18, on_invoke=write_ev("bookmark", "settings")))
    app.add(WidgetSpec("mnu_help", "menu", "help", 4, 106, 58, 18, on_invoke=write_ev("bookmark", "help")))

    # URL / search bar + filter checkbox + media & auth buttons
    app.add(WidgetSpec("fld_search", "field", "search", 70, 62, 180, 20))
    app.add(WidgetSpec("chk_filter", "checkbox", "filter", 70, 88, 92, 18, on_invoke=write_ev("adblock", lambda: app.get("chk_filter").checked)))
    app.add(WidgetSpec("sld_play", "slider", "play", 70, 112, 180, 18, slider_val=0.30, on_invoke=write_ev("volume", lambda: round(app.get("sld_play").slider_val, 2))))

    app.add(WidgetSpec("btn_submit", "button", "submit", 70, 188, 56, 20, on_invoke=write_ev("navigated_url", lambda: app.get("fld_search").text_val)))
    app.add(WidgetSpec("btn_play", "button", "play", 130, 188, 56, 20, on_invoke=write_ev("media", "playing")))
    app.add(WidgetSpec("btn_pause", "button", "pause", 190, 188, 60, 20, on_invoke=write_ev("media", "paused")))
    app.add(WidgetSpec("btn_login", "button", "login", 70, 216, 86, 20, on_invoke=write_ev("auth", "logged_in")))
    app.add(WidgetSpec("btn_cart", "button", "cart", 162, 216, 88, 20, on_invoke=write_ev("checkout", True)))

    app.redraw()
    app.pump()
    return app


def mount_recovery(root: Any, ws: str, setup: str, theme: str = "dark", font: str = "DejaVu Sans") -> TkDesktopApp:
    """Adversarial & recovery UI states (`specs/drive.md`):
    - disabled primary widget + active twin/fallback
    - stale non-functional twin widget requiring Rung-2 `exclude_box` shift
    - two-step dynamic confirmation banner appearing only after first click
    """
    app = TkDesktopApp(root, "Workspace", theme=theme, font_family=font)
    app.state_store.update({"has_sidebar": True, "has_toolbar": True, "done_key": None})

    def mark_done(tag: str):
        def _cb(a: TkDesktopApp, _: WidgetSpec) -> None:
            with open(os.path.join(ws, f"recovered_{tag}.ok"), "w") as f:
                f.write("ok")
        return _cb

    # Baseline distractors so screen has realistic density (10-12 widgets)
    app.add(WidgetSpec("mnu_home", "menu", "home", 4, 42, 58, 18))
    app.add(WidgetSpec("mnu_help", "menu", "help", 4, 64, 58, 18))
    app.add(WidgetSpec("tab_home", "tab", "home", 70, 42, 58, 16, state="selected"))
    app.add(WidgetSpec("tab_settings", "tab", "settings", 132, 42, 62, 16))
    app.add(WidgetSpec("ico_refresh", "icon", "refresh", 70, 20, 48, 16))
    app.add(WidgetSpec("ico_upload", "icon", "upload", 122, 20, 48, 16))

    if setup == "disabled_save_pair":
        # Disabled save button on left, active save button on right
        app.add(WidgetSpec("btn_save_dis", "button", "save", 72, 190, 80, 20, state="disabled"))
        app.add(WidgetSpec("btn_save_act", "button", "save", 164, 190, 80, 20, state="normal", on_invoke=mark_done("disabled_save")))
    elif setup == "disabled_export_pair":
        app.add(WidgetSpec("btn_exp_dis", "button", "export", 72, 160, 80, 20, state="disabled"))
        app.add(WidgetSpec("btn_exp_act", "button", "export", 164, 214, 80, 20, state="normal", on_invoke=mark_done("disabled_export")))
    elif setup == "disabled_delete_pair":
        app.add(WidgetSpec("btn_del_dis", "button", "delete", 164, 160, 80, 20, state="disabled"))
        app.add(WidgetSpec("btn_del_act", "button", "delete", 72, 214, 80, 20, state="normal", on_invoke=mark_done("disabled_delete")))
    elif setup == "disabled_menu_inbox":
        app.add(WidgetSpec("mnu_inbox_dis", "menu", "inbox", 4, 88, 58, 18, state="disabled"))
        app.add(WidgetSpec("mnu_inbox_act", "menu", "inbox", 4, 134, 58, 18, state="normal", on_invoke=mark_done("disabled_inbox")))

    elif setup == "stale_twin_save":
        # Visually active decoy twin (`stale_decoy=True` -> zero pixel/state change!) + live twin -> forces Rung-2 `exclude_box`!
        app.add(WidgetSpec("btn_save_stale", "button", "save", 72, 150, 170, 22, stale_decoy=True))
        app.add(WidgetSpec("btn_save_live", "button", "save", 72, 212, 84, 20, on_invoke=mark_done("twin_save")))
    elif setup == "stale_twin_submit":
        app.add(WidgetSpec("btn_sub_stale", "button", "submit", 72, 148, 170, 22, stale_decoy=True))
        app.add(WidgetSpec("btn_sub_live", "button", "submit", 160, 212, 86, 20, on_invoke=mark_done("twin_submit")))
    elif setup == "stale_twin_share_icon":
        app.add(WidgetSpec("ico_sh_stale", "icon", "share", 174, 20, 44, 16, stale_decoy=True))
        app.add(WidgetSpec("ico_sh_live", "icon", "share", 72, 96, 54, 18, on_invoke=mark_done("twin_share")))
    elif setup == "stale_twin_checkbox":
        app.add(WidgetSpec("chk_ex_stale", "checkbox", "export", 72, 92, 100, 18, stale_decoy=True))
        app.add(WidgetSpec("chk_ex_live", "checkbox", "export", 72, 136, 100, 18, on_invoke=mark_done("twin_chk")))

    elif setup.startswith("modal_"):
        tag = setup.split("_", 1)[1]  # delete | logout | archive | share
        def open_confirm(a: TkDesktopApp, _: WidgetSpec) -> None:
            # Dynamically spawns confirmation submit/cancel buttons on the live X11 window!
            if not any(w.wid == "btn_modal_submit" for w in a.widgets):
                a.add(WidgetSpec("btn_modal_submit", "button", "submit", 74, 142, 80, 22, on_invoke=mark_done(f"modal_{tag}")))
                a.add(WidgetSpec("btn_modal_cancel", "button", "cancel", 162, 142, 80, 22))

        app.add(WidgetSpec("btn_trigger", "button", tag, 74, 208, 168, 22, on_invoke=open_confirm))

    app.redraw()
    app.pump()
    return app


# ---------------------------------------------------------------------------
# 60 Deterministic Task Definitions (`specs/eval_suite.md`)
# ---------------------------------------------------------------------------

def build_eval_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        # ==================== 1. FILE MANAGER (12 tasks) ====================
        {
            "id": "fm-001", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "dark",
            "instruction": "delete every .iso file in Downloads",
            "plan": [("open the download tab", {}), ("click the delete button", {})],
            "success": {"check": "no_iso_in_downloads", "args": {}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-002", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "light",
            "instruction": "rename the first file in Home to backup01",
            "plan": [("click the rename input box", {"type_text": "backup01"}), ("click the rename button", {})],
            "success": {"check": "file_exists", "args": {"rel": "Home/backup01"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-003", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "dark",
            "instruction": "archive all files from Home into Archive",
            "plan": [("click the archive icon", {})],
            "success": {"check": "file_exists", "args": {"rel": "Archive/draft.txt"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "fm-004", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "high_contrast",
            "instruction": "enable export option and export Home manifest",
            "plan": [("toggle the export checkbox", {}), ("click the export button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "Home/manifest.json", "key": "include_hidden", "val": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-005", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "light",
            "instruction": "upload the current Home directory bundle",
            "plan": [("click the upload icon", {})],
            "success": {"check": "file_exists", "args": {"rel": "Uploads/uploaded.done"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "fm-006", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "dark",
            "instruction": "filter Home files by md extension",
            "plan": [("click the search input box", {"type_text": "md"}), ("toggle the filter checkbox", {})],
            "success": {"check": "app_state_equals", "args": {"key": "filtered_files", "val": ["notes.md"]}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-007", "category": "file_manager", "app": "filemanager", "setup": "external_arrival", "theme": "dark",
            "instruction": "refresh Home directory index to pick up incoming.log",
            "plan": [("click the refresh icon", {})],
            "success": {"check": "app_state_contains", "args": {"key": "indexed_files", "item": "incoming.log"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "fm-008", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "light",
            "instruction": "open Inbox from sidebar and save a folder snapshot",
            "plan": [("choose inbox from the menu", {}), ("click the save button", {})],
            "success": {"check": "file_exists", "args": {"rel": "Inbox/snapshot.json"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-009", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "high_contrast",
            "instruction": "share the Downloads folder link",
            "plan": [("open the download tab", {}), ("click the share icon", {})],
            "success": {"check": "file_contains", "args": {"rel": "shared_links.txt", "substr": "shared:Downloads"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "fm-010", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "dark",
            "instruction": "archive Home files then export the Archive manifest",
            "plan": [("click the archive icon", {}), ("open the archive tab", {}), ("click the export button", {})],
            "success": {"check": "file_exists", "args": {"rel": "Archive/manifest.json"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "fm-011", "category": "file_manager", "app": "filemanager", "setup": "pending_batch", "theme": "light",
            "instruction": "cancel the pending destructive batch in Files",
            "plan": [("click the cancel button", {})],
            "success": {"check": "app_state_equals", "args": {"key": "pending_delete", "val": False}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "fm-012", "category": "file_manager", "app": "filemanager", "setup": "default", "theme": "dark",
            "instruction": "archive only markdown files from Home and save Home snapshot",
            "plan": [("toggle the filter checkbox", {}), ("click the archive icon", {}), ("click the save button", {})],
            "success": {"check": "md_archived_and_saved", "args": {}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },

        # ==================== 2. EDITOR (12 tasks) ====================
        {
            "id": "ed-001", "category": "editor", "app": "editor", "setup": "default", "theme": "light",
            "instruction": "append release2026 to notes.txt and save",
            "plan": [("click the inbox input box", {"type_text": "release2026"}), ("click the save button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/notes.txt", "substr": "release2026"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "ed-002", "category": "editor", "app": "editor", "setup": "default", "theme": "dark",
            "instruction": "switch to profile tab (main.py), append pass and save",
            "plan": [("open the profile tab", {}), ("click the inbox input box", {"type_text": "pass"}), ("click the save button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/main.py", "substr": "pass"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "ed-003", "category": "editor", "app": "editor", "setup": "default", "theme": "high_contrast",
            "instruction": "replace m7 with shipped in notes.txt",
            "plan": [("click the search input box", {"type_text": "m7"}), ("click the rename input box", {"type_text": "shipped"}), ("click the submit button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/notes.txt", "substr": "todo: ship shipped"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "ed-004", "category": "editor", "app": "editor", "setup": "default", "theme": "light",
            "instruction": "switch to help document and spool to printer",
            "plan": [("open the help tab", {}), ("click the print icon", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/print.spool", "substr": "help.md"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "ed-005", "category": "editor", "app": "editor", "setup": "default", "theme": "dark",
            "instruction": "enable both filter and export options then export bundle",
            "plan": [("toggle the filter checkbox", {}), ("toggle the export checkbox", {}), ("click the export button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/exported.bundle", "substr": "wrap=True;pdf=True"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "ed-006", "category": "editor", "app": "editor", "setup": "default", "theme": "light",
            "instruction": "share the profile document",
            "plan": [("open the profile tab", {}), ("click the share icon", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/shared.doc", "substr": "main.py"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "ed-007", "category": "editor", "app": "editor", "setup": "default", "theme": "dark",
            "instruction": "sync editor workspace to cloud via upload icon",
            "plan": [("click the upload icon", {})],
            "success": {"check": "file_exists", "args": {"rel": "docs/cloud.sync"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "ed-008", "category": "editor", "app": "editor", "setup": "default", "theme": "high_contrast",
            "instruction": "clear the contents of help.md using delete",
            "plan": [("open the help tab", {}), ("click the delete button", {})],
            "success": {"check": "file_empty", "args": {"rel": "docs/help.md"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "ed-009", "category": "editor", "app": "editor", "setup": "default", "theme": "light",
            "instruction": "create archive.md on archive tab with text archived01",
            "plan": [("open the archive tab", {}), ("click the inbox input box", {"type_text": "archived01"}), ("click the save button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/archive.md", "substr": "archived01"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "ed-010", "category": "editor", "app": "editor", "setup": "default", "theme": "dark",
            "instruction": "reload open buffer from disk",
            "plan": [("click the refresh icon", {})],
            "success": {"check": "app_state_equals", "args": {"key": "reloaded", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "ed-011", "category": "editor", "app": "editor", "setup": "default", "theme": "light",
            "instruction": "cancel unsaved editor dialog",
            "plan": [("click the cancel button", {})],
            "success": {"check": "app_state_equals", "args": {"key": "canceled", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "ed-012", "category": "editor", "app": "editor", "setup": "default", "theme": "nord_slate",
            "instruction": "export help.md with pdf export checked on nord theme",
            "plan": [("open the help tab", {}), ("toggle the export checkbox", {}), ("click the export button", {})],
            "success": {"check": "file_contains", "args": {"rel": "docs/exported.bundle", "substr": "doc=help.md;wrap=False;pdf=True"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },

        # ==================== 3. SETTINGS (12 tasks) ====================
        {
            "id": "st-001", "category": "settings", "app": "settings", "setup": "default", "theme": "dark",
            "instruction": "enable share in Settings and save",
            "plan": [("toggle the share checkbox", {}), ("click the save button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "settings.json", "key": "share", "val": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-002", "category": "settings", "app": "settings", "setup": "default", "theme": "light",
            "instruction": "disable auto refresh and enable upload then save",
            "plan": [("toggle the refresh checkbox", {}), ("toggle the upload checkbox", {}), ("click the save button", {})],
            "success": {"check": "settings_multi", "args": {"refresh": False, "upload": True, "saved": True}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "st-003", "category": "settings", "app": "settings", "setup": "default", "theme": "dark",
            "instruction": "drag play volume slider above 0.75 and save",
            "plan": [("drag the play slider", {"drag_to_frac": 0.90}), ("click the save button", {})],
            "success": {"check": "json_field_gt", "args": {"rel": "settings.json", "key": "play_level", "val": 0.75}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-004", "category": "settings", "app": "settings", "setup": "default", "theme": "high_contrast",
            "instruction": "lower pause timeout slider below 0.20 and save",
            "plan": [("drag the pause slider", {"drag_to_frac": 0.05}), ("click the save button", {})],
            "success": {"check": "json_field_lt", "args": {"rel": "settings.json", "key": "pause_level", "val": 0.20}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-005", "category": "settings", "app": "settings", "setup": "default", "theme": "light",
            "instruction": "raise download limit slider above 0.80 and submit",
            "plan": [("drag the download slider", {"drag_to_frac": 0.92}), ("click the submit button", {})],
            "success": {"check": "json_field_gt", "args": {"rel": "settings.json", "key": "download_limit", "val": 0.80}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-006", "category": "settings", "app": "settings", "setup": "default", "theme": "dark",
            "instruction": "switch to profile tab, set profile name to agent03 and submit",
            "plan": [("open the profile tab", {}), ("click the profile input box", {"type_text": "agent03"}), ("click the submit button", {})],
            "success": {"check": "settings_multi", "args": {"active_tab": "profile", "profile_name": "agent03", "submitted": True}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "st-007", "category": "settings", "app": "settings", "setup": "default", "theme": "light",
            "instruction": "set home path field to /tmp/work and save settings",
            "plan": [("click the home input box", {"type_text": "work03"}), ("click the save button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "settings.json", "key": "home_dir", "val": "work03"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-008", "category": "settings", "app": "settings", "setup": "default", "theme": "high_contrast",
            "instruction": "log out of current session from Settings",
            "plan": [("click the logout button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "settings.json", "key": "logged_out", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "st-009", "category": "settings", "app": "settings", "setup": "default", "theme": "dark",
            "instruction": "enable export checkbox on login tab and save",
            "plan": [("open the login tab", {}), ("toggle the export checkbox", {}), ("click the save button", {})],
            "success": {"check": "settings_multi", "args": {"active_tab": "login", "export": True, "saved": True}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
        {
            "id": "st-010", "category": "settings", "app": "settings", "setup": "default", "theme": "solarized_light",
            "instruction": "open help tab in solarized Settings and submit",
            "plan": [("open the help tab", {}), ("click the submit button", {})],
            "success": {"check": "settings_multi", "args": {"active_tab": "help", "submitted": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "st-011", "category": "settings", "app": "settings", "setup": "default", "theme": "light",
            "instruction": "cancel Settings changes without saving",
            "plan": [("click the cancel button", {})],
            "success": {"check": "app_state_equals", "args": {"key": "canceled", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "st-012", "category": "settings", "app": "settings", "setup": "default", "theme": "nord_slate", "font": "Liberation Sans",
            "instruction": "enable share and upload on unseen font/theme and save",
            "plan": [("toggle the share checkbox", {}), ("toggle the upload checkbox", {}), ("click the save button", {})],
            "success": {"check": "settings_multi", "args": {"share": True, "upload": True, "saved": True}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },

        # ==================== 4. BROWSER / CROWDED TOOLBAR (12 tasks) ====================
        {
            "id": "br-001", "category": "browser", "app": "browser", "setup": "default", "theme": "dark",
            "instruction": "navigate browser search bar to docslocal and submit",
            "plan": [("click the search input box", {"type_text": "docslocal"}), ("click the submit button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "browser/state.json", "key": "navigated_url", "val": "docslocal"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-002", "category": "browser", "app": "browser", "setup": "default", "theme": "high_contrast",
            "instruction": "download package binary from crowded browser toolbar",
            "plan": [("click the download icon", {})],
            "success": {"check": "file_exists", "args": {"rel": "Downloads/pkg.bin"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "br-003", "category": "browser", "app": "browser", "setup": "default", "theme": "light",
            "instruction": "switch to cart tab and click checkout cart button",
            "plan": [("open the cart tab", {}), ("click the cart button", {})],
            "success": {"check": "browser_multi", "args": {"tab": "cart", "checkout": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-004", "category": "browser", "app": "browser", "setup": "default", "theme": "dark",
            "instruction": "enable adblock filter checkbox and refresh page",
            "plan": [("toggle the filter checkbox", {}), ("click the refresh icon", {})],
            "success": {"check": "browser_multi", "args": {"adblock": True, "refreshed": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-005", "category": "browser", "app": "browser", "setup": "default", "theme": "light",
            "instruction": "open archive bookmark from sidebar",
            "plan": [("choose archive from the menu", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "browser/state.json", "key": "bookmark", "val": "archive"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "br-006", "category": "browser", "app": "browser", "setup": "default", "theme": "high_contrast",
            "instruction": "start media playback and raise volume slider above 0.70",
            "plan": [("click the play button", {}), ("drag the play slider", {"drag_to_frac": 0.88})],
            "success": {"check": "json_field_gt", "args": {"rel": "browser/state.json", "key": "volume", "val": 0.70}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-007", "category": "browser", "app": "browser", "setup": "default", "theme": "dark",
            "instruction": "pause media player in browser",
            "plan": [("click the pause button", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "browser/state.json", "key": "media", "val": "paused"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "br-008", "category": "browser", "app": "browser", "setup": "default", "theme": "light",
            "instruction": "switch to profile tab and click login",
            "plan": [("open the profile tab", {}), ("click the login button", {})],
            "success": {"check": "browser_multi", "args": {"tab": "profile", "auth": "logged_in"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-009", "category": "browser", "app": "browser", "setup": "default", "theme": "nord_slate", "font": "FreeSans",
            "instruction": "press share icon on crowded toolbar (unseen FreeSans + nord)",
            "plan": [("click the share icon", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "browser/state.json", "key": "shared", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "br-010", "category": "browser", "app": "browser", "setup": "default", "theme": "solarized_light", "font": "Liberation Sans",
            "instruction": "open settings bookmark and click home toolbar icon (unseen solarized + Liberation)",
            "plan": [("choose settings from the menu", {}), ("click the home icon", {})],
            "success": {"check": "browser_multi", "args": {"bookmark": "settings", "toolbar_home": True}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "br-011", "category": "browser", "app": "browser", "setup": "default", "theme": "dark",
            "instruction": "open cart popup from crowded top icon bar",
            "plan": [("click the cart icon", {})],
            "success": {"check": "json_field_equals", "args": {"rel": "browser/state.json", "key": "cart_opened", "val": True}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "br-012", "category": "browser", "app": "browser", "setup": "default", "theme": "high_contrast",
            "instruction": "open inbox bookmark, type query status01 and submit",
            "plan": [("choose inbox from the menu", {}), ("click the search input box", {"type_text": "status01"}), ("click the submit button", {})],
            "success": {"check": "browser_multi", "args": {"bookmark": "inbox", "navigated_url": "status01"}},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },

        # ==================== 5. RECOVERY (12 tasks) ====================
        {
            "id": "rc-001", "category": "recovery", "app": "recovery", "setup": "disabled_save_pair", "theme": "dark",
            "instruction": "click the active save button while ignoring or recovering past the disabled save button",
            "plan": [("click the save button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_disabled_save.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-002", "category": "recovery", "app": "recovery", "setup": "disabled_export_pair", "theme": "light",
            "instruction": "click the active export button when primary export is disabled",
            "plan": [("click the export button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_disabled_export.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-003", "category": "recovery", "app": "recovery", "setup": "disabled_delete_pair", "theme": "high_contrast",
            "instruction": "click the enabled delete button while avoiding disabled delete",
            "plan": [("click the delete button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_disabled_delete.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-004", "category": "recovery", "app": "recovery", "setup": "disabled_menu_inbox", "theme": "dark",
            "instruction": "select the active inbox menu entry past disabled inbox entry",
            "plan": [("choose inbox from the menu", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_disabled_inbox.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-005", "category": "recovery", "app": "recovery", "setup": "stale_twin_save", "theme": "dark",
            "instruction": "recover via exclude_box when top save button is frozen",
            "plan": [("click the save button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_twin_save.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-006", "category": "recovery", "app": "recovery", "setup": "stale_twin_submit", "theme": "light",
            "instruction": "recover via exclude_box when primary submit button is unresponsive",
            "plan": [("click the submit button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_twin_submit.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-007", "category": "recovery", "app": "recovery", "setup": "stale_twin_share_icon", "theme": "high_contrast",
            "instruction": "recover via exclude_box when toolbar share icon is frozen",
            "plan": [("click the share icon", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_twin_share.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-008", "category": "recovery", "app": "recovery", "setup": "stale_twin_checkbox", "theme": "nord_slate",
            "instruction": "toggle live export checkbox when upper export checkbox is frozen",
            "plan": [("toggle the export checkbox", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_twin_chk.ok"}},
            "budget": {"max_steps": 6, "max_seconds": 30},
        },
        {
            "id": "rc-009", "category": "recovery", "app": "recovery", "setup": "modal_delete", "theme": "dark",
            "instruction": "click delete then confirm dynamic modal by clicking submit",
            "plan": [("click the delete button", {}), ("click the submit button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_modal_delete.ok"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "rc-010", "category": "recovery", "app": "recovery", "setup": "modal_logout", "theme": "light",
            "instruction": "click logout then confirm dynamic modal by clicking submit",
            "plan": [("click the logout button", {}), ("click the submit button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_modal_logout.ok"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "rc-011", "category": "recovery", "app": "recovery", "setup": "modal_archive", "theme": "high_contrast",
            "instruction": "click archive then confirm dynamic modal by clicking submit",
            "plan": [("click the archive button", {}), ("click the submit button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_modal_archive.ok"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
        {
            "id": "rc-012", "category": "recovery", "app": "recovery", "setup": "modal_share", "theme": "solarized_light",
            "instruction": "click share then confirm dynamic modal by clicking submit",
            "plan": [("click the share button", {}), ("click the submit button", {})],
            "success": {"check": "file_exists", "args": {"rel": "recovered_modal_share.ok"}},
            "budget": {"max_steps": 8, "max_seconds": 30},
        },
    ]
    return tasks


# ---------------------------------------------------------------------------
# Deterministic code-checked success assertions (`specs/eval_suite.md`)
# ---------------------------------------------------------------------------

def evaluate_success_check(spec: dict[str, Any], ws: str, app: TkDesktopApp) -> bool:
    chk = spec["check"]
    args = spec.get("args", {})

    if chk == "no_iso_in_downloads":
        isos = glob.glob(os.path.join(ws, "Downloads", "*.iso"))
        keep = os.path.exists(os.path.join(ws, "Downloads", "readme.txt"))
        return len(isos) == 0 and keep

    if chk == "file_exists":
        return os.path.exists(os.path.join(ws, args["rel"]))

    if chk == "file_empty":
        p = os.path.join(ws, args["rel"])
        return os.path.exists(p) and os.path.getsize(p) == 0

    if chk == "file_contains":
        p = os.path.join(ws, args["rel"])
        return os.path.exists(p) and (args["substr"] in open(p).read())

    if chk == "json_field_equals":
        p = os.path.join(ws, args["rel"])
        if not os.path.exists(p):
            return False
        data = json.load(open(p))
        return data.get(args["key"]) == args["val"]

    if chk == "json_field_gt":
        p = os.path.join(ws, args["rel"])
        if not os.path.exists(p):
            return False
        data = json.load(open(p))
        return float(data.get(args["key"], -1e9)) > float(args["val"])

    if chk == "json_field_lt":
        p = os.path.join(ws, args["rel"])
        if not os.path.exists(p):
            return False
        data = json.load(open(p))
        return float(data.get(args["key"], 1e9)) < float(args["val"])

    if chk == "settings_multi":
        p = os.path.join(ws, "settings.json")
        if not os.path.exists(p):
            return False
        data = json.load(open(p))
        return all(data.get(k) == v for k, v in args.items())

    if chk == "browser_multi":
        p = os.path.join(ws, "browser", "state.json")
        if not os.path.exists(p):
            return False
        data = json.load(open(p))
        return all(data.get(k) == v for k, v in args.items())

    if chk == "app_state_equals":
        return app.state_store.get(args["key"]) == args["val"]

    if chk == "app_state_contains":
        seq = app.state_store.get(args["key"], [])
        return args["item"] in seq

    if chk == "md_archived_and_saved":
        return (
            os.path.exists(os.path.join(ws, "Archive", "notes.md"))
            and os.path.exists(os.path.join(ws, "Home", "draft.txt"))
            and os.path.exists(os.path.join(ws, "Home", "snapshot.json"))
        )

    raise ValueError(f"Unknown check: {chk}")


# ---------------------------------------------------------------------------
# Failure Strip Renderer (`specs/eval_suite.md`)
# ---------------------------------------------------------------------------

def save_failure_strip(task_id: str, trace: list[dict[str, Any]], why: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = max(1, len(trace))
    tile_w, tile_h = 256, 292
    strip = Image.new("RGB", (tile_w * n, tile_h), (18, 20, 26))
    draw = ImageDraw.Draw(strip)

    for i, step in enumerate(trace):
        x0 = i * tile_w
        img = Image.fromarray(step["rgb"])
        strip.paste(img, (x0, 0))
        px, py = int(step["x"]), int(step["y"])
        r = 6
        draw.ellipse((x0 + px - r, py - r, x0 + px + r, py + r), outline=(255, 65, 65), width=2)
        draw.line((x0 + px - r - 3, py, x0 + px + r + 3, py), fill=(255, 65, 65), width=1)
        draw.line((x0 + px, py - r - 3, x0 + px, py + r + 3), fill=(255, 65, 65), width=1)
        draw.text((x0 + 4, 258), f"#{i+1} {step['instr'][:32]}", fill=(235, 238, 245))
        draw.text((x0 + 4, 273), f"click=({px},{py}) {why[:30]}", fill=(255, 120, 120))

    strip.save(out_path)


# ---------------------------------------------------------------------------
# Headless Xvfb Display Manager & Suite Runner
# ---------------------------------------------------------------------------

def _pick_free_display(start: int = 85, end: int = 135) -> int:
    for disp in range(start, end):
        if not os.path.exists(f"/tmp/.X{disp}-lock") and not os.path.exists(f"/tmp/.X11-unix/X{disp}"):
            return disp
    return 119


_SHARED_XVFB_PROC: subprocess.Popen[bytes] | None = None
_SHARED_DISPLAY: str | None = None


def _cleanup_shared_xvfb() -> None:
    global _SHARED_XVFB_PROC, _SHARED_DISPLAY
    if _SHARED_XVFB_PROC is not None:
        _SHARED_XVFB_PROC.terminate()
        try:
            _SHARED_XVFB_PROC.wait(timeout=3)
        except Exception:
            _SHARED_XVFB_PROC.kill()
        _SHARED_XVFB_PROC = None
        _SHARED_DISPLAY = None


atexit.register(_cleanup_shared_xvfb)


def _ensure_xvfb_display() -> str:
    """Start one headless 256x256x24 Xvfb per process so Tcl/Tk never rebinds a dead Display*."""
    global _SHARED_XVFB_PROC, _SHARED_DISPLAY
    if _SHARED_XVFB_PROC is not None and _SHARED_XVFB_PROC.poll() is None and _SHARED_DISPLAY:
        os.environ["DISPLAY"] = _SHARED_DISPLAY
        return _SHARED_DISPLAY

    disp_num = _pick_free_display()
    _SHARED_XVFB_PROC = subprocess.Popen(
        ["Xvfb", f":{disp_num}", "-screen", "0", "256x256x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.45)
    _SHARED_DISPLAY = f":{disp_num}"
    os.environ["DISPLAY"] = _SHARED_DISPLAY
    return _SHARED_DISPLAY


def save_composite_failure_gallery(failures: list[dict[str, Any]], out_png: str) -> None:
    if not failures:
        return
    strips = [Image.open(f["strip"]).convert("RGB") for f in failures if os.path.exists(f.get("strip", ""))]
    if not strips:
        return
    max_w = max(im.width for im in strips)
    total_h = sum(im.height + 6 for im in strips)
    canvas = Image.new("RGB", (max_w, total_h), (12, 14, 18))
    y = 0
    for im in strips:
        canvas.paste(im, (0, y))
        y += im.height + 6
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


def _git_head_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def run_eval_suite(
    checkpoint_dir: str | None = "checkpoints/grip_v2",
    grounder_override: Any = None,
    task_ids: list[str] | None = None,
    task_limit: int | None = None,
    out_path: str = "/home/user/fleet/agents/03-eyes/scoreboard.json",
    strips_dir: str = "/home/user/fleet/agents/03-eyes/failure_strips",
) -> dict[str, Any]:
    import tkinter as tk
    from deskhands.capture import Capture
    from deskhands.core import DesktopHands

    all_tasks = build_eval_tasks()
    if task_ids is not None:
        wanted = set(task_ids)
        all_tasks = [t for t in all_tasks if t["id"] in wanted]
    if task_limit is not None:
        all_tasks = all_tasks[:task_limit]

    grounder = grounder_override if grounder_override is not None else load_exported_grounder(checkpoint_dir or "checkpoints/grip_v2")
    grip_step = int(getattr(grounder, "step", 3550))
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        grip_step = int(json.load(open(os.path.join(checkpoint_dir, "config.json"))).get("step", grip_step))

    _ensure_xvfb_display()

    app_mounters = {
        "filemanager": mount_file_manager,
        "editor": mount_editor,
        "settings": mount_settings,
        "browser": mount_browser,
        "recovery": mount_recovery,
    }

    per_task_records: list[dict[str, Any]] = []
    failures_out: list[dict[str, Any]] = []
    recovered_count = 0
    recovery_triggered = 0

    root = tk.Tk()
    try:
        root.overrideredirect(True)
        root.geometry("256x256+0+0")
        root.update()

        with Capture() as cap:
            hands = DesktopHands()
            try:
                for task in all_tasks:
                    t0 = time.perf_counter()
                    max_steps = int(task["budget"]["max_steps"])
                    max_sec = float(task["budget"]["max_seconds"])
                    trace: list[dict[str, Any]] = []
                    steps_used = 0
                    had_retry = False

                    with tempfile.TemporaryDirectory(prefix=f"nb_eval_{task['id']}_") as ws:
                        _seed_common_workspace(ws)
                        mounter = app_mounters[task["app"]]
                        app = mounter(
                            root,
                            ws,
                            task["setup"],
                            theme=task.get("theme", "dark"),
                            font=task.get("font", "DejaVu Sans"),
                        )
                        app.pump(cycles=8, delay_s=0.004)

                        plan = task["plan"]
                        plan_idx = 0
                        exclude_boxes: list[tuple[int, int, int, int]] = []
                        final_step_tried: list[tuple[int, int, int, int]] = []

                        while plan_idx < len(plan) and steps_used < max_steps and (time.perf_counter() - t0) < max_sec:
                            step_instr, step_meta = plan[plan_idx]
                            rgb_before = np.asarray(cap.full_screen().image, dtype=np.uint8)

                            # Combine excluded boxes for Rung-2 recovery (no-op boxes + failed commit-step boxes)
                            is_last = (plan_idx == len(plan) - 1)
                            active_ex = list(exclude_boxes) + (final_step_tried if is_last else [])
                            eb = active_ex if active_ex else None
                            pred = grounder.ground(rgb_before, step_instr, exclude_box=eb)
                            px, py = int(pred["x"]), int(pred["y"])
                            steps_used += 1
                            if is_last:
                                final_step_tried.append((max(0, px - 20), max(0, py - 10), 40, 20))

                            if "drag_to_frac" in step_meta:
                                frac = float(step_meta["drag_to_frac"])
                                target_x2 = int(round(100 + frac * 138))
                                hands.perform({"type": "drag", "x": px, "y": py, "x2": target_x2, "y2": py, "ms": 70})
                                app.pump(cycles=8, delay_s=0.004)
                            else:
                                hands.perform({"type": "click", "x": px, "y": py})
                                app.pump(cycles=6, delay_s=0.004)
                                if "type_text" in step_meta:
                                    hands.perform({"type": "type", "text": str(step_meta["type_text"])})
                                    app.pump(cycles=6, delay_s=0.004)

                            rgb_after = np.asarray(cap.full_screen().image, dtype=np.uint8)
                            trace.append(
                                {
                                    "instr": step_instr,
                                    "x": px,
                                    "y": py,
                                    "action": pred.get("action", "click"),
                                    "rgb": rgb_before,
                                }
                            )

                            # Check immediate completion!
                            if evaluate_success_check(task["success"], ws, app):
                                break

                            # Pure pixel-observation check: did screen change by >= 3 pixels?
                            changed_px = int((np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)).max(axis=-1) > 4).sum())
                            if changed_px < 3:
                                # Rung 2 (`specs/drive.md`): no visual progress -> mask failed box and re-ground!
                                had_retry = True
                                exclude_boxes.append((max(0, px - 22), max(0, py - 11), 44, 22))
                            else:
                                plan_idx += 1
                                exclude_boxes.clear()
                                # Verdict-aware Rung 2 (`drive.md` 22:32 spec-change): if final step finished without
                                # passing runtime success check and budget remains, retry final commit step masking prior click!
                                if plan_idx >= len(plan) and steps_used < max_steps and not evaluate_success_check(task["success"], ws, app):
                                    had_retry = True
                                    plan_idx = len(plan) - 1

                        passed = evaluate_success_check(task["success"], ws, app)
                        elapsed = time.perf_counter() - t0
                        if had_retry:
                            recovery_triggered += 1
                            if passed:
                                recovered_count += 1

                        if not passed:
                            # Post-mortem diagnostic (runs ONLY after episode finished)
                            last = trace[-1] if trace else {"x": -1, "y": -1, "instr": ""}
                            hit_w = app.widget_at(int(last["x"]), int(last["y"]))
                            hit_desc = f"{hit_w.role}:{hit_w.label}({hit_w.state})" if hit_w else "background"
                            why = f"clicked {hit_desc} at ({last['x']},{last['y']}) on '{last['instr']}'; check '{task['success']['check']}' failed"
                            strip_path = os.path.join(strips_dir, f"{task['id']}.png")
                            save_failure_strip(task["id"], trace, why, strip_path)
                            failures_out.append({"id": task["id"], "category": task["category"], "why": why, "strip": strip_path})

                        per_task_records.append(
                            {
                                "id": task["id"],
                                "category": task["category"],
                                "passed": passed,
                                "steps": steps_used,
                                "seconds": round(elapsed, 4),
                                "had_retry": had_retry,
                            }
                        )
            finally:
                hands.close()
    finally:
        try:
            root.update_idletasks()
            root.destroy()
        except Exception:
            pass

    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in per_task_records}):
        sub = [r for r in per_task_records if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "passed": sum(1 for r in sub if r["passed"]),
            "success": round(sum(1 for r in sub if r["passed"]) / max(1, len(sub)), 4),
            "median_steps": float(statistics.median(r["steps"] for r in sub)),
            "median_seconds": round(float(statistics.median(r["seconds"] for r in sub)), 4),
        }

    n_total = len(per_task_records)
    n_pass = sum(1 for r in per_task_records if r["passed"])
    scoreboard = {
        "commit": _git_head_short(),
        "model": {
            "lm_step": 0,
            "grip_step": grip_step,
            "checkpoint": checkpoint_dir or "override",
        },
        "n": n_total,
        "passed": n_pass,
        "success": round(n_pass / max(1, n_total), 4),
        "by_category": by_cat,
        "median_steps": float(statistics.median(r["steps"] for r in per_task_records)) if per_task_records else 0,
        "median_seconds": round(float(statistics.median(r["seconds"] for r in per_task_records)), 4) if per_task_records else 0.0,
        "ram_peak_mb": round(peak_rss_mb(), 1),
        "recovery_stats": {
            "tasks_triggering_retry": recovery_triggered,
            "tasks_recovered_to_done": recovered_count,
            "recovery_rate": round(recovered_count / max(1, recovery_triggered), 4) if recovery_triggered else 1.0,
        },
        "failures": failures_out,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(scoreboard, f, indent=2)

    return scoreboard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grip_v2")
    ap.add_argument("--out", default="/home/user/fleet/agents/03-eyes/scoreboard.json")
    ap.add_argument("--strips-dir", default="/home/user/fleet/agents/03-eyes/failure_strips")
    ap.add_argument("--composite-strip", default="/home/user/fleet/agents/03-eyes/galleries/m7_failures_strip.png")
    ap.add_argument("--tasks-manifest", default="/home/user/fleet/agents/03-eyes/eval_tasks_60.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)

    if args.tasks_manifest:
        os.makedirs(os.path.dirname(args.tasks_manifest), exist_ok=True)
        clean_manifest = [
            {k: t[k] for k in ("id", "category", "instruction", "app", "setup", "success", "budget")}
            for t in build_eval_tasks()
        ]
        with open(args.tasks_manifest, "w") as f:
            json.dump(clean_manifest, f, indent=2)

    sb = run_eval_suite(
        checkpoint_dir=args.checkpoint,
        task_limit=args.limit,
        out_path=args.out,
        strips_dir=args.strips_dir,
    )
    if args.composite_strip and sb["failures"]:
        save_composite_failure_gallery(sb["failures"], args.composite_strip)
    print(
        f"[eval_suite] n={sb['n']} passed={sb['passed']}/{sb['n']} ({sb['success']*100:.2f}%) | "
        f"med_steps={sb['median_steps']} | med_sec={sb['median_seconds']:.3f}s | ram={sb['ram_peak_mb']}MB | "
        f"recovery={sb['recovery_stats']}",
        flush=True,
    )
    for cat, cv in sb["by_category"].items():
        print(f"  - {cat:14s}: {cv['passed']:2d}/{cv['n']:2d} ({cv['success']*100:.1f}%) med_steps={cv['median_steps']}", flush=True)
    if sb["failures"]:
        print(f"  failures ({len(sb['failures'])}):", flush=True)
        for fl in sb["failures"]:
            print(f"    * {fl['id']}: {fl['why']}", flush=True)


if __name__ == "__main__":
    main()
