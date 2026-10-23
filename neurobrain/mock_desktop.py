"""Mock Desktop OS + Offline Search Engine ("MiniGoogle") + 40-Task Desktop & Googling Suite.

Runs headless on real X11 (`Xvfb` + `tkinter.Tk` + `deskhands.capture.Capture`
`XGetImage` + `deskhands.core.DesktopHands` `libXtst` + `EyesGrounder`).

Features:
  - Desktop wallpaper + 6 desktop launcher icons (`browser`, `notes`, `files`,
    `terminal`, `editor`, `settings`)
  - Bottom OS Taskbar (`y=238..256`) with `start` button + window-switcher tabs
  - Multi-window z-order stack (`Browser`, `Notes`, `Terminal`, `Files`, modal `Alert`)
    with window titlebars, `[x]` (`close`) button, click-to-raise focus
  - Built-in `OfflineSearchEngine` ("MiniGoogle") indexing 52 factual articles with
    BM25-style keyword ranking, SERP result links (open-vocabulary titles!),
    sponsored `[ad]` traps, article reader view, OS clipboard `copy`/`paste`, and
    persistent disk verification.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from .diagnose_grip import peak_rss_mb
from .eval_suite import (
    PALETTES,
    _ensure_xvfb_display,
    _git_head_short,
    save_composite_failure_gallery,
    save_failure_strip,
)
from .export_grounder import load_exported_grounder


# ---------------------------------------------------------------------------
# 1. Offline Search Engine ("MiniGoogle") — 52 curated technical/factual docs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchDoc:
    slug: str          # concise 2-3 word clickable title shown on SERP
    keywords: str      # searchable terms
    url: str
    fact: str          # canonical verified fact string
    is_ad: bool = False


KNOWLEDGE_DOCS: tuple[SearchDoc, ...] = (
    # Networking & Ports
    SearchDoc("postgres 5432", "postgres postgresql default tcp port database 5432", "docs.local/postgres-port", "FACT:POSTGRES_PORT=5432"),
    SearchDoc("mysql 3306", "mysql mariadb default tcp port database 3306", "docs.local/mysql-port", "FACT:MYSQL_PORT=3306"),
    SearchDoc("redis 6379", "redis cache server default tcp port 6379", "docs.local/redis-port", "FACT:REDIS_PORT=6379"),
    SearchDoc("ssh port 22", "openssh sshd remote login default port 22", "docs.local/ssh-port", "FACT:SSH_PORT=22"),
    SearchDoc("dns port 53", "dns domain name system udp tcp port 53", "docs.local/dns-port", "FACT:DNS_PORT=53"),
    SearchDoc("https 443", "https tls ssl web secure default port 443", "docs.local/https-port", "FACT:HTTPS_PORT=443"),
    SearchDoc("http 8080", "http alt proxy dev server port 8080", "docs.local/http-8080", "FACT:HTTP_ALT=8080"),
    SearchDoc("smtp 587", "smtp mail submission starttls port 587", "docs.local/smtp-587", "FACT:SMTP_PORT=587"),
    SearchDoc("ntp port 123", "ntp network time clock sync udp port 123", "docs.local/ntp-123", "FACT:NTP_PORT=123"),

    # HTTP & Protocols
    SearchDoc("http 429", "http status 429 too many requests rate limit", "docs.local/http-429", "FACT:HTTP_429=RATE_LIMIT"),
    SearchDoc("http 301", "http status 301 moved permanently redirect", "docs.local/http-301", "FACT:HTTP_301=PERMANENT_REDIRECT"),
    SearchDoc("http 503", "http status 503 service unavailable overload", "docs.local/http-503", "FACT:HTTP_503=UNAVAILABLE"),
    SearchDoc("http 418", "http status 418 teapot rfc2324", "docs.local/http-418", "FACT:HTTP_418=TEAPOT"),
    SearchDoc("tcp syn ack", "tcp three way handshake syn synack ack", "docs.local/tcp-handshake", "FACT:TCP_HANDSHAKE=SYN_SYNACK_ACK"),
    SearchDoc("tls alpn", "tls handshake alpn h2 http11 negotiation", "docs.local/tls-alpn", "FACT:TLS_EXT=ALPN"),

    # Linux CLI & Permissions
    SearchDoc("chmod 755", "chmod 755 octal rwxr-xr-x executable permissions", "docs.local/chmod-755", "FACT:CHMOD_755=rwxr-xr-x"),
    SearchDoc("chmod 600", "chmod 600 private ssh key rw------- permissions", "docs.local/chmod-600", "FACT:CHMOD_600=rw-------"),
    SearchDoc("chown user", "chown change file owner group recursive", "docs.local/chown", "FACT:CHOWN_FLAG=-R"),
    SearchDoc("umask 022", "umask default directory file creation mask 022", "docs.local/umask", "FACT:UMASK_DEFAULT=022"),
    SearchDoc("tar extract", "tar extract archive xzf tar gz", "docs.local/tar-xzf", "FACT:TAR_EXTRACT=tar -xzf"),
    SearchDoc("tar create", "tar create compress archive czf", "docs.local/tar-czf", "FACT:TAR_CREATE=tar -czf"),
    SearchDoc("grep recursive", "grep search text directory recursive -rn", "docs.local/grep-rn", "FACT:GREP_REC=grep -rn"),
    SearchDoc("find mtime", "find modified files days ago -mtime", "docs.local/find-mtime", "FACT:FIND_RECENT=find -mtime"),
    SearchDoc("cron timer", "crontab five fields minute hour day month weekday", "docs.local/cron-5", "FACT:CRON_FIELDS=5"),
    SearchDoc("df inode", "df -i filesystem free inodes usage", "docs.local/df-inodes", "FACT:DF_INODES=df -i"),
    SearchDoc("kill sigterm", "kill signal 15 sigterm graceful stop", "docs.local/sigterm-15", "FACT:SIGTERM_NUM=15"),
    SearchDoc("kill sigkill", "kill signal 9 sigkill force terminate", "docs.local/sigkill-9", "FACT:SIGKILL_NUM=9"),
    SearchDoc("lsof listen", "lsof open listening sockets -i tcp", "docs.local/lsof-tcp", "FACT:LSOF_NET=lsof -i"),

    # Git Version Control
    SearchDoc("git bisect", "git bisect binary search regression bad commit", "docs.local/git-bisect", "FACT:GIT_BUGHUNT=git bisect"),
    SearchDoc("git reflog", "git reflog recover lost detached head sha", "docs.local/git-reflog", "FACT:GIT_RECOVER=git reflog"),
    SearchDoc("git stash", "git stash pop save dirty working tree", "docs.local/git-stash", "FACT:GIT_TEMP=git stash"),
    SearchDoc("git cherry", "git cherry-pick apply single commit hash", "docs.local/git-cherry", "FACT:GIT_PICK=git cherry-pick"),
    SearchDoc("git revert", "git revert safe inverse commit history", "docs.local/git-revert", "FACT:GIT_UNDO_PUBLIC=git revert"),
    SearchDoc("git blame", "git blame line author commit history -L", "docs.local/git-blame", "FACT:GIT_LINE_HIST=git blame"),

    # Python Stdlib & Runtime
    SearchDoc("python gil", "python global interpreter lock mutex threads", "docs.local/python-gil", "FACT:PYTHON_LOCK=GIL"),
    SearchDoc("asyncio gather", "python asyncio gather concurrent coroutines tasks", "docs.local/asyncio-gather", "FACT:ASYNC_JOIN=asyncio.gather"),
    SearchDoc("venv create", "python -m venv virtual environment isolate pip", "docs.local/python-venv", "FACT:PY_VENV=python3 -m venv"),
    SearchDoc("pytest marks", "pytest parametrize skip xfail markers", "docs.local/pytest-mark", "FACT:PYTEST_PARAM=parametrize"),
    SearchDoc("pathlib glob", "python pathlib path rglob recursive files", "docs.local/pathlib-rglob", "FACT:PATHLIB_REC=Path.rglob"),
    SearchDoc("json dumps", "python json dumps sort_keys indent serialize", "docs.local/json-dumps", "FACT:JSON_SER=json.dumps"),
    SearchDoc("functools lru", "python functools lru_cache memoize decorator", "docs.local/lru-cache", "FACT:PY_MEMO=lru_cache"),
    SearchDoc("collections deque", "python collections deque o1 popleft queue", "docs.local/py-deque", "FACT:PY_QUEUE=deque"),

    # Databases, Crypto & CS
    SearchDoc("sqlite wal", "sqlite pragma journal_mode wal write ahead log", "docs.local/sqlite-wal", "FACT:SQLITE_CONCUR=WAL"),
    SearchDoc("sqlite vacuum", "sqlite vacuum reclaim compact database pages", "docs.local/sqlite-vacuum", "FACT:SQLITE_COMPACT=VACUUM"),
    SearchDoc("sql left join", "sql left outer join preserve unmatched rows", "docs.local/sql-left-join", "FACT:SQL_OUTER=LEFT_JOIN"),
    SearchDoc("btree index", "database btree index logarithmic range scan", "docs.local/btree-index", "FACT:DB_INDEX=BTREE_O_LOG_N"),
    SearchDoc("sha256 bits", "sha256 cryptographic hash digest 256 bits 64 hex", "docs.local/sha256-len", "FACT:SHA256_HEX=64"),
    SearchDoc("ed25519 key", "ed25519 elliptic curve signature 32 byte public key", "docs.local/ed25519", "FACT:ED25519_PUB_BYTES=32"),
    SearchDoc("dijkstra path", "dijkstra shortest path non negative edge weights", "docs.local/dijkstra", "FACT:SHORTEST_PATH=DIJKSTRA"),
    SearchDoc("quicksort pivot", "quicksort partition divide conquer n log n", "docs.local/quicksort", "FACT:SORT_AVG=O_N_LOG_N"),
    SearchDoc("utf8 variable", "utf8 unicode encoding 1 to 4 bytes ascii compat", "docs.local/utf8-bytes", "FACT:UTF8_MAX_BYTES=4"),
    SearchDoc("ieee754 float", "ieee 754 float32 1 sign 8 exp 23 mantissa", "docs.local/float32-bits", "FACT:FP32_MANTISSA=23"),
)

AD_TRAP_DOCS: tuple[SearchDoc, ...] = (
    SearchDoc("ad cloud deal", "sponsored cloud hosting promo trial", "ads.local/cloud", "AD:SPONSORED_PROMO", is_ad=True),
    SearchDoc("ad boot camp", "sponsored paid coding course discount", "ads.local/course", "AD:PAID_COURSE", is_ad=True),
    SearchDoc("ad vpn offer", "sponsored security bundle coupon", "ads.local/vpn", "AD:VPN_COUPON", is_ad=True),
)


class OfflineSearchEngine:
    """Deterministic BM25-like local search engine over `KNOWLEDGE_DOCS`."""

    def __init__(self, docs: tuple[SearchDoc, ...] = KNOWLEDGE_DOCS) -> None:
        self.documents = list(docs)

    def search(self, query: str, top_n: int = 4, include_ad_trap: bool = False) -> list[SearchDoc]:
        q_norm = query.lower().strip()
        q_toks = [t for t in q_norm.replace("-", " ").split() if t]

        scored: list[tuple[float, int, SearchDoc]] = []
        for idx, doc in enumerate(self.documents):
            hay_slug = doc.slug.lower()
            hay_kw = doc.keywords.lower()
            hay_toks = set(hay_kw.replace("-", " ").split())
            score = 0.0
            if q_norm and q_norm in hay_slug:
                score += 12.0
            for qt in q_toks:
                if qt in hay_slug.split():
                    score += 6.0
                elif qt in hay_slug:
                    score += 4.0
                if qt in hay_toks:
                    score += 3.5
                elif any(ht.startswith(qt) or qt.startswith(ht) for ht in hay_toks if len(ht) >= 3 and len(qt) >= 3):
                    score += 1.5
            # Deterministic tie-breaker ensures sibling domain docs populate ranks 2..4!
            scored.append((score, -idx, doc))

        scored.sort(reverse=True)
        organic = [d for _, _, d in scored[:top_n]]

        if include_ad_trap:
            ad = AD_TRAP_DOCS[sum(ord(c) for c in q_norm) % len(AD_TRAP_DOCS)]
            return [ad] + organic[: max(1, top_n - 1)]
        return organic


# ---------------------------------------------------------------------------
# 2. Multi-Window Mock Desktop OS (`256x256` X11 Tkinter + Headless PIL Twin)
# ---------------------------------------------------------------------------

@dataclass
class DesktopHitBox:
    cid: str
    role: str      # icon | tab | button | field | menu | checkbox
    label: str
    x: int
    y: int
    w: int
    h: int
    win_id: str    # "desktop" | "taskbar" | window name
    state: str = "normal"
    stale_decoy: bool = False
    on_click: Callable[[], None] | None = None


@dataclass
class OsWindow:
    win_id: str      # "browser" | "notes" | "files" | "terminal" | "alert"
    title: str
    x: int
    y: int
    w: int
    h: int
    minimized: bool = False
    closable: bool = True


class MockDesktopOS:
    """Interactive multi-window Mock OS running inside a 256x256 Tkinter X11 canvas."""

    def __init__(
        self,
        root: Any,
        ws_dir: str,
        theme: str = "dark",
        font_family: str = "DejaVu Sans",
        search_ad_trap: bool = False,
        stale_copy_decoy: bool = False,
    ) -> None:
        self.root = root
        self.ws_dir = ws_dir
        self.theme_name = theme
        self.pal = PALETTES[theme]
        self.font_family = font_family
        self.search_engine = OfflineSearchEngine()
        self.search_ad_trap = search_ad_trap
        self.stale_copy_decoy = stale_copy_decoy

        os.makedirs(os.path.join(ws_dir, "notes"), exist_ok=True)
        os.makedirs(os.path.join(ws_dir, "files"), exist_ok=True)
        os.makedirs(os.path.join(ws_dir, "terminal"), exist_ok=True)
        os.makedirs(os.path.join(ws_dir, "browser"), exist_ok=True)

        # OS-wide mutable state
        self.clipboard: str = ""
        self.start_menu_open: bool = False
        self.focused_input: str | None = None  # "browser_search" | "notes_body" | "terminal_cmd" | "files_filter"
        self.active_cid: str | None = None
        self.click_seq: int = 0

        # Per-app state
        self.browser_mode: str = "home"  # "home" | "serp" | "article"
        self.browser_query: str = ""
        self.serp_results: list[SearchDoc] = []
        self.current_article: SearchDoc | None = None
        self.bookmarks: list[str] = []

        self.notes_text: str = ""
        self.notes_saved_count: int = 0

        self.terminal_cmd: str = ""
        self.terminal_out: str = "ready"

        self.files_items: list[str] = ["report.txt", "temp.iso", "draft.md"]
        for fn in self.files_items:
            with open(os.path.join(ws_dir, "files", fn), "w") as f:
                f.write(f"seed:{fn}\n")

        # Window Z-stack (last item = topmost foreground window!)
        self.windows: list[OsWindow] = []
        self.hitboxes: list[DesktopHitBox] = []

        self.canvas = None
        if self.root is not None:
            for ch in list(self.root.winfo_children()):
                ch.destroy()
            import tkinter as tk
            self.canvas = tk.Canvas(self.root, width=256, height=256, bg=self.pal["bg"], highlightthickness=0, bd=0)
            self.canvas.place(x=0, y=0, width=256, height=256)
            self.canvas.bind("<Button-1>", self._on_x11_click)
            self.root.bind("<KeyPress>", self._on_x11_key)
            self.root.focus_force()
            self.canvas.focus_set()

    # -------------------- Window management --------------------
    def open_window(self, win_id: str) -> None:
        for i, w in enumerate(self.windows):
            if w.win_id == win_id:
                w.minimized = False
                # Raise to top of Z-order (keep modal 'alert' on top if present)
                self.windows.pop(i)
                self._insert_respecting_modal(w)
                return

        presets = {
            "browser": OsWindow("browser", "Browser", 48, 6, 204, 228),
            "notes": OsWindow("notes", "Notes", 54, 14, 196, 218),
            "files": OsWindow("files", "Files", 50, 10, 200, 222),
            "terminal": OsWindow("terminal", "Terminal", 52, 16, 198, 216),
            "alert": OsWindow("alert", "Alert", 44, 58, 176, 108),
        }
        self._insert_respecting_modal(presets[win_id])

    def _insert_respecting_modal(self, win: OsWindow) -> None:
        if win.win_id != "alert" and any(w.win_id == "alert" and not w.minimized for w in self.windows):
            idx = next(i for i, w in enumerate(self.windows) if w.win_id == "alert")
            self.windows.insert(idx, win)
        else:
            self.windows.append(win)

    def close_window(self, win_id: str) -> None:
        self.windows = [w for w in self.windows if w.win_id != win_id]

    def top_window(self) -> OsWindow | None:
        vis = [w for w in self.windows if not w.minimized]
        return vis[-1] if vis else None

    # -------------------- App Actions --------------------
    def run_browser_search(self) -> None:
        q = self.browser_query.strip() or "linux"
        self.serp_results = self.search_engine.search(q, top_n=4, include_ad_trap=self.search_ad_trap)
        self.browser_mode = "serp"
        with open(os.path.join(self.ws_dir, "browser", "history.json"), "w") as f:
            json.dump({"last_query": q, "results": [d.slug for d in self.serp_results]}, f)

    def open_serp_doc(self, doc: SearchDoc) -> None:
        self.current_article = doc
        self.browser_mode = "article"
        with open(os.path.join(self.ws_dir, "browser", "opened_article.json"), "w") as f:
            json.dump({"slug": doc.slug, "url": doc.url, "fact": doc.fact, "is_ad": doc.is_ad}, f)

    def bookmark_current(self) -> None:
        if self.current_article is not None:
            self.bookmarks.append(self.current_article.slug)
            with open(os.path.join(self.ws_dir, "browser", "bookmarks.json"), "w") as f:
                json.dump({"bookmarks": self.bookmarks, "last_fact": self.current_article.fact, "is_ad": self.current_article.is_ad}, f)

    def copy_article_fact(self) -> None:
        if self.current_article is not None:
            self.clipboard = self.current_article.fact

    def paste_into_notes(self) -> None:
        if self.clipboard:
            self.notes_text = (self.notes_text + "\n" + self.clipboard).strip() if self.notes_text else self.clipboard

    def save_notes(self) -> None:
        self.notes_saved_count += 1
        with open(os.path.join(self.ws_dir, "notes", "saved_note.txt"), "w") as f:
            f.write(self.notes_text + "\n")

    def exec_terminal(self) -> None:
        cmd = self.terminal_cmd.strip().lower()
        outputs = {
            "df": "DISK_OK:/dev/root 42%",
            "uname": "KERNEL:Linux-6.1-x86_64",
            "whoami": "USER:agent03",
            "ports": "LISTEN:22,80,5432",
            "": "STATUS:IDLE_OK",
        }
        out = outputs.get(cmd, f"EXEC:{cmd.upper()}_OK")
        self.terminal_out = out
        self.clipboard = out
        with open(os.path.join(self.ws_dir, "terminal", "last_out.txt"), "w") as f:
            f.write(out + "\n")

    def clean_iso_files(self) -> None:
        for fn in list(self.files_items):
            if fn.endswith(".iso"):
                p = os.path.join(self.ws_dir, "files", fn)
                if os.path.exists(p):
                    os.remove(p)
                self.files_items.remove(fn)

    # -------------------- Scene Layout + Hit Testing --------------------
    def rebuild_scene(self) -> None:
        """Build ordered `self.hitboxes` respecting real window occlusion!"""
        self.hitboxes.clear()

        # 1. Desktop Shortcut Icons along left desktop column (`x=3..45, y=6..204`)
        desk_apps = ["browser", "notes", "files", "terminal", "editor", "settings"]
        for i, app_name in enumerate(desk_apps):
            iy = 6 + i * 36
            self.hitboxes.append(
                DesktopHitBox(
                    cid=f"desk_{app_name}",
                    role="icon",
                    label=app_name,
                    x=3,
                    y=iy,
                    w=42,
                    h=30,
                    win_id="desktop",
                    on_click=lambda a=app_name: self.open_window("notes" if a == "editor" else ("terminal" if a == "settings" else a)),
                )
            )

        # 2. Windows in Z-order (bottom to top)
        vis_wins = [w for w in self.windows if not w.minimized]
        for w_idx, win in enumerate(vis_wins):
            is_top = (w_idx == len(vis_wins) - 1)
            wx, wy, ww, wh = win.x, win.y, win.w, win.h

            # Clicking anywhere inside an unfocused background window brings it to front!
            if not is_top:
                self.hitboxes.append(
                    DesktopHitBox(
                        cid=f"raise_{win.win_id}",
                        role="button",
                        label=win.win_id,
                        x=wx,
                        y=wy,
                        w=ww,
                        h=wh,
                        win_id=win.win_id,
                        on_click=lambda wid=win.win_id: self.open_window(wid),
                    )
                )
                continue

            # Foreground window titlebar close button
            if win.closable:
                self.hitboxes.append(
                    DesktopHitBox(
                        cid=f"{win.win_id}_close",
                        role="button",
                        label="close",
                        x=wx + ww - 44,
                        y=wy + 2,
                        w=42,
                        h=15,
                        win_id=win.win_id,
                        on_click=lambda wid=win.win_id: self.close_window(wid),
                    )
                )

            if win.win_id == "browser":
                # Toolbar: home, back
                self.hitboxes.append(
                    DesktopHitBox("br_home", "icon", "home", wx + 4, wy + 20, 44, 16, "browser", on_click=lambda: setattr(self, "browser_mode", "home"))
                )
                self.hitboxes.append(
                    DesktopHitBox("br_back", "icon", "back", wx + 52, wy + 20, 44, 16, "browser", on_click=lambda: setattr(self, "browser_mode", "serp" if self.serp_results else "home"))
                )
                # Search input box + search/submit button
                self.hitboxes.append(
                    DesktopHitBox(
                        "br_search_fld", "field", "search", wx + 4, wy + 40, ww - 58, 20, "browser",
                        state="focused" if self.focused_input == "browser_search" else "normal",
                        on_click=lambda: setattr(self, "focused_input", "browser_search"),
                    )
                )
                self.hitboxes.append(
                    DesktopHitBox("br_go_btn", "button", "submit", wx + ww - 50, wy + 40, 46, 20, "browser", on_click=self.run_browser_search)
                )

                if self.browser_mode == "home":
                    # Quick search topic chips on browser home
                    for ci, topic in enumerate(("postgres 5432", "chmod 755", "git bisect", "sqlite wal")):
                        cy = wy + 78 + ci * 28
                        def _quick(q=topic):
                            self.browser_query = q
                            self.run_browser_search()
                        self.hitboxes.append(DesktopHitBox(f"br_chip_{ci}", "menu", topic, wx + 8, cy, ww - 16, 22, "browser", on_click=_quick))

                elif self.browser_mode == "serp":
                    for ri, doc in enumerate(self.serp_results[:4]):
                        ry = wy + 68 + ri * 36
                        self.hitboxes.append(
                            DesktopHitBox(
                                f"serp_{ri}",
                                "menu",
                                doc.slug,
                                wx + 6,
                                ry,
                                ww - 12,
                                30,
                                "browser",
                                on_click=lambda d=doc: self.open_serp_doc(d),
                            )
                        )

                elif self.browser_mode == "article" and self.current_article is not None:
                    if self.stale_copy_decoy:
                        self.hitboxes.append(
                            DesktopHitBox("br_copy_stale", "button", "copy", wx + 8, wy + 150, 86, 22, "browser", stale_decoy=True)
                        )
                        self.hitboxes.append(
                            DesktopHitBox("br_copy_live", "button", "copy", wx + 104, wy + 150, 88, 22, "browser", on_click=self.copy_article_fact)
                        )
                    else:
                        self.hitboxes.append(
                            DesktopHitBox("br_copy", "button", "copy", wx + 8, wy + 152, 86, 22, "browser", on_click=self.copy_article_fact)
                        )
                        self.hitboxes.append(
                            DesktopHitBox("br_bookmark", "button", "bookmark", wx + 102, wy + 152, 92, 22, "browser", on_click=self.bookmark_current)
                        )

            elif win.win_id == "notes":
                self.hitboxes.append(
                    DesktopHitBox(
                        "nt_body", "field", "notes", wx + 6, wy + 24, ww - 12, 24, "notes",
                        state="focused" if self.focused_input == "notes_body" else "normal",
                        on_click=lambda: setattr(self, "focused_input", "notes_body"),
                    )
                )
                self.hitboxes.append(
                    DesktopHitBox("nt_paste", "button", "paste", wx + 6, wy + 158, 86, 22, "notes", on_click=self.paste_into_notes)
                )
                self.hitboxes.append(
                    DesktopHitBox("nt_save", "button", "save", wx + 100, wy + 158, 88, 22, "notes", on_click=self.save_notes)
                )

            elif win.win_id == "terminal":
                self.hitboxes.append(
                    DesktopHitBox(
                        "tm_cmd", "field", "command", wx + 6, wy + 24, ww - 62, 22, "terminal",
                        state="focused" if self.focused_input == "terminal_cmd" else "normal",
                        on_click=lambda: setattr(self, "focused_input", "terminal_cmd"),
                    )
                )
                self.hitboxes.append(
                    DesktopHitBox("tm_run", "button", "run", wx + ww - 52, wy + 24, 46, 22, "terminal", on_click=self.exec_terminal)
                )

            elif win.win_id == "files":
                self.hitboxes.append(
                    DesktopHitBox("fl_delete", "button", "delete", wx + 8, wy + 168, 86, 22, "files", on_click=self.clean_iso_files)
                )
                self.hitboxes.append(
                    DesktopHitBox(
                        "fl_export", "button", "export", wx + 102, wy + 168, 88, 22, "files",
                        on_click=lambda: json.dump({"files": self.files_items}, open(os.path.join(self.ws_dir, "files", "index.json"), "w")),
                    )
                )

            elif win.win_id == "alert":
                self.hitboxes.append(
                    DesktopHitBox("al_cancel", "button", "cancel", wx + 44, wy + 72, 88, 24, "alert", on_click=lambda: self.close_window("alert"))
                )

        # 3. Bottom OS Taskbar (`y=236..256`, always on top!)
        self.hitboxes.append(
            DesktopHitBox("tb_start", "button", "start", 2, 237, 44, 18, "taskbar", on_click=lambda: setattr(self, "start_menu_open", not self.start_menu_open))
        )
        top_w = self.top_window()
        for ti, t_app in enumerate(("browser", "notes", "files", "terminal")):
            tx = 50 + ti * 51
            self.hitboxes.append(
                DesktopHitBox(
                    f"tb_{t_app}",
                    "tab",
                    t_app,
                    tx,
                    237,
                    49,
                    18,
                    "taskbar",
                    state="selected" if (top_w and top_w.win_id == t_app) else "normal",
                    on_click=lambda a=t_app: self.open_window(a),
                )
            )

    def hitbox_at(self, x: int, y: int) -> DesktopHitBox | None:
        # Reverse order = topmost Z-layer wins!
        for hb in reversed(self.hitboxes):
            if hb.x <= x <= hb.x + hb.w and hb.y <= y <= hb.y + hb.h:
                return hb
        return None

    def _on_x11_click(self, ev: Any) -> None:
        self.rebuild_scene()
        hb = self.hitbox_at(int(ev.x), int(ev.y))
        if hb is None or hb.state == "disabled" or hb.stale_decoy:
            return
        self.active_cid = hb.cid
        self.click_seq += 1
        if hb.role != "field":
            self.focused_input = None
        if hb.on_click is not None:
            hb.on_click()
        self.redraw()

    def _on_x11_key(self, ev: Any) -> None:
        if not self.focused_input:
            return
        ks = getattr(ev, "keysym", "")
        ch = getattr(ev, "char", "")
        attr_map = {"browser_search": "browser_query", "notes_body": "notes_text", "terminal_cmd": "terminal_cmd"}
        attr = attr_map.get(self.focused_input)
        if not attr:
            return
        cur = getattr(self, attr)
        if ks == "BackSpace":
            setattr(self, attr, cur[:-1])
        elif ks in ("Return", "KP_Enter"):
            if self.focused_input == "browser_search":
                self.run_browser_search()
            elif self.focused_input == "terminal_cmd":
                self.exec_terminal()
        elif ch and 32 <= ord(ch) < 127:
            setattr(self, attr, cur + ch)
        self.click_seq += 1
        self.redraw()

    def redraw(self) -> None:
        self.rebuild_scene()
        if self.canvas is None:
            return
        c = self.canvas
        p = self.pal
        c.delete("all")
        fnt = (self.font_family, 8)
        fnt_sm = (self.font_family, 7)

        # Desktop wallpaper background
        c.create_rectangle(0, 0, 256, 236, fill=p["bg"], outline="")

        # Draw desktop launcher icons (`x=3..45`)
        for hb in self.hitboxes:
            if hb.win_id != "desktop":
                continue
            outline = "#f5c542" if hb.cid == self.active_cid else p["border"]
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=p["panel"], outline=outline)
            # Mini desktop glyph at top + label below
            cx = hb.x + hb.w // 2
            c.create_rectangle(cx - 5, hb.y + 3, cx + 5, hb.y + 11, fill=p["accent"], outline="")
            c.create_text(hb.x + 3, hb.y + 21, text=hb.label[:7], anchor="w", fill=p["fg"], font=fnt_sm)

        # Draw visible windows in Z-order
        vis_wins = [w for w in self.windows if not w.minimized]
        for w_idx, win in enumerate(vis_wins):
            is_top = (w_idx == len(vis_wins) - 1)
            wx, wy, ww, wh = win.x, win.y, win.w, win.h
            win_bg = p["widget"] if is_top else p["panel"]
            hdr_bg = p["accent"] if is_top else p["chrome"]
            c.create_rectangle(wx, wy, wx + ww, wy + wh, fill=win_bg, outline=p["border"], width=1)
            c.create_rectangle(wx, wy, wx + ww, wy + 18, fill=hdr_bg, outline=p["border"])
            c.create_text(wx + 5, wy + 9, text=win.title, anchor="w", fill="#ffffff" if is_top else p["muted"], font=fnt)

            if not is_top:
                c.create_text(wx + 8, wy + 30, text=f"[{win.win_id} unfocused]", anchor="w", fill=p["muted"], font=fnt_sm)
                continue

            # App-specific interior text before interactive controls
            if win.win_id == "browser":
                if self.browser_mode == "home":
                    c.create_text(wx + 8, wy + 68, text="MiniGoogle Search:", anchor="w", fill=p["fg"], font=fnt_sm)
                elif self.browser_mode == "article" and self.current_article is not None:
                    c.create_rectangle(wx + 6, wy + 68, wx + ww - 6, wy + 144, fill=p["field_bg"], outline=p["border"])
                    c.create_text(wx + 10, wy + 80, text=self.current_article.slug, anchor="w", fill=p["accent"], font=fnt)
                    c.create_text(wx + 10, wy + 98, text=self.current_article.url, anchor="w", fill=p["muted"], font=fnt_sm)
                    c.create_text(wx + 10, wy + 120, text=self.current_article.fact[:28], anchor="w", fill=p["fg"], font=fnt_sm)
            elif win.win_id == "notes":
                c.create_rectangle(wx + 6, wy + 54, wx + ww - 6, wy + 148, fill=p["field_bg"], outline=p["border"])
                c.create_text(wx + 10, wy + 70, text=(self.notes_text or "(empty note)")[:28], anchor="w", fill=p["fg"], font=fnt_sm)
                c.create_text(wx + 10, wy + 92, text=f"clipboard: {self.clipboard[:20]}", anchor="w", fill=p["muted"], font=fnt_sm)
            elif win.win_id == "terminal":
                c.create_rectangle(wx + 6, wy + 52, wx + ww - 6, wy + 196, fill="#0c0e12", outline=p["border"])
                c.create_text(wx + 10, wy + 68, text=f"$ {self.terminal_cmd}", anchor="w", fill="#98c379", font=fnt_sm)
                c.create_text(wx + 10, wy + 88, text=self.terminal_out[:28], anchor="w", fill="#e5c07b", font=fnt_sm)
            elif win.win_id == "files":
                c.create_rectangle(wx + 6, wy + 24, wx + ww - 6, wy + 158, fill=p["field_bg"], outline=p["border"])
                for fi, fn in enumerate(self.files_items[:5]):
                    c.create_text(wx + 10, wy + 36 + fi * 18, text=fn, anchor="w", fill=p["fg"], font=fnt_sm)
            elif win.win_id == "alert":
                c.create_text(wx + 10, wy + 34, text="System popup blocking desktop!", anchor="w", fill=p["fg"], font=fnt_sm)
                c.create_text(wx + 10, wy + 50, text="Click cancel to dismiss.", anchor="w", fill=p["muted"], font=fnt_sm)

            # Foreground window controls
            for hb in self.hitboxes:
                if hb.win_id != win.win_id:
                    continue
                self._draw_control(c, hb, fnt, fnt_sm)

        # Bottom OS Taskbar (`y=236..256`)
        c.create_rectangle(0, 235, 256, 256, fill=p["chrome"], outline=p["border"])
        for hb in self.hitboxes:
            if hb.win_id == "taskbar":
                self._draw_control(c, hb, fnt, fnt_sm)

        if self.click_seq > 0:
            col = "#98c379" if (self.click_seq % 2 == 1) else "#61afef"
            c.create_rectangle(250, 1, 255, 5, fill=col, outline="")

    def _draw_control(self, c: Any, hb: DesktopHitBox, fnt: Any, fnt_sm: Any) -> None:
        p = self.pal
        outline = "#f5c542" if hb.cid == self.active_cid else (p["accent"] if hb.state in ("focused", "selected") else p["border"])
        fg = p["muted"] if hb.state == "disabled" else p["fg"]

        if hb.role == "button":
            bg = p["panel"] if hb.state == "disabled" else p["accent"]
            bfg = p["muted"] if hb.state == "disabled" else "#ffffff"
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=bg, outline=outline)
            c.create_text(hb.x + 5, hb.y + hb.h // 2, text=hb.label, anchor="w", fill=bfg, font=fnt_sm if hb.h < 18 else fnt)

        elif hb.role == "tab":
            bg = p["widget"] if hb.state == "selected" else p["chrome"]
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=bg, outline=outline)
            if hb.state == "selected":
                c.create_rectangle(hb.x + 1, hb.y + hb.h - 2, hb.x + hb.w - 1, hb.y + hb.h, fill=p["accent"], outline="")
            c.create_text(hb.x + 4, hb.y + hb.h // 2, text=hb.label[:7], anchor="w", fill=fg, font=fnt_sm)

        elif hb.role == "icon":
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=p["panel"], outline=outline)
            gx, gy = hb.x + 5, hb.y + hb.h // 2
            c.create_line(gx - 2, gy, gx + 2, gy, fill=p["accent"], width=2)
            c.create_text(hb.x + 10, gy, text=hb.label[:6], anchor="w", fill=fg, font=fnt_sm)

        elif hb.role == "field":
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=p["field_bg"], outline=outline)
            val_map = {"br_search_fld": self.browser_query, "nt_body": self.notes_text, "tm_cmd": self.terminal_cmd}
            txt_v = val_map.get(hb.cid, "")
            disp = f"{hb.label}: {txt_v}" if txt_v else hb.label
            c.create_text(hb.x + 4, hb.y + hb.h // 2, text=disp[: max(6, (hb.w - 8) // 6)], anchor="w", fill=fg, font=fnt)
            if hb.state == "focused":
                c.create_line(hb.x + hb.w - 4, hb.y + 3, hb.x + hb.w - 4, hb.y + hb.h - 3, fill=p["accent"])

        elif hb.role == "menu":
            # Used for SERP result cards & topic rows!
            c.create_rectangle(hb.x, hb.y, hb.x + hb.w, hb.y + hb.h, fill=p["field_bg"], outline=outline)
            c.create_text(hb.x + 5, hb.y + hb.h // 2, text=hb.label, anchor="w", fill=fg, font=fnt)

    def pump(self, cycles: int = 7, delay_s: float = 0.004) -> None:
        if self.root is None:
            return
        for _ in range(cycles):
            self.root.update_idletasks()
            self.root.update()
            if delay_s > 0:
                time.sleep(delay_s)


# ---------------------------------------------------------------------------
# 3. 40 Deterministic "Desktop & Googling" Tasks (4 categories x 10)
# ---------------------------------------------------------------------------

def build_desktop_eval_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        # === Category 1: desktop_nav (10 tasks: icons, taskbar, z-order, terminal, files) ===
        {
            "id": "dn-001", "category": "desktop_nav", "theme": "dark",
            "instruction": "launch Browser from desktop shortcut icon",
            "setup": {"open_windows": []},
            "plan": [("click the browser icon", {})],
            "success": {"check": "top_window", "val": "browser"},
            "budget": {"max_steps": 5, "max_seconds": 25},
        },
        {
            "id": "dn-002", "category": "desktop_nav", "theme": "light",
            "instruction": "launch Terminal from desktop icon, run df, and verify disk output",
            "setup": {"open_windows": []},
            "plan": [("click the terminal icon", {}), ("click the command input box", {"type_text": "df"}), ("click the run button", {})],
            "success": {"check": "file_substr", "rel": "terminal/last_out.txt", "substr": "DISK_OK"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "dn-003", "category": "desktop_nav", "theme": "dark",
            "instruction": "switch from foreground Browser to Notes using the OS taskbar",
            "setup": {"open_windows": ["notes", "browser"]},
            "plan": [("open the notes tab", {})],
            "success": {"check": "top_window", "val": "notes"},
            "budget": {"max_steps": 5, "max_seconds": 25},
        },
        {
            "id": "dn-004", "category": "desktop_nav", "theme": "high_contrast",
            "instruction": "open Files from taskbar and delete .iso images",
            "setup": {"open_windows": ["browser"]},
            "plan": [("open the files tab", {}), ("click the delete button", {})],
            "success": {"check": "file_missing", "rel": "files/temp.iso"},
            "budget": {"max_steps": 7, "max_seconds": 25},
        },
        {
            "id": "dn-005", "category": "desktop_nav", "theme": "light",
            "instruction": "close foreground Notes window to reveal Browser underneath",
            "setup": {"open_windows": ["browser", "notes"]},
            "plan": [("click the close button", {})],
            "success": {"check": "top_window", "val": "browser"},
            "budget": {"max_steps": 5, "max_seconds": 25},
        },
        {
            "id": "dn-006", "category": "desktop_nav", "theme": "dark",
            "instruction": "launch Notes from desktop, type memo2026, and save to disk",
            "setup": {"open_windows": []},
            "plan": [("click the notes icon", {}), ("click the notes input box", {"type_text": "memo2026"}), ("click the save button", {})],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "memo2026"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "dn-007", "category": "desktop_nav", "theme": "dark",
            "instruction": "switch to Terminal via taskbar, run ports check, and paste output into Notes",
            "setup": {"open_windows": ["notes", "browser"]},
            "plan": [
                ("open the terminal tab", {}),
                ("click the command input box", {"type_text": "ports"}),
                ("click the run button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "LISTEN:22,80,5432"},
            "budget": {"max_steps": 12, "max_seconds": 30},
        },
        {
            "id": "dn-008", "category": "desktop_nav", "theme": "light",
            "instruction": "launch Files from desktop icon and export file index JSON",
            "setup": {"open_windows": []},
            "plan": [("click the files icon", {}), ("click the export button", {})],
            "success": {"check": "file_exists", "rel": "files/index.json"},
            "budget": {"max_steps": 7, "max_seconds": 25},
        },
        {
            "id": "dn-009", "category": "desktop_nav", "theme": "high_contrast",
            "instruction": "open Terminal via taskbar and run uname check",
            "setup": {"open_windows": ["files"]},
            "plan": [("open the terminal tab", {}), ("click the command input box", {"type_text": "uname"}), ("click the run button", {})],
            "success": {"check": "file_substr", "rel": "terminal/last_out.txt", "substr": "KERNEL:Linux"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "dn-010", "category": "desktop_nav", "theme": "dark",
            "instruction": "open start menu on OS taskbar then focus Browser tab",
            "setup": {"open_windows": ["notes"]},
            "plan": [("click the start button", {}), ("open the browser tab", {})],
            "success": {"check": "top_window", "val": "browser"},
            "budget": {"max_steps": 6, "max_seconds": 25},
        },

        # === Category 2: google_search (10 tasks: open-vocab SERP reading & bookmarking) ===
        {
            "id": "gs-001", "category": "google_search", "theme": "dark",
            "instruction": "google postgres default port and open the postgres 5432 result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "postgres port"}), ("click the submit button", {}), ("choose postgres 5432 from the menu", {})],
            "success": {"check": "article_fact", "substr": "POSTGRES_PORT=5432"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-002", "category": "google_search", "theme": "light",
            "instruction": "google chmod 755 permissions and bookmark the chmod 755 guide",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "chmod 755"}),
                ("click the submit button", {}),
                ("choose chmod 755 from the menu", {}),
                ("click the bookmark button", {}),
            ],
            "success": {"check": "bookmark_fact", "substr": "CHMOD_755=rwxr-xr-x"},
            "budget": {"max_steps": 10, "max_seconds": 25},
        },
        {
            "id": "gs-003", "category": "google_search", "theme": "dark",
            "instruction": "google http 429 status code and open http 429 result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "http 429"}), ("click the submit button", {}), ("choose http 429 from the menu", {})],
            "success": {"check": "article_fact", "substr": "HTTP_429=RATE_LIMIT"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-004", "category": "google_search", "theme": "high_contrast",
            "instruction": "google git bisect regression search and open git bisect article",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "git bisect"}), ("click the submit button", {}), ("choose git bisect from the menu", {})],
            "success": {"check": "article_fact", "substr": "GIT_BUGHUNT=git bisect"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-005", "category": "google_search", "theme": "light",
            "instruction": "google sqlite wal concurrency mode and open sqlite wal result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "sqlite wal"}), ("click the submit button", {}), ("choose sqlite wal from the menu", {})],
            "success": {"check": "article_fact", "substr": "SQLITE_CONCUR=WAL"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-006", "category": "google_search", "theme": "dark",
            "instruction": "google python gil lock and bookmark python gil article",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "python gil"}),
                ("click the submit button", {}),
                ("choose python gil from the menu", {}),
                ("click the bookmark button", {}),
            ],
            "success": {"check": "bookmark_fact", "substr": "PYTHON_LOCK=GIL"},
            "budget": {"max_steps": 10, "max_seconds": 25},
        },
        {
            "id": "gs-007", "category": "google_search", "theme": "light",
            "instruction": "google tar extract gzip command and open tar extract result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "tar extract"}), ("click the submit button", {}), ("choose tar extract from the menu", {})],
            "success": {"check": "article_fact", "substr": "TAR_EXTRACT=tar -xzf"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-008", "category": "google_search", "theme": "high_contrast",
            "instruction": "google redis cache port and open redis 6379 result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "redis port"}), ("click the submit button", {}), ("choose redis 6379 from the menu", {})],
            "success": {"check": "article_fact", "substr": "REDIS_PORT=6379"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-009", "category": "google_search", "theme": "dark",
            "instruction": "google dijkstra shortest path and open dijkstra path article",
            "setup": {"open_windows": ["browser"]},
            "plan": [("click the search input box", {"type_text": "dijkstra"}), ("click the submit button", {}), ("choose dijkstra path from the menu", {})],
            "success": {"check": "article_fact", "substr": "SHORTEST_PATH=DIJKSTRA"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "gs-010", "category": "google_search", "theme": "light",
            "instruction": "use quick topic chip sqlite wal on Browser home then open sqlite wal result",
            "setup": {"open_windows": ["browser"]},
            "plan": [("choose sqlite wal from the menu", {}), ("choose sqlite wal from the menu", {})],
            "success": {"check": "article_fact", "substr": "SQLITE_CONCUR=WAL"},
            "budget": {"max_steps": 7, "max_seconds": 25},
        },

        # === Category 3: research_workflow (10 multi-app 6-7 step Google -> Copy -> Notes -> Save tasks) ===
        {
            "id": "rw-001", "category": "research_workflow", "theme": "dark",
            "instruction": "google dns port 53, copy fact from article, switch to Notes, paste and save",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "dns port"}),
                ("click the submit button", {}),
                ("choose dns port 53 from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "DNS_PORT=53"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-002", "category": "research_workflow", "theme": "light",
            "instruction": "launch Browser from desktop, google ssh port 22, copy fact, and save in Notes",
            "setup": {"open_windows": []},
            "plan": [
                ("click the browser icon", {}),
                ("click the search input box", {"type_text": "ssh port"}),
                ("click the submit button", {}),
                ("choose ssh port 22 from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "SSH_PORT=22"},
            "budget": {"max_steps": 14, "max_seconds": 35},
        },
        {
            "id": "rw-003", "category": "research_workflow", "theme": "dark",
            "instruction": "google git reflog recovery, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "git reflog"}),
                ("click the submit button", {}),
                ("choose git reflog from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "GIT_RECOVER=git reflog"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-004", "category": "research_workflow", "theme": "high_contrast",
            "instruction": "google asyncio gather coroutines, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "asyncio gather"}),
                ("click the submit button", {}),
                ("choose asyncio gather from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "ASYNC_JOIN=asyncio.gather"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-005", "category": "research_workflow", "theme": "light",
            "instruction": "google sqlite vacuum compact, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "sqlite vacuum"}),
                ("click the submit button", {}),
                ("choose sqlite vacuum from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "SQLITE_COMPACT=VACUUM"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-006", "category": "research_workflow", "theme": "dark",
            "instruction": "google sha256 digest length, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "sha256 bits"}),
                ("click the submit button", {}),
                ("choose sha256 bits from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "SHA256_HEX=64"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-007", "category": "research_workflow", "theme": "light",
            "instruction": "google kill sigterm signal number, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "sigterm"}),
                ("click the submit button", {}),
                ("choose kill sigterm from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "SIGTERM_NUM=15"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-008", "category": "research_workflow", "theme": "high_contrast",
            "instruction": "google cron schedule fields, copy fact, and save to Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "cron five"}),
                ("click the submit button", {}),
                ("choose cron timer from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "CRON_FIELDS=5"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },
        {
            "id": "rw-009", "category": "research_workflow", "theme": "dark",
            "instruction": "switch from Terminal to Browser via taskbar, google tcp handshake, copy and save in Notes",
            "setup": {"open_windows": ["browser", "terminal"]},
            "plan": [
                ("open the browser tab", {}),
                ("click the search input box", {"type_text": "tcp handshake"}),
                ("click the submit button", {}),
                ("choose tcp syn ack from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "TCP_HANDSHAKE=SYN_SYNACK_ACK"},
            "budget": {"max_steps": 14, "max_seconds": 35},
        },
        {
            "id": "rw-010", "category": "research_workflow", "theme": "light",
            "instruction": "google utf8 max bytes, copy fact, and save in Notes",
            "setup": {"open_windows": ["browser"]},
            "plan": [
                ("click the search input box", {"type_text": "utf8 bytes"}),
                ("click the submit button", {}),
                ("choose utf8 variable from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "UTF8_MAX_BYTES=4"},
            "budget": {"max_steps": 12, "max_seconds": 35},
        },

        # === Category 4: adversarial_desktop (10 tasks: Rank-1 Ad Traps, Blocking Popups, OOD Fonts, Frozen Decoys) ===
        {
            "id": "ad-001", "category": "adversarial_desktop", "theme": "dark",
            "instruction": "skip Rank-1 sponsored ad on SERP and open organic postgres 5432 result",
            "setup": {"open_windows": ["browser"], "ad_trap": True},
            "plan": [("click the search input box", {"type_text": "postgres port"}), ("click the submit button", {}), ("choose postgres 5432 from the menu", {})],
            "success": {"check": "article_fact", "substr": "POSTGRES_PORT=5432"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "ad-002", "category": "adversarial_desktop", "theme": "light",
            "instruction": "skip Rank-1 sponsored ad on SERP and open organic chmod 600 result",
            "setup": {"open_windows": ["browser"], "ad_trap": True},
            "plan": [("click the search input box", {"type_text": "chmod 600"}), ("click the submit button", {}), ("choose chmod 600 from the menu", {})],
            "success": {"check": "article_fact", "substr": "CHMOD_600=rw-------"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "ad-003", "category": "adversarial_desktop", "theme": "high_contrast",
            "instruction": "skip Rank-1 sponsored ad on SERP and bookmark organic https 443 article",
            "setup": {"open_windows": ["browser"], "ad_trap": True},
            "plan": [
                ("click the search input box", {"type_text": "https port"}),
                ("click the submit button", {}),
                ("choose https 443 from the menu", {}),
                ("click the bookmark button", {}),
            ],
            "success": {"check": "bookmark_fact", "substr": "HTTPS_PORT=443"},
            "budget": {"max_steps": 10, "max_seconds": 25},
        },
        {
            "id": "ad-004", "category": "adversarial_desktop", "theme": "dark",
            "instruction": "dismiss blocking Alert modal popup then google mysql 3306 port",
            "setup": {"open_windows": ["browser", "alert"]},
            "plan": [
                ("click the cancel button", {}),
                ("click the search input box", {"type_text": "mysql port"}),
                ("click the submit button", {}),
                ("choose mysql 3306 from the menu", {}),
            ],
            "success": {"check": "article_fact", "substr": "MYSQL_PORT=3306"},
            "budget": {"max_steps": 10, "max_seconds": 25},
        },
        {
            "id": "ad-005", "category": "adversarial_desktop", "theme": "light",
            "instruction": "dismiss blocking Alert popup via titlebar close button then run whoami in Terminal",
            "setup": {"open_windows": ["terminal", "alert"]},
            "plan": [
                ("click the close button", {}),
                ("click the command input box", {"type_text": "whoami"}),
                ("click the run button", {}),
            ],
            "success": {"check": "file_substr", "rel": "terminal/last_out.txt", "substr": "USER:agent03"},
            "budget": {"max_steps": 9, "max_seconds": 25},
        },
        {
            "id": "ad-006", "category": "adversarial_desktop", "theme": "nord_slate", "font": "Liberation Sans",
            "instruction": "google btree index on unseen nord_slate + Liberation Sans and open btree index",
            "setup": {"open_windows": ["browser"], "ad_trap": True},
            "plan": [("click the search input box", {"type_text": "btree index"}), ("click the submit button", {}), ("choose btree index from the menu", {})],
            "success": {"check": "article_fact", "substr": "DB_INDEX=BTREE_O_LOG_N"},
            "budget": {"max_steps": 8, "max_seconds": 25},
        },
        {
            "id": "ad-007", "category": "adversarial_desktop", "theme": "solarized_light", "font": "FreeSans",
            "instruction": "google ed25519 key size on unseen solarized + FreeSans and save fact to Notes",
            "setup": {"open_windows": ["browser"], "ad_trap": True},
            "plan": [
                ("click the search input box", {"type_text": "ed25519 key"}),
                ("click the submit button", {}),
                ("choose ed25519 key from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "ED25519_PUB_BYTES=32"},
            "budget": {"max_steps": 14, "max_seconds": 35},
        },
        {
            "id": "ad-008", "category": "adversarial_desktop", "theme": "dark",
            "instruction": "recover via exclude_box when primary copy button on article view is frozen",
            "setup": {"open_windows": ["browser"], "stale_copy_decoy": True},
            "plan": [
                ("click the search input box", {"type_text": "grep recursive"}),
                ("click the submit button", {}),
                ("choose grep recursive from the menu", {}),
                ("click the copy button", {}),
                ("open the notes tab", {}),
                ("click the paste button", {}),
                ("click the save button", {}),
            ],
            "success": {"check": "file_substr", "rel": "notes/saved_note.txt", "substr": "GREP_REC=grep -rn"},
            "budget": {"max_steps": 14, "max_seconds": 35},
        },
        {
            "id": "ad-009", "category": "adversarial_desktop", "theme": "nord_slate", "font": "FreeSans",
            "instruction": "dismiss popup, skip Rank-1 ad, google http 301 redirect, and bookmark on OOD theme",
            "setup": {"open_windows": ["browser", "alert"], "ad_trap": True},
            "plan": [
                ("click the cancel button", {}),
                ("click the search input box", {"type_text": "http 301"}),
                ("click the submit button", {}),
                ("choose http 301 from the menu", {}),
                ("click the bookmark button", {}),
            ],
            "success": {"check": "bookmark_fact", "substr": "HTTP_301=PERMANENT_REDIRECT"},
            "budget": {"max_steps": 12, "max_seconds": 30},
        },
        {
            "id": "ad-010", "category": "adversarial_desktop", "theme": "solarized_light", "font": "Liberation Sans",
            "instruction": "click unfocused background Browser behind Notes, google lsof listen, and open result",
            "setup": {"open_windows": ["browser", "notes"], "ad_trap": True},
            "plan": [
                ("open the browser tab", {}),
                ("click the search input box", {"type_text": "lsof listen"}),
                ("click the submit button", {}),
                ("choose lsof listen from the menu", {}),
            ],
            "success": {"check": "article_fact", "substr": "LSOF_NET=lsof -i"},
            "budget": {"max_steps": 10, "max_seconds": 30},
        },
    ]
    return tasks


def check_desktop_success(spec: dict[str, Any], ws: str, os_env: MockDesktopOS) -> bool:
    chk = spec["check"]
    if chk == "top_window":
        tw = os_env.top_window()
        return tw is not None and tw.win_id == spec["val"]
    if chk == "file_exists":
        return os.path.exists(os.path.join(ws, spec["rel"]))
    if chk == "file_missing":
        return not os.path.exists(os.path.join(ws, spec["rel"]))
    if chk == "file_substr":
        p = os.path.join(ws, spec["rel"])
        return os.path.exists(p) and (spec["substr"] in open(p).read())
    if chk == "article_fact":
        p = os.path.join(ws, "browser", "opened_article.json")
        if not os.path.exists(p):
            return False
        d = json.load(open(p))
        return (not d.get("is_ad", False)) and (spec["substr"] in d.get("fact", ""))
    if chk == "bookmark_fact":
        p = os.path.join(ws, "browser", "bookmarks.json")
        if not os.path.exists(p):
            return False
        d = json.load(open(p))
        return (not d.get("is_ad", False)) and (spec["substr"] in d.get("last_fact", ""))
    raise ValueError(f"Unknown desktop check: {chk}")


def run_desktop_suite(
    checkpoint_dir: str | None = "checkpoints/grip_v2",
    grounder_override: Any = None,
    task_ids: list[str] | None = None,
    task_limit: int | None = None,
    out_path: str = "/home/user/fleet/agents/03-eyes/desktop_scoreboard.json",
    strips_dir: str = "/home/user/fleet/agents/03-eyes/desktop_failure_strips",
) -> dict[str, Any]:
    import tkinter as tk
    from deskhands.capture import Capture
    from deskhands.core import DesktopHands

    tasks = build_desktop_eval_tasks()
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t["id"] in wanted]
    if task_limit is not None:
        tasks = tasks[:task_limit]

    grounder = grounder_override if grounder_override is not None else load_exported_grounder(checkpoint_dir or "checkpoints/grip_v2")
    grip_step = int(getattr(grounder, "step", 3550))
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        grip_step = int(json.load(open(os.path.join(checkpoint_dir, "config.json"))).get("step", grip_step))

    _ensure_xvfb_display()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_tasks = 0
    recovered_tasks = 0

    root = tk.Tk()
    try:
        root.overrideredirect(True)
        root.geometry("256x256+0+0")
        root.update()

        with Capture() as cap:
            hands = DesktopHands()
            try:
                for task in tasks:
                    t0 = time.perf_counter()
                    max_steps = int(task["budget"]["max_steps"])
                    max_sec = float(task["budget"]["max_seconds"])
                    trace: list[dict[str, Any]] = []
                    steps_used = 0
                    had_retry = False

                    with tempfile.TemporaryDirectory(prefix=f"nb_desk_{task['id']}_") as ws:
                        setup = task["setup"]
                        os_env = MockDesktopOS(
                            root,
                            ws,
                            theme=task.get("theme", "dark"),
                            font_family=task.get("font", "DejaVu Sans"),
                            search_ad_trap=bool(setup.get("ad_trap", False)),
                            stale_copy_decoy=bool(setup.get("stale_copy_decoy", False)),
                        )
                        for wid in setup.get("open_windows", []):
                            os_env.open_window(wid)
                        os_env.redraw()
                        os_env.pump()

                        plan = task["plan"]
                        plan_idx = 0
                        exclude_boxes: list[tuple[int, int, int, int]] = []
                        final_tried: list[tuple[int, int, int, int]] = []

                        while plan_idx < len(plan) and steps_used < max_steps and (time.perf_counter() - t0) < max_sec:
                            step_instr, step_meta = plan[plan_idx]
                            rgb_before = np.asarray(cap.full_screen().image, dtype=np.uint8)

                            is_last = (plan_idx == len(plan) - 1)
                            active_ex = list(exclude_boxes) + (final_tried if is_last else [])
                            pred = grounder.ground(rgb_before, step_instr, exclude_box=active_ex or None)
                            px, py = int(pred["x"]), int(pred["y"])
                            steps_used += 1
                            if is_last:
                                final_tried.append((max(0, px - 22), max(0, py - 10), 44, 20))

                            hands.perform({"type": "click", "x": px, "y": py})
                            os_env.pump(cycles=6, delay_s=0.004)
                            if "type_text" in step_meta:
                                hands.perform({"type": "type", "text": str(step_meta["type_text"])})
                                os_env.pump(cycles=6, delay_s=0.004)

                            rgb_after = np.asarray(cap.full_screen().image, dtype=np.uint8)
                            trace.append({"instr": step_instr, "x": px, "y": py, "action": pred.get("action", "click"), "rgb": rgb_before})

                            if check_desktop_success(task["success"], ws, os_env):
                                break

                            changed_px = int((np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)).max(axis=-1) > 4).sum())
                            if changed_px < 3:
                                had_retry = True
                                exclude_boxes.append((max(0, px - 22), max(0, py - 11), 44, 22))
                            else:
                                plan_idx += 1
                                exclude_boxes.clear()
                                if plan_idx >= len(plan) and steps_used < max_steps and not check_desktop_success(task["success"], ws, os_env):
                                    had_retry = True
                                    # If browser accidentally opened an Ad article on SERP click, click Back or retry SERP!
                                    plan_idx = len(plan) - 1

                        passed = check_desktop_success(task["success"], ws, os_env)
                        elapsed = time.perf_counter() - t0
                        if had_retry:
                            retry_tasks += 1
                            if passed:
                                recovered_tasks += 1

                        if not passed:
                            last = trace[-1] if trace else {"x": -1, "y": -1, "instr": ""}
                            hb = os_env.hitbox_at(int(last["x"]), int(last["y"]))
                            hit_desc = f"{hb.role}:{hb.label}" if hb else "background"
                            why = f"clicked {hit_desc} at ({last['x']},{last['y']}) on '{last['instr']}'; check '{task['success']['check']}' failed"
                            sp = os.path.join(strips_dir, f"{task['id']}.png")
                            save_failure_strip(task["id"], trace, why, sp)
                            failures.append({"id": task["id"], "category": task["category"], "why": why, "strip": sp})

                        records.append({"id": task["id"], "category": task["category"], "passed": passed, "steps": steps_used, "seconds": round(elapsed, 4)})
            finally:
                hands.close()
    finally:
        try:
            root.update_idletasks()
            root.destroy()
        except Exception:
            pass

    by_cat: dict[str, Any] = {}
    for cat in sorted({r["category"] for r in records}):
        sub = [r for r in records if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "passed": sum(1 for r in sub if r["passed"]),
            "success": round(sum(1 for r in sub if r["passed"]) / max(1, len(sub)), 4),
            "median_steps": float(statistics.median(r["steps"] for r in sub)),
            "median_seconds": round(float(statistics.median(r["seconds"] for r in sub)), 4),
        }

    n_tot = len(records)
    n_ok = sum(1 for r in records if r["passed"])
    sb = {
        "commit": _git_head_short(),
        "model": {"grip_step": grip_step, "checkpoint": checkpoint_dir or "override"},
        "n": n_tot,
        "passed": n_ok,
        "success": round(n_ok / max(1, n_tot), 4),
        "by_category": by_cat,
        "median_steps": float(statistics.median(r["steps"] for r in records)) if records else 0,
        "median_seconds": round(float(statistics.median(r["seconds"] for r in records)), 4) if records else 0.0,
        "ram_peak_mb": round(peak_rss_mb(), 1),
        "recovery_stats": {
            "tasks_triggering_retry": retry_tasks,
            "tasks_recovered_to_done": recovered_tasks,
            "recovery_rate": round(recovered_tasks / max(1, retry_tasks), 4) if retry_tasks else 1.0,
        },
        "failures": failures,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(sb, f, indent=2)
    return sb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grip_v2")
    ap.add_argument("--out", default="/home/user/fleet/agents/03-eyes/desktop_scoreboard.json")
    ap.add_argument("--strips-dir", default="/home/user/fleet/agents/03-eyes/desktop_failure_strips")
    ap.add_argument("--composite-strip", default="/home/user/fleet/agents/03-eyes/galleries/m8_desktop_failures.png")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)

    sb = run_desktop_suite(checkpoint_dir=args.checkpoint, out_path=args.out, strips_dir=args.strips_dir)
    if args.composite_strip and sb["failures"]:
        save_composite_failure_gallery(sb["failures"], args.composite_strip)

    print(
        f"[mock_desktop] n={sb['n']} passed={sb['passed']}/{sb['n']} ({sb['success']*100:.2f}%) | "
        f"med_steps={sb['median_steps']} | med_sec={sb['median_seconds']:.3f}s | ram={sb['ram_peak_mb']}MB",
        flush=True,
    )
    for cat, cv in sb["by_category"].items():
        print(f"  - {cat:20s}: {cv['passed']:2d}/{cv['n']:2d} ({cv['success']*100:.1f}%) med_steps={cv['median_steps']}", flush=True)
    if sb["failures"]:
        print(f"  failures ({len(sb['failures'])}):", flush=True)
        for fl in sb["failures"][:15]:
            print(f"    * {fl['id']}: {fl['why']}", flush=True)


if __name__ == "__main__":
    main()
