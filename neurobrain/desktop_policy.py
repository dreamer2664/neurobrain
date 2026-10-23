"""Autonomous From-Scratch `DesktopBrainPolicy` + Unscripted X11 Goal Suite (M10).

Pairs a compact from-scratch neural policy (`DesktopBrainPolicy`, ~580k params) with
`EyesGrounder` (`checkpoints/grip_v3`, 1.57M params) so the brain can execute high-level
natural-language desktop & Google research goals end-to-end on live headless X11 (`Xvfb` +
`tkinter` + `Capture` + `DesktopHands`) with ZERO scripted sub-steps, ZERO DOM access,
ZERO regex/rule cheats at inference, and ZERO pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from deskhands.capture import Capture
from deskhands.core import DesktopHands

from .diagnose_grip import peak_rss_mb
from .eval_suite import PALETTES
from .export_grounder import load_exported_grounder
from .mock_desktop import (
    KNOWLEDGE_DOCS,
    MockDesktopOS,
    _ensure_xvfb_display,
    check_desktop_success,
)
from .ui_world import CHAR_VOCAB

CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate(CHAR_VOCAB)}

MAX_GOAL_CHARS = 96
HIST_LEN = 8

PRIMITIVE_OPS: tuple[str, ...] = (
    "DONE",
    "DISMISS_ALERT",
    "OPEN_BROWSER_ICON",
    "FOCUS_SEARCH",
    "TYPE_SEARCH_QUERY",
    "SUBMIT_SEARCH",
    "CHOOSE_SERP_RESULT",
    "CLICK_BROWSER_BACK",
    "COPY_ARTICLE_FACT",
    "BOOKMARK_ARTICLE",
    "OPEN_NOTES_TAB",
    "PASTE_INTO_NOTES",
    "SAVE_NOTES",
    "OPEN_TERMINAL_ICON",
    "FOCUS_COMMAND",
    "TYPE_COMMAND",
    "RUN_COMMAND",
    "OPEN_FILES_ICON",
    "DELETE_TEMP_FILES",
    "EXPORT_FILES_INDEX",
)
OP_TO_ID = {name: i for i, name in enumerate(PRIMITIVE_OPS)}


def encode_goal_chars(goal: str, max_len: int = MAX_GOAL_CHARS) -> list[int]:
    s = goal.lower().strip()[:max_len]
    ids = [CHAR_TO_ID.get(ch, 0) for ch in s]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def build_span_char_mask(goal: str, span_substr: str, max_len: int = MAX_GOAL_CHARS) -> np.ndarray:
    """Create binary character mask `[max_len]` marking where `span_substr` appears inside `goal`."""
    mask = np.zeros((max_len,), dtype=np.float32)
    g_low = goal.lower().strip()[:max_len]
    sub = span_substr.lower().strip()
    if not sub:
        return mask
    idx = g_low.find(sub)
    if idx >= 0:
        end = min(max_len, idx + len(sub))
        mask[idx:end] = 1.0
    return mask


def decode_char_span(goal: str, char_probs: np.ndarray, threshold: float = 0.45) -> str:
    """Decode highest-mass contiguous word-aligned span from neural per-character probabilities."""
    g_clean = goal.lower().strip()[:MAX_GOAL_CHARS]
    n = len(g_clean)
    if n == 0:
        return ""
    p = char_probs[:n]

    # Score every word token in `g_clean` by its mean character probability, then merge adjacent active words!
    words: list[tuple[int, int, str, float]] = []
    cur_start: int | None = None
    for idx, ch in enumerate(g_clean):
        if ch != " " and cur_start is None:
            cur_start = idx
        elif ch == " " and cur_start is not None:
            w_str = g_clean[cur_start:idx]
            words.append((cur_start, idx, w_str, float(p[cur_start:idx].mean())))
            cur_start = None
    if cur_start is not None:
        words.append((cur_start, n, g_clean[cur_start:n], float(p[cur_start:n].mean())))
    if not words:
        return ""

    # Group consecutive words whose mean char prob >= threshold into contiguous islands
    best_words: list[str] = []
    best_mass = -1.0
    cur_island: list[str] = []
    cur_mass = 0.0
    for _, _, w_str, w_prob in words:
        if w_prob >= threshold:
            cur_island.append(w_str)
            cur_mass += w_prob
            if cur_mass > best_mass:
                best_mass = cur_mass
                best_words = list(cur_island)
        else:
            cur_island = []
            cur_mass = 0.0

    if not best_words:
        best_w = max(words, key=lambda item: item[3])[2]
        best_words = [best_w]

    raw = " ".join(best_words)
    cleaned = "".join(ch for ch in raw if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == " ")
    return " ".join(cleaned.split())


class DesktopBrainPolicy(nn.Module):
    """Compact from-scratch Visuomotor Desktop & Search Policy (`~565k` parameters).

    Maps `(128x128 X11 screenshot, raw goal string, episodic history)` ->
    `(next primitive op, open-vocab character span for typing/SERP selection)`.
    """

    def __init__(self, d_goal: int = 128, d_vis: int = 160, d_hist: int = 80) -> None:
        super().__init__()
        # 1. Vision encoder on 128x128 RGB (`(B, 3, 128, 128)` -> spatial `4x4` + global -> `d_vis`)
        self.vis_cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
            nn.Conv2d(80, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vis_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 16, d_vis),
            nn.LayerNorm(d_vis),
            nn.SiLU(inplace=True),
        )

        # 2. Character-level Goal Encoder (`(B, 96)` -> per-char `(B, 96, 128)` + pooled `(B, 128)`)
        self.char_emb = nn.Embedding(len(CHAR_VOCAB) + 1, 48, padding_idx=0)
        self.char_conv = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.goal_gru = nn.GRU(96, d_goal // 2, num_layers=1, batch_first=True, bidirectional=True)

        # 3. Episodic Action History Encoder (`last 8 ops` + `last_changed`)
        self.hist_emb = nn.Embedding(len(PRIMITIVE_OPS) + 2, 40, padding_idx=len(PRIMITIVE_OPS))
        self.hist_gru = nn.GRU(40, d_hist, num_layers=1, batch_first=True)
        self.meta_proj = nn.Linear(2, 16)

        # 4. Joint Policy Trunk + Heads
        d_joint = d_vis + d_goal + d_hist + 16
        self.trunk = nn.Sequential(
            nn.Linear(d_joint, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Linear(192, 160),
            nn.SiLU(inplace=True),
        )
        self.op_head = nn.Linear(160, len(PRIMITIVE_OPS))

        # Character-level Open-Vocabulary Span Tagger (`[query_or_cmd_mask, serp_target_mask]`)
        self.span_head = nn.Sequential(
            nn.Linear(d_goal + 160, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 2),
        )

    def forward(
        self,
        img_128: torch.Tensor,
        goal_ids: torch.Tensor,
        hist_ids: torch.Tensor,
        meta_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.vis_proj(self.vis_cnn(img_128))

        ce = self.char_emb(goal_ids).transpose(1, 2)
        cc = self.char_conv(ce).transpose(1, 2)
        h_char, _ = self.goal_gru(cc)  # (B, L, d_goal)
        mask = (goal_ids != 0).float().unsqueeze(-1)
        g_pool = (h_char * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        he = self.hist_emb(hist_ids)
        _, h_last = self.hist_gru(he)
        m_vec = h_last.squeeze(0)
        meta_f = F.silu(self.meta_proj(meta_vec))

        joint = self.trunk(torch.cat([v, g_pool, m_vec, meta_f], dim=1))
        op_logits = self.op_head(joint)

        joint_exp = joint.unsqueeze(1).expand(-1, h_char.shape[1], -1)
        span_logits = self.span_head(torch.cat([h_char, joint_exp], dim=-1))  # (B, L, 2)
        return op_logits, span_logits

    @torch.no_grad()
    def act(
        self,
        rgb_uint8: np.ndarray,
        goal: str,
        history_ops: list[int],
        last_changed: int = 1,
    ) -> dict[str, Any]:
        self.eval()
        arr128 = np.asarray(Image.fromarray(rgb_uint8).resize((128, 128), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(arr128).permute(2, 0, 1).unsqueeze(0)

        g_ids = torch.tensor([encode_goal_chars(goal)], dtype=torch.long)
        pad_id = len(PRIMITIVE_OPS)
        padded_hist = ([pad_id] * max(0, HIST_LEN - len(history_ops)) + list(history_ops)[-HIST_LEN:])
        h_ids = torch.tensor([padded_hist], dtype=torch.long)
        meta_t = torch.tensor([[min(1.0, len(history_ops) / 10.0), float(last_changed)]], dtype=torch.float32)

        op_logits, span_logits = self.forward(img_t, g_ids, h_ids, meta_t)
        probs = F.softmax(op_logits[0], dim=0).cpu().numpy()
        op_id = int(np.argmax(probs))
        op_name = PRIMITIVE_OPS[op_id]

        span_probs = torch.sigmoid(span_logits[0]).cpu().numpy()  # (L, 2)
        query_text = decode_char_span(goal, span_probs[:, 0])
        target_text = decode_char_span(goal, span_probs[:, 1])

        sub_instr = primitive_to_instruction(op_name, target_text)
        return {
            "op_id": op_id,
            "op_name": op_name,
            "confidence": round(float(probs[op_id]), 4),
            "query_text": query_text,
            "target_text": target_text,
            "sub_instruction": sub_instr,
        }


def primitive_to_instruction(op_name: str, target_text: str) -> str | None:
    mapping = {
        "DONE": None,
        "DISMISS_ALERT": "click the cancel button",
        "OPEN_BROWSER_ICON": "select the browser icon",
        "FOCUS_SEARCH": "click the search input box",
        "TYPE_SEARCH_QUERY": None,
        "SUBMIT_SEARCH": "click the submit button",
        "CHOOSE_SERP_RESULT": f"choose {target_text or 'postgres 5432'} from the menu",
        "CLICK_BROWSER_BACK": "select the back icon",
        "COPY_ARTICLE_FACT": "click the copy button",
        "BOOKMARK_ARTICLE": "click the bookmark button",
        "OPEN_NOTES_TAB": "open the notes tab",
        "PASTE_INTO_NOTES": "click the paste button",
        "SAVE_NOTES": "click the save button",
        "OPEN_TERMINAL_ICON": "select the terminal icon",
        "FOCUS_COMMAND": "click the command input box",
        "TYPE_COMMAND": None,
        "RUN_COMMAND": "click the run button",
        "OPEN_FILES_ICON": "select the files icon",
        "DELETE_TEMP_FILES": "click the delete button",
        "EXPORT_FILES_INDEX": "click the export button",
    }
    return mapping[op_name]


def _check_goal_verdict(spec: dict[str, Any], ws: str, os_env: MockDesktopOS) -> bool:
    t = spec.get("type") or spec.get("check")
    if t == "article_fact":
        return check_desktop_success({"check": "article_fact", "substr": spec["expected"]}, ws, os_env)
    if t == "bookmark_fact":
        return check_desktop_success({"check": "bookmark_fact", "substr": spec["expected"]}, ws, os_env)
    if t == "file_substr":
        return check_desktop_success({"check": "file_substr", "rel": spec["path"], "substr": spec["substr"]}, ws, os_env)
    if t == "files_cleaned_and_exported":
        idx_p = os.path.join(ws, "files", "index.json")
        if not os.path.exists(idx_p):
            return False
        files_list = json.load(open(idx_p)).get("files", [])
        return not any(str(fn).endswith(".iso") for fn in files_list)
    return check_desktop_success(spec, ws, os_env)

@dataclass
class UnscriptedGoalTask:
    id: str
    family: str
    split: str  # 'seen_phrasing' | 'ood_phrasing'
    goal: str
    theme: str
    font: str
    open_windows: list[str]
    ad_trap: bool
    success: dict[str, Any]
    max_steps: int = 14
    max_seconds: float = 9.0


def build_unscripted_goal_tasks() -> list[UnscriptedGoalTask]:
    themes = ["dark", "light", "high_contrast", "nord_slate", "solarized_light"]
    fonts = ["DejaVu Sans", "Liberation Sans", "FreeSans"]
    fact_of = {d.slug: d.fact for d in KNOWLEDGE_DOCS}

    raw_specs: list[tuple[str, str, str, str, list[str], bool, dict[str, Any]]] = [
        # --- 20 Seen-Template Phrasings (diverse topics, themes, initial desktop states) ---
        ("ug-001", "google_lookup", "seen_phrasing", "search minigoogle for postgres 5432", ["browser"], False, {"type": "article_fact", "expected": fact_of["postgres 5432"]}),
        ("ug-002", "google_lookup", "seen_phrasing", "google git bisect in browser", [], False, {"type": "article_fact", "expected": fact_of["git bisect"]}),
        ("ug-003", "google_lookup", "seen_phrasing", "search minigoogle for redis 6379", ["browser"], True, {"type": "article_fact", "expected": fact_of["redis 6379"]}),
        ("ug-004", "google_lookup", "seen_phrasing", "google dijkstra path in browser", ["notes"], True, {"type": "article_fact", "expected": fact_of["dijkstra path"]}),
        ("ug-005", "google_bookmark", "seen_phrasing", "google ssh port 22 and bookmark the article", ["browser"], False, {"type": "bookmark_fact", "expected": fact_of["ssh port 22"]}),
        ("ug-006", "google_bookmark", "seen_phrasing", "search minigoogle for tar extract and bookmark page", [], False, {"type": "bookmark_fact", "expected": fact_of["tar extract"]}),
        ("ug-007", "google_bookmark", "seen_phrasing", "google sha256 bits and bookmark the article", ["browser"], True, {"type": "bookmark_fact", "expected": fact_of["sha256 bits"]}),
        ("ug-008", "google_to_notes", "seen_phrasing", "google dns port 53 copy the fact and save it in notes", ["browser", "notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["dns port 53"]}),
        ("ug-009", "google_to_notes", "seen_phrasing", "google chmod 755 copy the fact and save it in notes", ["browser", "notes"], True, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["chmod 755"]}),
        ("ug-010", "google_to_notes", "seen_phrasing", "search minigoogle for python gil copy fact and save in notes", ["browser", "notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["python gil"]}),
        ("ug-011", "google_to_notes", "seen_phrasing", "google tls alpn copy the fact and save it in notes", ["browser", "notes"], True, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["tls alpn"]}),
        ("ug-012", "google_to_notes", "seen_phrasing", "search minigoogle for quicksort pivot copy fact and save in notes", ["browser", "notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["quicksort pivot"]}),
        ("ug-013", "popup_then_google", "seen_phrasing", "dismiss the alert popup and google ntp port 123", ["browser", "alert"], True, {"type": "article_fact", "expected": fact_of["ntp port 123"]}),
        ("ug-014", "popup_then_google", "seen_phrasing", "dismiss the alert popup and google btree index", ["browser", "alert"], False, {"type": "article_fact", "expected": fact_of["btree index"]}),
        ("ug-015", "popup_then_google", "seen_phrasing", "dismiss the alert popup and google utf8 variable", ["browser", "alert"], True, {"type": "article_fact", "expected": fact_of["utf8 variable"]}),
        ("ug-016", "terminal_to_notes", "seen_phrasing", "open terminal run uname and save the output in notes", ["notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": "KERNEL:Linux-6.1-x86_64"}),
        ("ug-017", "terminal_to_notes", "seen_phrasing", "open terminal run df and save the output in notes", ["notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": "DISK_OK:/dev/root 42%"}),
        ("ug-018", "terminal_to_notes", "seen_phrasing", "open terminal run ports and save the output in notes", ["notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": "LISTEN:22,80,5432"}),
        ("ug-019", "files_cleanup_export", "seen_phrasing", "open files delete temp iso and export the file index", [], False, {"type": "files_cleaned_and_exported"}),
        ("ug-020", "files_cleanup_export", "seen_phrasing", "open files delete temp iso and export the file index", ["browser"], False, {"type": "files_cleaned_and_exported"}),

        # --- 10 Strictly Held-Out OOD Phrasings (unseen surface templates!) ---
        ("ug-021", "google_lookup", "ood_phrasing", "on the web look up ed25519 key right now", ["browser"], True, {"type": "article_fact", "expected": fact_of["ed25519 key"]}),
        ("ug-022", "google_lookup", "ood_phrasing", "use web search to inspect sqlite vacuum", [], False, {"type": "article_fact", "expected": fact_of["sqlite vacuum"]}),
        ("ug-023", "google_bookmark", "ood_phrasing", "please find functools lru online and bookmark it", ["browser"], True, {"type": "bookmark_fact", "expected": fact_of["functools lru"]}),
        ("ug-024", "google_bookmark", "ood_phrasing", "browse for asyncio gather and add a bookmark", ["notes"], False, {"type": "bookmark_fact", "expected": fact_of["asyncio gather"]}),
        ("ug-025", "google_to_notes", "ood_phrasing", "research smtp 587 online then paste and store in notes", ["browser", "notes"], True, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["smtp 587"]}),
        ("ug-026", "google_to_notes", "ood_phrasing", "find ieee754 float on browser and record fact into notes", ["browser", "notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["ieee754 float"]}),
        ("ug-027", "google_to_notes", "ood_phrasing", "check collections deque on web and write result to notes", ["browser", "notes"], True, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": fact_of["collections deque"]}),
        ("ug-028", "popup_then_google", "ood_phrasing", "clear warning dialog then look up kill sigterm online", ["browser", "alert"], True, {"type": "article_fact", "expected": fact_of["kill sigterm"]}),
        ("ug-029", "terminal_to_notes", "ood_phrasing", "launch terminal execute whoami then store output into notes", ["notes"], False, {"type": "file_substr", "path": "notes/saved_note.txt", "substr": "USER:agent03"}),
        ("ug-030", "files_cleanup_export", "ood_phrasing", "in file manager purge iso files and dump index", ["notes"], False, {"type": "files_cleaned_and_exported"}),
    ]

    out: list[UnscriptedGoalTask] = []
    for idx, (tid, fam, split, goal, wins, ad_trap, succ) in enumerate(raw_specs):
        out.append(
            UnscriptedGoalTask(
                id=tid,
                family=fam,
                split=split,
                goal=goal,
                theme=themes[idx % len(themes)],
                font=fonts[idx % len(fonts)],
                open_windows=wins,
                ad_trap=ad_trap,
                success=succ,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Authentic X11 State Collection + Supervised Imitation Training
# ---------------------------------------------------------------------------

def _sample_goal_and_spec(rng: random.Random) -> dict[str, Any]:
    """Sample combinatorial goal surface phrasings (strictly disjoint from the 10 OOD phrasings!)."""
    fam = rng.choices(
        ["google_lookup", "google_bookmark", "google_to_notes", "popup_then_google", "terminal_to_notes", "files_cleanup_export"],
        weights=[0.20, 0.16, 0.28, 0.14, 0.12, 0.10],
        k=1,
    )[0]
    doc = rng.choice(KNOWLEDGE_DOCS)
    slug = doc.slug

    search_verbs = ["google", "search minigoogle for", "search for", "query", "find", "look for", "search web for", "look up", "check"]
    loc_sfx = ["", " in browser", " on web", " online", " on browser", " today"]

    if fam == "google_lookup":
        v = rng.choice(search_verbs)
        s = rng.choice(loc_sfx)
        lead = rng.choice(["", "please ", "open browser and "])
        tpl = f"{lead}{v} {slug}{s}".strip()
        return {"family": fam, "goal": tpl, "query": slug, "target": slug}

    if fam == "google_bookmark":
        v = rng.choice(search_verbs)
        s = rng.choice(["", " online", " in browser"])
        bm_tail = rng.choice(["and bookmark the article", "and bookmark page", "and save a bookmark", "then bookmark it", "and add bookmark"])
        tpl = f"{rng.choice(['', 'please '])}{v} {slug}{s} {bm_tail}".strip()
        return {"family": fam, "goal": tpl, "query": slug, "target": slug}

    if fam == "google_to_notes":
        v = rng.choice(search_verbs + ["research"])
        s = rng.choice(["", " in browser", " on web", " online"])
        nt_tail = rng.choice([
            "copy the fact and save it in notes",
            "copy fact and save in notes",
            "copy fact then paste into notes",
            "and save the copied fact in notes",
            "then copy and store in notes",
            "and write fact into notes",
            "and record result in notes",
        ])
        tpl = f"{v} {slug}{s} {nt_tail}".strip()
        return {"family": fam, "goal": tpl, "query": slug, "target": slug}

    if fam == "popup_then_google":
        pop_lead = rng.choice([
            "dismiss the alert popup",
            "close the alert dialog",
            "cancel popup window",
            "clear alert dialog",
            "dismiss warning popup",
        ])
        conn = rng.choice([" and ", " then "])
        v = rng.choice(["google", "search for", "look up", "find"])
        s = rng.choice(["", " in browser", " online"])
        tpl = f"{pop_lead}{conn}{v} {slug}{s}"
        return {"family": fam, "goal": tpl, "query": slug, "target": slug}

    if fam == "terminal_to_notes":
        cmd = rng.choice(["uname", "df", "whoami", "ports"])
        t_lead = rng.choice(["open terminal", "in terminal", "start terminal"])
        t_verb = rng.choice(["run", "type", "exec", "call"])
        t_tail = rng.choice([
            "and save the output in notes",
            "then paste and save in notes",
            "and store output in notes",
            "then write output into notes",
        ])
        tpl = f"{t_lead} {t_verb} {cmd} {t_tail}"
        return {"family": fam, "goal": tpl, "query": cmd, "target": cmd}

    f_lead = rng.choice(["open files", "open file manager", "in files", "in file manager"])
    f_act = rng.choice(["delete temp iso", "delete iso", "clean temp iso", "remove iso files"])
    f_tail = rng.choice(["and export the file index", "and export the index", "and export index", "then save index"])
    return {"family": "files_cleanup_export", "goal": f"{f_lead} {f_act} {f_tail}", "query": "", "target": ""}


def _oracle_next_op_and_advance(os_env: MockDesktopOS, spec: dict[str, Any], history: list[int]) -> int:
    """Compute exact state-dependent primitive from `os_env` AND advance `os_env` to next state."""
    fam = spec["family"]
    top = os_env.top_window()
    top_id = top.win_id if top is not None else None

    # 1. Always dismiss modal alert first if present on top!
    if top_id == "alert":
        os_env.close_window("alert")
        return OP_TO_ID["DISMISS_ALERT"]

    if fam in ("google_lookup", "google_bookmark", "google_to_notes", "popup_then_google"):
        want_notes = (fam == "google_to_notes")
        want_bm = (fam == "google_bookmark")

        # Check terminal conditions first!
        if not want_notes and not want_bm:
            if os_env.browser_mode == "article" and os_env.current_article and os_env.current_article.slug == spec["target"]:
                return OP_TO_ID["DONE"]
        if want_bm and len(os_env.bookmarks) > 0:
            return OP_TO_ID["DONE"]
        if want_notes and os_env.notes_saved_count > 0:
            return OP_TO_ID["DONE"]

        # If clipboard already holds researched fact and we want notes:
        if want_notes and os_env.clipboard:
            if top_id != "notes":
                os_env.open_window("notes")
                return OP_TO_ID["OPEN_NOTES_TAB"]
            if not os_env.notes_text:
                os_env.paste_into_notes()
                return OP_TO_ID["PASTE_INTO_NOTES"]
            os_env.save_notes()
            return OP_TO_ID["SAVE_NOTES"]

        # Otherwise we are in the browser research phase!
        if top_id != "browser":
            os_env.open_window("browser")
            return OP_TO_ID["OPEN_BROWSER_ICON"]

        if os_env.browser_mode == "article" and os_env.current_article:
            if os_env.current_article.slug != spec["target"]:
                os_env.browser_mode = "serp" if os_env.serp_results else "home"
                return OP_TO_ID["CLICK_BROWSER_BACK"]
            if want_bm:
                os_env.bookmark_current()
                return OP_TO_ID["BOOKMARK_ARTICLE"]
            os_env.copy_article_fact()
            return OP_TO_ID["COPY_ARTICLE_FACT"]

        if os_env.browser_mode == "serp" and os_env.serp_results:
            # Select target doc from SERP
            match_docs = [d for d in os_env.serp_results if d.slug == spec["target"]]
            chosen = match_docs[0] if match_docs else os_env.serp_results[0]
            os_env.open_serp_doc(chosen)
            return OP_TO_ID["CHOOSE_SERP_RESULT"]

        if os_env.browser_query.strip().lower() == spec["query"]:
            os_env.run_browser_search()
            return OP_TO_ID["SUBMIT_SEARCH"]

        if os_env.focused_input != "browser_search":
            os_env.focused_input = "browser_search"
            os_env.active_cid = "br_search_fld"
            return OP_TO_ID["FOCUS_SEARCH"]

        os_env.browser_query = spec["query"]
        return OP_TO_ID["TYPE_SEARCH_QUERY"]

    if fam == "terminal_to_notes":
        if os_env.notes_saved_count > 0:
            return OP_TO_ID["DONE"]
        if os_env.clipboard:
            if top_id != "notes":
                os_env.open_window("notes")
                return OP_TO_ID["OPEN_NOTES_TAB"]
            if not os_env.notes_text:
                os_env.paste_into_notes()
                return OP_TO_ID["PASTE_INTO_NOTES"]
            os_env.save_notes()
            return OP_TO_ID["SAVE_NOTES"]

        if top_id != "terminal":
            os_env.open_window("terminal")
            return OP_TO_ID["OPEN_TERMINAL_ICON"]
        if os_env.terminal_cmd.strip().lower() == spec["query"]:
            os_env.exec_terminal()
            return OP_TO_ID["RUN_COMMAND"]
        if os_env.focused_input != "terminal_cmd":
            os_env.focused_input = "terminal_cmd"
            os_env.active_cid = "tm_cmd"
            return OP_TO_ID["FOCUS_COMMAND"]
        os_env.terminal_cmd = spec["query"]
        return OP_TO_ID["TYPE_COMMAND"]

    # files_cleanup_export
    idx_file = os.path.join(os_env.ws_dir, "files", "index.json")
    if os.path.exists(idx_file):
        return OP_TO_ID["DONE"]
    if top_id != "files":
        os_env.open_window("files")
        return OP_TO_ID["OPEN_FILES_ICON"]
    if any(fn.endswith(".iso") for fn in os_env.files_items):
        os_env.clean_iso_files()
        return OP_TO_ID["DELETE_TEMP_FILES"]
    with open(idx_file, "w") as f:
        json.dump({"files": os_env.files_items}, f)
    return OP_TO_ID["EXPORT_FILES_INDEX"]


def collect_x11_policy_dataset(n_episodes: int = 440, seed: int = 202610) -> dict[str, np.ndarray]:
    """Collect authentic `Xvfb` + `tkinter` + `Capture` X11 pixel trajectories in ~15 seconds."""
    _ensure_xvfb_display()
    rng = random.Random(seed)
    themes = list(PALETTES.keys())
    fonts = ["DejaVu Sans", "Liberation Sans", "FreeSans"]

    imgs_128: list[np.ndarray] = []
    goals_arr: list[list[int]] = []
    hists_arr: list[list[int]] = []
    metas_arr: list[list[float]] = []
    ops_arr: list[int] = []
    spans_arr: list[np.ndarray] = []

    pad_id = len(PRIMITIVE_OPS)
    t0 = time.time()

    root = tk.Tk()
    try:
        root.overrideredirect(True)
        root.geometry("256x256+0+0")
        root.update()
        with Capture() as cap:
            with tempfile.TemporaryDirectory(prefix="nb_pol_gen_") as tmp_base:
                for ep in range(n_episodes):
                    ws = os.path.join(tmp_base, f"ep_{ep % 16}")
                    if os.path.exists(ws):
                        import shutil
                        shutil.rmtree(ws)
                    spec = _sample_goal_and_spec(rng)
                    theme = rng.choice(themes)
                    font = rng.choice(fonts)
                    ad_trap = rng.random() < 0.5

                    os_env = MockDesktopOS(root, ws, theme=theme, font_family=font, search_ad_trap=ad_trap)
                    # Randomize starting windows!
                    if spec["family"] == "popup_then_google":
                        os_env.open_window("browser")
                        os_env.open_window("alert")
                    else:
                        init_choices = [[], ["browser"], ["notes"], ["browser", "notes"], ["notes", "browser"]]
                        for wid in rng.choice(init_choices):
                            os_env.open_window(wid)

                    g_ids = encode_goal_chars(spec["goal"])
                    q_mask = build_span_char_mask(spec["goal"], spec["query"])
                    t_mask = build_span_char_mask(spec["goal"], spec["target"])
                    span_2 = np.stack([q_mask, t_mask], axis=-1)

                    hist: list[int] = []
                    did_perturb = False
                    for step_i in range(13):
                        os_env.redraw()
                        root.update_idletasks()
                        root.update()
                        rgb256 = np.asarray(cap.full_screen().image, dtype=np.uint8)
                        rgb128 = np.asarray(Image.fromarray(rgb256).resize((128, 128), Image.BILINEAR), dtype=np.uint8)

                        padded_h = ([pad_id] * max(0, HIST_LEN - len(hist)) + hist[-HIST_LEN:])
                        last_chg = 1.0 if (step_i == 0 or rng.random() < 0.90) else 0.0
                        op_id = _oracle_next_op_and_advance(os_env, spec, hist)

                        # 32% of SERP transitions: simulate accidentally opening an Ad / wrong SERP link
                        # so the policy learns visual self-recovery (`CLICK_BROWSER_BACK`) from pixels!
                        if op_id == OP_TO_ID["CHOOSE_SERP_RESULT"] and not did_perturb and rng.random() < 0.32:
                            wrongs = [d for d in os_env.serp_results if d.slug != spec["target"]]
                            if wrongs:
                                did_perturb = True
                                os_env.open_serp_doc(wrongs[0])

                        imgs_128.append(rgb128)
                        goals_arr.append(g_ids)
                        hists_arr.append(padded_h)
                        metas_arr.append([min(1.0, len(hist) / 10.0), last_chg])
                        ops_arr.append(op_id)
                        spans_arr.append(span_2)

                        hist.append(op_id)
                        if op_id == OP_TO_ID["DONE"]:
                            break
    finally:
        root.destroy()

    print(
        f"[policy_data] collected {len(ops_arr)} authentic X11 frames across {n_episodes} episodes "
        f"in {time.time()-t0:.1f}s rss={peak_rss_mb():.0f}MB",
        flush=True,
    )
    return {
        "imgs": np.stack(imgs_128, axis=0),
        "goals": np.asarray(goals_arr, dtype=np.int64),
        "hists": np.asarray(hists_arr, dtype=np.int64),
        "metas": np.asarray(metas_arr, dtype=np.float32),
        "ops": np.asarray(ops_arr, dtype=np.int64),
        "spans": np.stack(spans_arr, axis=0),
    }


def train_desktop_brain_policy(
    steps: int = 1100,
    batch_size: int = 32,
    lr: float = 2.0e-3,
    n_episodes: int = 440,
    seed: int = 202610,
    out_dir: str = "checkpoints/desktop_brain_v1",
) -> DesktopBrainPolicy:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = collect_x11_policy_dataset(n_episodes=n_episodes, seed=seed)
    N = len(data["ops"])
    perm = rng.permutation(N)
    n_val = max(120, int(0.10 * N))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = DesktopBrainPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        b_idx = rng.choice(tr_idx, size=batch_size, replace=True)
        imgs = torch.from_numpy(data["imgs"][b_idx].astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2)
        goals = torch.from_numpy(data["goals"][b_idx])
        hists = torch.from_numpy(data["hists"][b_idx])
        metas = torch.from_numpy(data["metas"][b_idx])
        ops = torch.from_numpy(data["ops"][b_idx])
        spans = torch.from_numpy(data["spans"][b_idx])

        op_logits, span_logits = model(imgs, goals, hists, metas)
        loss_op = F.cross_entropy(op_logits, ops)
        # Weight positive span chars 4x so narrow 1-2 word spans never vanish!
        pos_w = torch.tensor([4.0, 4.0])
        loss_span = F.binary_cross_entropy_with_logits(span_logits, spans, pos_weight=pos_w)
        loss = loss_op + 2.5 * loss_span

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        cur_lr = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vi = torch.from_numpy(data["imgs"][val_idx].astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2)
                vg = torch.from_numpy(data["goals"][val_idx])
                vh = torch.from_numpy(data["hists"][val_idx])
                vm = torch.from_numpy(data["metas"][val_idx])
                vo = torch.from_numpy(data["ops"][val_idx])
                v_op, v_sp = model(vi, vg, vh, vm)
                op_acc = float((v_op.argmax(dim=1) == vo).float().mean().item())
                sp_pred = (torch.sigmoid(v_sp) > 0.45).float()
                sp_tgt = torch.from_numpy(data["spans"][val_idx])
                inter = (sp_pred * sp_tgt).sum()
                f1 = float((2.0 * inter / (sp_pred.sum() + sp_tgt.sum() + 1e-6)).item())
            print(
                f"[policy_train] step {step}/{steps} loss={loss.item():.4f} val_op_acc={op_acc:.4f} "
                f"val_span_f1={f1:.4f} rss={peak_rss_mb():.0f}MB elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"policy": model.state_dict(), "steps": steps, "n_params": sum(p.numel() for p in model.parameters())}, os.path.join(out_dir, "weights.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "kind": "desktop_brain_policy",
                "version": 1,
                "steps": steps,
                "n_params": sum(p.numel() for p in model.parameters()),
                "primitives": list(PRIMITIVE_OPS),
                "max_goal_chars": MAX_GOAL_CHARS,
            },
            f,
            indent=2,
        )
    return model


def load_desktop_brain_policy(ckpt_dir: str = "checkpoints/desktop_brain_v1") -> DesktopBrainPolicy:
    model = DesktopBrainPolicy()
    raw = torch.load(os.path.join(ckpt_dir, "weights.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(raw["policy"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Live Headless X11 Closed-Loop Evaluation (`DesktopBrainPolicy` + `EyesGrounder`)
# ---------------------------------------------------------------------------

def _save_policy_strip(task_id: str, goal: str, trace: list[dict[str, Any]], out_path: str) -> None:
    if not trace:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    panels = trace[:10]
    pw, ph = 256, 292
    canvas = Image.new("RGB", (pw * len(panels), ph), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, st in enumerate(panels):
        x0 = idx * pw
        canvas.paste(Image.fromarray(st["rgb"]), (x0, 0))
        if st.get("x") is not None and st.get("y") is not None:
            cx, cy = x0 + int(st["x"]), int(st["y"])
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 70, 70), width=2)
        draw.text((x0 + 4, 259), f"#{idx+1} {st['op_name'][:24]}", fill=(235, 240, 250))
        draw.text((x0 + 4, 274), f"{st.get('detail', '')[:34]}", fill=(180, 210, 255))
    canvas.save(out_path)


def run_unscripted_x11_suite(
    policy: DesktopBrainPolicy,
    grounder_ckpt: str = "checkpoints/grip_v3",
    out_json: str | None = None,
    strips_dir: str | None = None,
    gallery_path: str | None = None,
) -> dict[str, Any]:
    """Run all 30 UNSCRIPTED natural-language Desktop & Googling tasks on live X11."""
    _ensure_xvfb_display()
    tasks = build_unscripted_goal_tasks()
    grounder = load_exported_grounder(grounder_ckpt)

    records: list[dict[str, Any]] = []
    fail_strips: list[str] = []

    root = tk.Tk()
    try:
        root.overrideredirect(True)
        root.geometry("256x256+0+0")
        root.update()

        with Capture() as cap:
            hands = DesktopHands()
            for task in tasks:
                t0 = time.perf_counter()
                history_ops: list[int] = []
                last_changed = 1
                exclude_boxes: list[tuple[int, int, int, int]] = []
                serp_exclude_boxes: list[tuple[int, int, int, int]] = []
                last_serp_xy: tuple[int, int] | None = None
                trace: list[dict[str, Any]] = []
                steps_used = 0

                with tempfile.TemporaryDirectory(prefix=f"nb_u_{task.id}_") as ws:
                    os_env = MockDesktopOS(
                        root,
                        ws,
                        theme=task.theme,
                        font_family=task.font,
                        search_ad_trap=task.ad_trap,
                    )
                    for wid in task.open_windows:
                        os_env.open_window(wid)
                    os_env.redraw()
                    os_env.pump()

                    while steps_used < task.max_steps and (time.perf_counter() - t0) < task.max_seconds:
                        rgb_before = np.asarray(cap.full_screen().image, dtype=np.uint8)
                        dec = policy.act(rgb_before, task.goal, history_ops, last_changed=last_changed)
                        op_name = dec["op_name"]
                        op_id = dec["op_id"]

                        if op_name == "DONE":
                            trace.append({"op_name": "DONE", "detail": "policy declared DONE", "x": None, "y": None, "rgb": rgb_before})
                            break

                        steps_used += 1
                        if op_name in ("TYPE_SEARCH_QUERY", "TYPE_COMMAND"):
                            text_to_type = dec["query_text"]
                            hands.perform({"type": "type", "text": text_to_type})
                            os_env.pump(cycles=6, delay_s=0.004)
                            rgb_after = np.asarray(cap.full_screen().image, dtype=np.uint8)
                            changed = int((np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)).max(axis=-1) > 4).sum()) >= 3
                            last_changed = 1 if changed else 0
                            if changed:
                                history_ops.append(op_id)
                                exclude_boxes.clear()
                            trace.append({"op_name": op_name, "detail": f"typed '{text_to_type}'", "x": None, "y": None, "rgb": rgb_before})
                        else:
                            sub_instr = dec["sub_instruction"] or "click the submit button"
                            active_ex = (exclude_boxes + serp_exclude_boxes) if op_name == "CHOOSE_SERP_RESULT" else exclude_boxes
                            pred = grounder.ground(rgb_before, sub_instr, exclude_box=active_ex or None)
                            px, py = int(pred["x"]), int(pred["y"])
                            if op_name == "CHOOSE_SERP_RESULT":
                                last_serp_xy = (px, py)
                            elif op_name == "CLICK_BROWSER_BACK" and last_serp_xy is not None:
                                sx, sy = last_serp_xy
                                serp_exclude_boxes.append((max(0, sx - 25), max(0, sy - 13), 50, 26))
                                last_serp_xy = None
                            hands.perform({"type": "click", "x": px, "y": py})
                            os_env.pump(cycles=6, delay_s=0.004)
                            rgb_after = np.asarray(cap.full_screen().image, dtype=np.uint8)
                            changed = int((np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)).max(axis=-1) > 4).sum()) >= 3
                            last_changed = 1 if changed else 0
                            if changed:
                                history_ops.append(op_id)
                                exclude_boxes.clear()
                            else:
                                exclude_boxes.append((max(0, px - 22), max(0, py - 11), 44, 22))
                            trace.append({"op_name": op_name, "detail": sub_instr, "x": px, "y": py, "rgb": rgb_before})

                    passed = _check_goal_verdict(task.success, ws, os_env)
                    elapsed = time.perf_counter() - t0
                    rec = {
                        "id": task.id,
                        "family": task.family,
                        "split": task.split,
                        "goal": task.goal,
                        "passed": bool(passed),
                        "steps": steps_used,
                        "seconds": round(elapsed, 4),
                        "ops_executed": [PRIMITIVE_OPS[i] for i in history_ops],
                    }
                    records.append(rec)

                    if not passed and strips_dir:
                        sp = os.path.join(strips_dir, f"{task.id}.png")
                        _save_policy_strip(task.id, task.goal, trace, sp)
                        fail_strips.append(sp)
    finally:
        root.destroy()

    if gallery_path and fail_strips:
        os.makedirs(os.path.dirname(gallery_path) or ".", exist_ok=True)
        imgs = [Image.open(p) for p in fail_strips[:8]]
        max_w = max(im.width for im in imgs)
        tot_h = sum(im.height + 6 for im in imgs)
        comp = Image.new("RGB", (max_w, tot_h), (12, 14, 18))
        cy = 0
        for im in imgs:
            comp.paste(im, (0, cy))
            cy += im.height + 6
        comp.save(gallery_path)

    n_tot = len(records)
    n_pass = sum(1 for r in records if r["passed"])
    seen_recs = [r for r in records if r["split"] == "seen_phrasing"]
    ood_recs = [r for r in records if r["split"] == "ood_phrasing"]

    by_fam: dict[str, dict[str, Any]] = {}
    for fam in sorted({r["family"] for r in records}):
        sub = [r for r in records if r["family"] == fam]
        by_fam[fam] = {
            "n": len(sub),
            "passed": sum(1 for r in sub if r["passed"]),
            "success_rate": round(sum(1 for r in sub if r["passed"]) / len(sub), 4),
            "median_steps": round(float(np.median([r["steps"] for r in sub])), 2),
        }

    summary = {
        "suite": "m10_unscripted_autonomous_desktop_and_googling_x11",
        "n_tasks": n_tot,
        "passed": n_pass,
        "success_rate": round(n_pass / max(1, n_tot), 4),
        "seen_phrasing": {
            "n": len(seen_recs),
            "passed": sum(1 for r in seen_recs if r["passed"]),
            "success_rate": round(sum(1 for r in seen_recs if r["passed"]) / max(1, len(seen_recs)), 4),
        },
        "ood_phrasing": {
            "n": len(ood_recs),
            "passed": sum(1 for r in ood_recs if r["passed"]),
            "success_rate": round(sum(1 for r in ood_recs if r["passed"]) / max(1, len(ood_recs)), 4),
        },
        "median_steps": round(float(np.median([r["steps"] for r in records])), 2),
        "median_seconds": round(float(np.median([r["seconds"] for r in records])), 3),
        "ram_peak_mb": round(peak_rss_mb(), 1),
        "by_family": by_fam,
        "tasks": records,
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
    print(
        f"[m10_unscripted_x11] passed={n_pass}/{n_tot} ({100.0*summary['success_rate']:.2f}%) | "
        f"seen={summary['seen_phrasing']['passed']}/{summary['seen_phrasing']['n']} ({100.0*summary['seen_phrasing']['success_rate']:.1f}%) | "
        f"ood={summary['ood_phrasing']['passed']}/{summary['ood_phrasing']['n']} ({100.0*summary['ood_phrasing']['success_rate']:.1f}%) | "
        f"med_steps={summary['median_steps']} | med_sec={summary['median_seconds']}s | ram={summary['ram_peak_mb']}MB",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--episodes", type=int, default=480)
    ap.add_argument("--policy-dir", default="checkpoints/desktop_brain_v1")
    ap.add_argument("--grounder-dir", default="checkpoints/grip_v3")
    ap.add_argument("--baseline-out", default="/home/user/fleet/agents/03-eyes/m10_unscripted_random_baseline.json")
    ap.add_argument("--out", default="/home/user/fleet/agents/03-eyes/m10_unscripted_autonomous_x11.json")
    ap.add_argument("--strips-dir", default="/home/user/fleet/agents/03-eyes/m10_failure_strips")
    ap.add_argument("--gallery", default="/home/user/fleet/agents/03-eyes/galleries/m10_unscripted_failures.png")
    args = ap.parse_args()

    # 1. Measure honest zero-training random-init policy baseline on live X11!
    torch.manual_seed(202610)
    rand_pol = DesktopBrainPolicy()
    print("=== 1. Zero-Training Random-Init Policy Baseline (30 Unscripted X11 Tasks) ===", flush=True)
    run_unscripted_x11_suite(rand_pol, grounder_ckpt=args.grounder_dir, out_json=args.baseline_out)

    # 2. Train DesktopBrainPolicy from scratch on authentic X11 captures!
    print("=== 2. Training DesktopBrainPolicy from scratch ===", flush=True)
    trained_pol = train_desktop_brain_policy(
        steps=args.steps,
        n_episodes=args.episodes,
        out_dir=args.policy_dir,
    )

    # 3. Evaluate trained DesktopBrainPolicy + grip_v3 on all 30 unscripted live X11 tasks!
    print("=== 3. Trained DesktopBrainPolicy + grip_v3 (30 Unscripted Live X11 Tasks) ===", flush=True)
    run_unscripted_x11_suite(
        trained_pol,
        grounder_ckpt=args.grounder_dir,
        out_json=args.out,
        strips_dir=args.strips_dir,
        gallery_path=args.gallery,
    )


if __name__ == "__main__":
    main()
