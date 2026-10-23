"""A mock desktop where the job is: open Chrome, research something, report it
in Google Docs.

Why a mock: the product has to run unattended on a user's machine, and the
first honest milestone is a task the whole loop can be measured on, end to end,
with a verifier that cannot be fooled. The windows are drawn, the clicks are
coordinates, and every state transition is checkable -- which is exactly what
the real X11 backend needs to satisfy too, only with pixels instead of a
dataclass.

The task family:

    desktop -> chrome -> results -> article -> copy -> docs -> paste -> save

Every step is a click on a labelled element from the scan, and the only thing
that changes between instances is the topic, the answer text and the distractor
labels. The answer is never invented by the model: it is on the article page and
has to be carried into the document.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .ui_world import ACTION_OF_ROLE, SCALE, Element, make_layout

CANVAS = 1920  # the mock desktop is drawn in the protocol's own coordinate space

TOPICS = [
    "the capital of France", "the capital of Japan", "the capital of Canada",
    "the capital of Brazil", "the capital of Italy", "the capital of Egypt",
    "the symbol for iron", "the symbol for gold", "the symbol for oxygen",
    "the currency of Japan", "the currency of the United Kingdom",
    "the largest planet", "the boiling point of water", "the speed of light",
]

ANSWERS = {
    "the capital of France": "Paris",
    "the capital of Japan": "Tokyo",
    "the capital of Canada": "Ottawa",
    "the capital of Brazil": "Brasilia",
    "the capital of Italy": "Rome",
    "the capital of Egypt": "Cairo",
    "the symbol for iron": "Fe",
    "the symbol for gold": "Au",
    "the symbol for oxygen": "O",
    "the currency of Japan": "the yen",
    "the currency of the United Kingdom": "pound sterling",
    "the largest planet": "Jupiter",
    "the boiling point of water": "100 degrees Celsius",
    "the speed of light": "about 300,000 kilometres per second",
}

DESKTOP_ICONS = ["chrome", "docs", "files", "mail"]


@dataclass
class Win:
    """A drawn window: title bar plus a list of elements in canvas coords."""
    title: str
    x: int
    y: int
    w: int
    h: int
    elements: list[Element]


def _el(role: str, label: str, x: int, y: int, w: int, h: int) -> Element:
    return Element(role=role, label=label, x=x, y=y, w=w, h=h)


def _grid(items: list[str], role: str, x0: int, y0: int, cols: int,
          cw: int, ch: int, gap: int = 28) -> list[Element]:
    out = []
    for i, label in enumerate(items):
        r, c = divmod(i, cols)
        out.append(_el(role, label, x0 + c * (cw + gap), y0 + r * (ch + gap), cw, ch))
    return out


class WebTask:
    """One instance: a topic, its answer, and the distractor labels to beat."""

    def __init__(self, rng: random.Random):
        self.topic = rng.choice(TOPICS)
        self.answer = ANSWERS[self.topic]
        others = [t for t in TOPICS if t != self.topic]
        rng.shuffle(others)
        self.results = [self.topic] + others[:3]
        rng.shuffle(self.results)
        self.query = f"what is {self.topic}"
        self.distractors = rng.sample(
            [a for a in ANSWERS.values() if a != self.answer], k=3)

    def objective(self) -> str:
        # short on purpose: the objective and the screen scan share one
        # 256-token context, and the scan is what tells the model where to click
        return f"research {self.topic} in chrome and report it in docs"


class WebWorld:
    """The state machine the runtime executes. Verification is state, not prose."""

    name = "web"

    def __init__(self, task: WebTask):
        self.task = task
        self.state = "desktop"      # desktop | chrome | article | docs | saved
        self.clipboard: str | None = None
        self.doc_text: str = ""
        self.saved = False
        self.failed_clicks: list[str] = []

    # ------------------------------------------------------------------ views
    def _dock(self) -> list[Element]:
        """The dock is on every screen, so switching apps is always a real click."""
        return _grid(DESKTOP_ICONS, "icon", 120, 1700, 4, 220, 120, 30)

    def _desktop(self) -> Win:
        # no app icons on the wallpaper: a label that appears twice on one scan
        # is ambiguous, and an ambiguous scan teaches the policy to guess
        return Win("desktop", 0, 0, CANVAS, CANVAS, self._dock())

    def _chrome(self) -> Win:
        els = [_el("field", "address", 200, 90, 900, 90),
               _el("button", "search", 1130, 90, 220, 90)]
        els += _grid(self.task.results, "button", 200, 320, 1, 1500, 130, 30)
        els += _grid(self.task.distractors[:2], "icon", 200, 1200, 2, 420, 160, 40)
        return Win("chrome", 40, 40, 1840, 1600, els + self._dock())

    def _article(self) -> Win:
        els = [_el("button", "copy", 1450, 900, 260, 110)]
        els += _grid([self.task.answer] + self.task.distractors[:2], "field", 200, 300, 1, 1500, 110, 24)
        return Win("article", 40, 40, 1840, 1600, els + self._dock())

    def _docs(self) -> Win:
        els = [_el("button", "paste", 200, 820, 240, 110),
               _el("button", "save", 1650, 820, 240, 110)]
        # The editor's contents are part of the screen. Without them the scan
        # after a paste is identical to the scan before it, and the policy has
        # no way to know it should move on to save (it looped on paste for the
        # rest of the budget).
        body = self.doc_text or "empty document"
        status = "saved" if self.saved else "unsaved"
        els += [_el("field", body, 200, 260, 1500, 480),
                _el("label", status, 1650, 960, 200, 90)]
        return Win("docs", 120, 80, 1700, 1600, els + self._dock())

    def view(self) -> Win:
        if self.state == "desktop":
            return self._desktop()
        if self.state == "chrome":
            return self._chrome()
        if self.state == "article":
            return self._article()
        return self._docs()

    # -------------------------------------------------------------- protocol
    def observe(self) -> dict:
        win = self.view()
        return {
            "kind": "web", "state": self.state, "window": win.title,
            "elements": [{"label": e.label, "role": e.role, "cx": e.cx, "cy": e.cy,
                          "x": e.x, "y": e.y, "w": e.w, "h": e.h}
                         for e in win.elements],
            "clipboard": self.clipboard, "document": self.doc_text,
            # this world already speaks in protocol coordinates (0..1920)
            "scale": 1,
        }

    def _hit(self, point) -> Element | None:
        if point is None:
            return None
        px, py = point
        for e in self.view().elements:
            if e.contains(px, py):
                return e
        return None

    def act(self, action: dict, point) -> dict:
        if (action or {}).get("type") != "click":
            return {"ok": False, "detail": f"web world only clicks, got {action!r}"}
        el = self._hit(point)
        if el is None:
            return {"ok": False, "detail": "clicked empty desktop"}
        label, state = el.label, self.state

        # the dock is always live: chrome and docs are one click away from any app
        if label == "chrome" and self.state != "chrome":
            self.state = "chrome"
            return {"ok": True, "detail": "chrome opened, search field ready"}
        if label == "docs" and self.state != "docs":
            self.state = "docs"
            return {"ok": True, "detail": "google docs opened"}

        if state == "docs" and label == "document":
            return {"ok": True, "detail": f"clicked inside the document ({len(self.doc_text)} chars)"}

        if state == "desktop":
            if label == "chrome":
                self.state = "chrome"
                return {"ok": True, "detail": "chrome opened, search field ready"}
            if label == "docs":
                self.state = "docs"
                return {"ok": True, "detail": "google docs opened"}
            return {"ok": True, "detail": f"{label} opened, not what the task needs"}

        if state == "chrome":
            if label == self.task.topic:
                self.state = "article"
                return {"ok": True, "detail": f"opened the article on {label}"}
            if label in self.task.results or label in self.task.distractors:
                return {"ok": True, "detail": f"opened {label}, wrong result"}
            if label == "files":
                return {"ok": True, "detail": "files opened, wrong app"}
            return {"ok": True, "detail": f"clicked {label}, nothing opened"}

        if state == "article":
            if label == "copy":
                self.clipboard = self.task.answer
                return {"ok": True, "detail": f"copied {self.task.answer} to the clipboard"}
            if label == "close":
                self.state = "chrome"
                return {"ok": True, "detail": "article closed"}
            if label == "back":
                self.state = "chrome"
                return {"ok": True, "detail": "went back to the results"}
            return {"ok": True, "detail": f"clicked the text {label}"}

        if state == "docs":
            if label == "paste":
                if self.clipboard:
                    self.doc_text = self.clipboard
                    return {"ok": True, "detail": f"pasted {self.clipboard} into the document"}
                return {"ok": True, "detail": "clipboard is empty"}
            if label == "save":
                if self.doc_text:
                    self.saved = True
                    self.state = "saved"
                    return {"ok": True, "detail": f"saved the document: {self.doc_text}"}
                return {"ok": True, "detail": "document is empty, nothing saved"}
            if label == "close":
                self.state = "desktop"
                return {"ok": True, "detail": "docs closed"}
            return {"ok": True, "detail": f"clicked {label}"}

        return {"ok": True, "detail": f"clicked {label} after the task was saved"}

    def verify(self, before, after, step, act_result) -> tuple[str, str]:
        detail = act_result.get("detail", "")
        if self.saved and self.doc_text == self.task.answer:
            return "progress", detail
        if "wrong" in detail or "not what" in detail or "empty" in detail:
            return "wrong", detail
        return "progress", detail

    def progress(self) -> float:
        """Fraction of the objective that actually exists in the world."""
        return 1.0 if (self.saved and self.doc_text == self.task.answer) else 0.0

    # ------------------------------------------------------------- rendering
    def frame(self) -> np.ndarray:
        img = Image.new("RGB", (CANVAS // 3, CANVAS // 3), (28, 32, 40))
        d = ImageDraw.Draw(img)
        win = self.view()
        s = 1 / 3
        if win.title != "desktop":
            d.rectangle([win.x * s, win.y * s, (win.x + win.w) * s, (win.y + win.h) * s],
                        fill=(246, 248, 252), outline=(90, 90, 110), width=3)
            d.rectangle([win.x * s, win.y * s, (win.x + win.w) * s, (win.y + 60) * s],
                        fill=(58, 66, 86))
            d.text((win.x * s + 12, win.y * s + 20), win.title.upper(), fill=(255, 255, 255))
        for e in win.elements:
            x0, y0 = e.x * s, e.y * s
            x1, y1 = (e.x + e.w) * s, (e.y + e.h) * s
            fill = (255, 255, 255) if e.role == "field" else (228, 234, 246)
            d.rectangle([x0, y0, x1, y1], fill=fill, outline=(120, 130, 150))
            text = e.label if e.label != "document" else (self.doc_text or "")
            d.text((x0 + 6, y0 + 6), text[:34], fill=(20, 24, 32))
        return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------------------ episodes
def _scan(layout_els) -> str:
    items = sorted(layout_els, key=lambda e: e.label)
    return ("<|scan|>elements on screen: "
            + "; ".join(f"{e.label} {e.role}@{int(e.cx)},{int(e.cy)}"
                        for e in items))


def web_episode(rng: random.Random) -> str:
    """One complete research-and-report transcript, clicks and all.

    The coordinates come from the world's own element boxes, so the transcript
    is exactly what the runtime will accept.
    """
    task = WebTask(rng)
    world = WebWorld(task)
    parts = [f"<|drive|>objective: {task.objective()}"]
    step = 0

    def click(label: str, expect: str, obs: str | None = None) -> None:
        nonlocal step
        step += 1
        el = next(e for e in world.view().elements if e.label == label)
        x, y = int(el.cx), int(el.cy)
        before = world.observe()
        parts.append(f"<|step|>{step}<|act|>click({x},{y})")
        res = world.act({"type": "click"}, (el.cx, el.cy))
        parts.append(f"<|obs|>{obs or res['detail']}")
        ok = world.progress() >= 1.0
        parts.append("<|check|>objective satisfied" if ok
                     else f"<|check|>objective not satisfied: {expect}")
        after = world.observe()
        world.verify(before, after, step, res)

    parts.append(_scan(world.view().elements))
    click("chrome", "chrome is not open yet", "chrome opened, search field ready")
    parts.append(_scan(world.view().elements))
    click(task.topic, "the right search result is not open",
          f"opened the article on {task.topic}")
    parts.append(_scan(world.view().elements))
    click("copy", "the answer is not on the clipboard",
          f"copied {task.answer} to the clipboard")
    parts.append(_scan(world.view().elements))
    click("docs", "google docs is not open", "google docs opened")
    parts.append(_scan(world.view().elements))
    click("paste", "the document is still empty",
          f"pasted {task.answer} into the document")
    click("save", "the document is not saved", f"saved the document: {task.answer}")
    parts.append("<|done|>success")
    return "".join(parts)


def fixture_layout(rng: random.Random):
    """A ui_world layout, used only to keep the two worlds' vocabulary aligned."""
    return make_layout(rng)
