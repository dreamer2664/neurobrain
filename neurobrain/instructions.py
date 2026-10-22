"""Instruction synthesis: the words a user says, at scale.

Ten template families x slots x paraphrases, plus the three things real users
do that naive generators forget:

* **multi-constraint** -- "click save, not cancel": the scene contains both, and
  the instruction is only satisfiable if the model reads the negation;
* **relative reference** -- "the button under the search bar": the target has no
  unique name, it has a position;
* **degenerate typing** -- all-lowercase, missing apostrophes, one-keystroke
  typos, trailing noise. A model that only sees canonical text fails on the
  first real user.

Every sample is a `dict` with a stable schema (`RECORD_FIELDS`); the text is
what the language brain reads, the fields are what a verifier can check. The
engine is deterministic given a `random.Random`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import random

# ------------------------------------------------------------------ vocabulary
ROLE_VERBS = {
    "button": ["click", "press", "hit", "tap", "select", "activate"],
    "tab": ["open", "switch to", "click", "go to", "select"],
    "icon": ["click", "open", "select", "double-click", "tap"],
    "menu": ["choose", "click", "pick", "select", "open"],
    "checkbox": ["toggle", "check", "tick", "turn on", "enable"],
    "field": ["type into", "focus", "click into", "put the cursor in", "fill"],
    "slider": ["drag", "move", "adjust", "slide", "set"],
}
UNCHECK_VERBS = ["uncheck", "untick", "turn off", "disable", "clear"]
TYPE_VERBS = ["type", "enter", "write", "put", "paste"]

ROLES = ["button", "tab", "icon", "menu", "checkbox", "field", "slider"]
SECTIONS = ["toolbar", "sidebar", "tab strip", "menu bar", "dialog", "list", "settings panel"]
APPS = ["Files", "Settings", "Mail", "Editor", "Photos", "Store", "Terminal", "Notes",
        "Calendar", "Vault", "Studio", "Player", "Sync", "Browser"]

# Real interface vocabulary, including the multiword labels that make a
# grounding task hard ("auto save" vs "save", "dark mode" vs "mode").
LABELS = [
    "save", "cancel", "submit", "delete", "rename", "export", "import", "print",
    "search", "filter", "download", "upload", "share", "archive", "refresh",
    "settings", "profile", "inbox", "login", "logout", "play", "pause", "help",
    "home", "cart", "close", "retry", "undo", "history", "backup", "sync", "restore",
    "auto save", "dark mode", "light mode", "high contrast", "remember me",
    "show hidden files", "two factor", "notifications", "do not disturb", "airplane mode",
    "bluetooth", "wifi", "proxy", "language", "region", "time zone", "keyboard",
    "shortcuts", "appearance", "theme", "font size", "zoom", "line spacing",
    "word wrap", "spell check", "autocorrect", "live preview", "grid view", "list view",
    "sort by name", "sort by date", "group by type", "show sidebar", "hide sidebar",
    "new folder", "new document", "new window", "new tab", "duplicate", "move to",
    "copy link", "open in new tab", "add to favourites", "remove from favourites",
    "mark as read", "mark as unread", "move to spam", "reply all", "forward",
    "attach file", "insert image", "insert table", "add comment", "resolve",
    "assign to", "due date", "priority", "labels", "milestone", "sprint",
    "billing", "invoices", "payment method", "plan", "upgrade", "downgrade",
    "seats", "usage", "limits", "quota", "api keys", "webhooks", "integrations",
    "members", "roles", "permissions", "audit log", "sessions", "devices",
    "backup now", "restore point", "factory reset", "clear cache", "clear data",
    "check for updates", "release notes", "beta channel", "telemetry", "crash reports",
    "privacy", "cookies", "tracking", "ad personalisation", "location access",
    "camera access", "microphone access", "screen recording", "screen sharing",
    "record", "stop recording", "mute", "unmute", "volume", "brightness", "night light",
    "battery saver", "power mode", "sleep after", "lock screen", "sign out",
    "switch account", "add account", "verify email", "change password", "recovery code",
    "export data", "delete account", "contact support", "report a bug", "send feedback",
    "changelog", "documentation", "keyboard shortcuts", "what's new", "tour",
]

FILE_STEMS = ["report", "budget", "notes", "draft", "export", "scan", "photo", "log",
              "backup", "invoice", "roster", "plan", "sheet", "chart", "audio", "clip",
              "atlas", "forge", "sample", "final", "archive", "capture", "printer", "map"]
FILE_EXTS = ["txt", "py", "png", "csv", "log", "pdf", "zip", "md"]

TYPE_TEXTS = ["hello world", "2024-06-01", "invoice_07", "quarterly report", "user@example.com",
              "~/projects/atlas", "3.14", "untitled-2", "chapter one", "127.0.0.1:8080",
              "new folder", "final_final_v3", "%USERPROFILE%\\Downloads"]

ANCHORS = ["search bar", "toolbar", "menu bar", "status bar", "tab strip", "scrollbar",
           "title bar", "sidebar", "search field", "address bar"]


def _cap(s: str, title: bool) -> str:
    if not title:
        return s
    return s[0].upper() + s[1:]


# ----------------------------------------------------------------------- typos
def typo(word: str, rng: random.Random) -> str:
    """One realistic keystroke error; the word stays recoverable."""
    if len(word) < 3:
        return word
    kind = rng.choice(["swap", "drop", "double", "drop_e"])
    i = rng.randrange(0, len(word) - 1)
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    return word[:-1] if word.endswith("e") else word[:i] + word[i + 1:]


def _typo_text(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    words[i] = typo(words[i], rng)
    return " ".join(words)


# -------------------------------------------------------------------- records
RECORD_FIELDS = ["text", "target", "distractor", "anchor", "constraints", "family",
                 "variant", "app", "section", "type_text", "value"]


@dataclass
class Instruction:
    text: str
    target: dict
    family: str
    variant: str = "plain"
    distractor: dict | None = None
    anchor: str | None = None
    constraints: list[str] = field(default_factory=list)
    app: str | None = None
    section: str | None = None
    type_text: str | None = None
    value: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in RECORD_FIELDS if k in d}


def _target(rng: random.Random, role: str | None = None, label: str | None = None) -> dict:
    return {"label": label or rng.choice(LABELS), "role": role or rng.choice(ROLES)}


def _different(rng: random.Random, taken: list[str]) -> str:
    for _ in range(30):
        lab = rng.choice(LABELS)
        if lab not in taken:
            return lab
    return rng.choice([l for l in LABELS if l not in taken] or LABELS)


# ------------------------------------------------------------------- families
# Each family returns an Instruction. The families are deliberately uneven in
# their output space so the corpus is not dominated by the simplest template.

def scene_clause(rng: random.Random) -> str:
    """A locating clause: which file, row, page or app the instruction is about.

    This is what takes each family's core space from a few hundred strings to a
    few hundred thousand without making the sentences unnatural.
    """
    kind = rng.randrange(6)
    if kind == 0:
        return f" for {rng.choice(FILE_STEMS)}.{rng.choice(FILE_EXTS)}"
    if kind == 1:
        return f" in row {rng.randint(1, 40)}"
    if kind == 2:
        return f" on page {rng.randint(1, 9)}"
    if kind == 3:
        return f" under the {rng.choice(ANCHORS)}"
    if kind == 4:
        return f" in {rng.choice(APPS)}"
    return f" in {rng.choice(SECTIONS)}"


def fam_plain(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    return Instruction(text=f"{verb} the {tgt['label']} {tgt['role']}", target=tgt,
                       family="plain", section=section)


def fam_bare(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    return Instruction(text=f"{verb} {tgt['label']}", target=tgt, family="bare")


def fam_app(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng)
    app = rng.choice(APPS)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    style = rng.choice([
        f"in {app}, {verb} the {tgt['label']} {tgt['role']}",
        f"{verb} the {tgt['label']} {tgt['role']} in {app}",
        f"open {app} and {verb} {tgt['label']}",
    ])
    return Instruction(text=style, target=tgt, family="app", app=app)


def fam_section(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng)
    sec = section or rng.choice(SECTIONS)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    return Instruction(text=f"{verb} the {tgt['label']} {tgt['role']} in the {sec}",
                       target=tgt, family="section", section=sec)


def fam_relative(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng)
    anchor = rng.choice(ANCHORS)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    rel = rng.choice(["under", "next to", "above", "below", "right of", "left of"])
    style = rng.choice([
        f"{verb} the {tgt['role']} {rel} the {anchor}",
        f"{verb} the {tgt['role']} {rel} the {anchor}, the one labelled {tgt['label']}",
        f"the {tgt['role']} {rel} the {anchor}",
    ])
    return Instruction(text=style, target=tgt, family="relative", anchor=anchor,
                       constraints=[f"{rel} the {anchor}"])


def fam_multi(rng: random.Random, section: str | None = None) -> Instruction:
    """Two constraints in one sentence: the target and the thing to avoid."""
    tgt = _target(rng)
    other = _different(rng, [tgt["label"]])
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    style = rng.choice([
        f"{verb} {tgt['label']}, not {other}",
        f"{verb} {tgt['label']} instead of {other}",
        f"{verb} {tgt['label']} but leave {other} alone",
        f"not {other}, {verb} {tgt['label']}",
        f"{verb} the {tgt['label']} {tgt['role']}, do not touch {other}",
    ])
    return Instruction(text=style, target=tgt, family="multi",
                       distractor={"label": other, "role": tgt["role"]},
                       constraints=[f"not {other}"])


def fam_sequence(rng: random.Random, section: str | None = None) -> Instruction:
    a = _target(rng)
    b = _target(rng)
    b["label"] = _different(rng, [a["label"]])
    v1 = rng.choice(ROLE_VERBS[a["role"]])
    v2 = rng.choice(ROLE_VERBS[b["role"]])
    style = rng.choice([
        f"first {v1} {a['label']}, then {v2} {b['label']}",
        f"{v1} the {a['label']} {a['role']} and afterwards {v2} {b['label']}",
    ])
    return Instruction(text=style, target=a, family="sequence",
                       distractor=b, constraints=[f"then {v2} {b['label']}"])


def fam_type(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng, role="field")
    verb = rng.choice(TYPE_VERBS)
    text = rng.choice(TYPE_TEXTS)
    style = rng.choice([
        f"{verb} {text} into the {tgt['label']} field",
        f"type {text} in {tgt['label']}",
        f"put {text} in the {tgt['label']} box",
    ])
    return Instruction(text=style, target=tgt, family="type", type_text=text)


def fam_slider(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng, role="slider")
    value = rng.choice([0, 10, 25, 33, 50, 66, 75, 90, 100])
    verb = rng.choice(ROLE_VERBS["slider"])
    style = rng.choice([
        f"{verb} the {tgt['label']} slider to {value}%",
        f"set {tgt['label']} to {value}",
        f"drag {tgt['label']} until it reads {value} percent",
    ])
    return Instruction(text=style, target=tgt, family="slider", value=value)


def fam_checkbox_off(rng: random.Random, section: str | None = None) -> Instruction:
    tgt = _target(rng, role="checkbox")
    verb = rng.choice(UNCHECK_VERBS)
    return Instruction(text=f"{verb} {tgt['label']}", target=tgt, family="checkbox_off",
                       constraints=["state: off"])


def fam_contextual(rng: random.Random, section: str | None = None) -> Instruction:
    """A sentence a real user would type into a task box, with an outcome."""
    tgt = _target(rng)
    app = rng.choice(APPS)
    verb = rng.choice(ROLE_VERBS[tgt["role"]])
    style = rng.choice([
        f"i'm in {app} and i need to {verb} the {tgt['label']} {tgt['role']}",
        f"could you {verb} {tgt['label']} for me in {app}",
        f"{app} is open, {verb} {tgt['label']} and tell me when it's done",
    ])
    return Instruction(text=style, target=tgt, family="contextual", app=app)


FAMILIES = [
    (fam_plain, 0.16), (fam_bare, 0.08), (fam_app, 0.12), (fam_section, 0.10),
    (fam_relative, 0.12), (fam_multi, 0.14), (fam_sequence, 0.06),
    (fam_type, 0.08), (fam_slider, 0.05), (fam_checkbox_off, 0.04),
    (fam_contextual, 0.05),
]


# ------------------------------------------------------------------- variants
PREFIXES = ["", "", "", "", "please ", "pls ", "hey, ", "quickly ", "hey ", "so ",
            "now ", "ok ", "kindly ", "can you ", "could you ", "i need you to "]
SUFFIXES = ["", "", "", "", "", ".", "!", "..", " thanks", " asap", " please", "?"]


def _case(text: str, style: str) -> str:
    if style == "title":
        return _cap(text, True)
    if style == "sentence":
        return _cap(text, True)
    return text.lower()


def apply_variant(ins: Instruction, rng: random.Random) -> Instruction:
    """The ways real input deviates from the template. Mutates and returns.

    Surface form is a product of four independent choices -- prefix, suffix,
    case, and a possible keystroke error -- which is what takes a few hundred
    template outputs to a space large enough for a million samples with less
    than one duplicate in a hundred.
    """
    parts = []
    prefix = rng.choice(PREFIXES)
    if prefix:
        parts.append("prefix")
    suffix = rng.choice(SUFFIXES)
    if suffix:
        parts.append("suffix")
    case = rng.choices(["lower", "title", "plain"], weights=[0.55, 0.30, 0.15])[0]
    if case != "plain":
        parts.append(case)
    text = _case(ins.text, case)
    if rng.random() < 0.18:
        text = _typo_text(text, rng)
        parts.append("typo")
    if rng.random() < 0.08:
        text = text.replace(" do not ", " don't ").replace(" it is ", " it's ")
        parts.append("contraction")
    text = prefix + text + suffix
    ins.variant = "+".join(parts) if parts else "plain"
    ins.text = text
    return ins


def sample_instruction(rng: random.Random) -> Instruction:
    fams, weights = zip(*FAMILIES)
    fam = rng.choices(fams, weights=weights, k=1)[0]
    ins = fam(rng)
    if rng.random() < 0.55:
        # weave a locating clause in before the surface-form pass, so typos and
        # case changes also apply to it
        clause = scene_clause(rng)
        base = ins.text
        ins.text = base + clause
        ins.constraints = list(ins.constraints) + [f"locator:{clause.strip()}"]
    return apply_variant(ins, rng)


# ------------------------------------------------------------------- indexing
def normalize(text: str) -> str:
    """Dedupe key: case, surrounding punctuation and whitespace are not
    information. Typos stay distinct on purpose -- they are real training
    samples, not noise to be collapsed."""
    t = " ".join(text.lower().split())
    return t.strip(" .!?,")


def instruction_space() -> int:
    """Rough size of the product space, for the dedupe headroom check."""
    n_verbs = sum(len(v) for v in ROLE_VERBS.values())
    base = (len(FAMILIES) * len(ROLES) * len(LABELS) * n_verbs
            * len(SECTIONS) * len(APPS) * len(ANCHORS) * len(TYPE_TEXTS))
    return base * 6  # variants
