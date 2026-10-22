"""Drive: the persistence corpus.

The brain is trained to finish. Not by prose, by distribution: every transcript
in this module is an episode that only ends in a terminal token, and most of
them contain a failed attempt followed by a recovery. The model never sees a
trajectory that simply stops, and it never sees one that gives up while budget
remains.

Three generators:

* ``ui_episode``      -- desktop task: observe -> plan -> act -> check -> retry
                         -> ... -> done, with a deliberate wrong first move.
* ``code_episode``    -- write code -> run tests -> read the error -> patch ->
                         rerun -> green, repeated until it passes.
* ``plan_episode``    -- decompose an objective, execute steps, absorb an
                         obstacle, decompose again.

Vocabulary of the drive protocol (all inside the existing tokenizer):

    <|drive|> objective: ...
    <|step|> n <|act|> action(...)
    <|obs|>   what the screen / test run said
    <|check|> whether the objective is satisfied
    <|retry|> why the previous attempt failed, what changes
    <|done|>  terminal: success | success-partial | budget-exhausted

Invariants the trainer relies on:
  1. every episode ends with exactly one <|done|>
  2. no episode is emitted between <|retry|> and the next attempt
  3. the first attempt fails in >= 50% of episodes, so "failure" is
     correlated with "next attempt", never with "end"

Diversity note (M0 measurement): the first version of these generators drew
episodes from a fixed table, which made 32% of a 3k-episode sample exact
duplicates -- at 1M episodes that is a corpus of a thousand sentences repeated
a thousand times. Every generator now derives its literals (test arguments,
file names, pids, paths, timings) from the episode RNG, and every code episode
fails at least once. `dataforge stats` measures the duplicate rate on every
build.
"""

from __future__ import annotations

import random

from .ui_world import SCALE, LABELS, ROLES, make_layout, TASKS, ACTION_OF_ROLE

# --------------------------------------------------------------------- helpers


def _coords(el, rng: random.Random) -> tuple[int, int]:
    x = int(rng.uniform(el.x + el.w * 0.25, el.x + el.w * 0.75)) * SCALE
    y = int(rng.uniform(el.y + el.h * 0.25, el.y + el.h * 0.75)) * SCALE
    return x, y


def _near_miss(el, rng: random.Random) -> tuple[int, int]:
    """A click that misses: lands just outside the widget, like a real slip."""
    side = rng.choice("udlr")
    pad = rng.randint(4, 14)
    if side == "u":
        return (int(el.cx) * SCALE, max(0, (el.y - pad)) * SCALE)
    if side == "d":
        return (int(el.cx) * SCALE, (el.y + el.h + pad) * SCALE)
    if side == "l":
        return (max(0, el.x - pad) * SCALE, int(el.cy) * SCALE)
    return ((el.x + el.w + pad) * SCALE, int(el.cy) * SCALE)


# ------------------------------------------------------------------ ui episode
# Protocol (agent 01, brainbox/lm_drive.py on main): the runtime hands the model
# a scan of the screen -- "label role at x,y" in the same 0..1920 space the
# actions use -- and accepts exactly four terminals: success | failed | blocked |
# impossible. The corpus must speak that dialect or the model learns a language
# nothing at runtime speaks.
SCAN_TERMINALS = ("success", "failed", "blocked", "impossible")

OBS_BY_ROLE = {
    "button": "button {label} highlighted",
    "field": "caret blinking in {label} field",
    "checkbox": "checkbox {label} is now {state}",
    "tab": "tab {label} is active",
    "icon": "icon {label} selected",
    "menu": "menu closed after {label}",
    "slider": "slider {label} moved to {pct}%",
}
NO_CHANGE = [
    "no visible change",
    "screen unchanged",
    "focus did not move",
    "no repaint in the target region",
    "hover highlight only",
]

# specs/drive.md rung vocabulary: the model has to learn which rung follows
# which failure, and that a rung is tried before the next one is reached.
LADDER = {
    "retry_exact": "try the same action again at the same coordinates",
    "retry_shifted": "resample the click inside the target box",
    "replan": "replan from the current screen",
    "decompose": "split the objective into subtasks",
    "tool_switch": "use the shell path instead of the GUI path",
    "escalate": "park the task for the next idle window",
}
UI_FAILS = [
    ("wrong element", "retry_shifted"),
    ("click landed off target", "retry_shifted"),
    ("element scrolled out of view", "retry_exact"),
    ("modal dialog blocked the click", "retry_exact"),
    ("focus stayed in another window", "replan"),
    ("double activation toggled it twice", "retry_exact"),
    ("window repainted late", "retry_exact"),
]
BLOCKERS = [
    ("a permission prompt appeared and cannot be dismissed from here", "escalate"),
    ("the window is not responding to synthetic input", "tool_switch"),
    ("the operation needs admin rights this session does not have", "escalate"),
    ("a modal dialog is waiting for a human decision", "escalate"),
]


def scan_line(layout) -> str:
    """The element inventory the runtime hands the model before it acts.

    Byte-identical in shape to ``brainbox.lm_drive.scan_from_obs``:
    ``<|scan|>elements on screen: <label> <role> at <x>,<y>; ...`` sorted by
    label, coordinates in canvas pixels times SCALE (0..1920). Every episode
    carries one -- solvable, blocked, exhausted or impossible -- so the marker
    never predicts the answer.
    """
    items = sorted(layout.elements, key=lambda e: e.label)
    return ("<|scan|>elements on screen: "
            + "; ".join(f"{e.label} {e.role} at {int(e.cx) * SCALE},{int(e.cy) * SCALE}"
                        for e in items))


def _scan_centre(el) -> tuple[int, int]:
    return int(el.cx) * SCALE, int(el.cy) * SCALE


def _coords(el, rng: random.Random) -> tuple[int, int]:
    """A click on the widget. 70% of the time it copies the coordinate the scan
    advertises (the centre), because that is the behaviour the runtime needs:
    read the scan, click what it says. The rest is a human-ish point inside the
    box, so the model does not learn that only the exact centre is legal."""
    if rng.random() < 0.70:
        return _scan_centre(el)
    x = int(rng.uniform(el.x + el.w * 0.25, el.x + el.w * 0.75)) * SCALE
    y = int(rng.uniform(el.y + el.h * 0.25, el.y + el.h * 0.75)) * SCALE
    return x, y


def _near_miss(el, rng: random.Random) -> tuple[int, int]:
    """A click that misses: lands just outside the widget, like a real slip."""
    side = rng.choice("udlr")
    pad = rng.randint(4, 14)
    if side == "u":
        return (int(el.cx) * SCALE, max(0, (el.y - pad)) * SCALE)
    if side == "d":
        return (int(el.cx) * SCALE, (el.y + el.h + pad) * SCALE)
    if side == "l":
        return (max(0, el.x - pad) * SCALE, int(el.cy) * SCALE)
    return ((el.x + el.w + pad) * SCALE, int(el.cy) * SCALE)


def _act(el, x: int, y: int) -> str:
    return f"<|act|>{ACTION_OF_ROLE[el.role]}({x},{y})"


def ui_episode(rng: random.Random) -> str:
    """One desktop episode, ending in exactly one of the four runtime terminals.

    Mix (measured by `dataforge stats`):
      success    ~71%  wrong attempt -> near miss -> recovery, or clean
      failed     ~18%  attempts exhaust the step budget -> <|budget|> -> failed
      blocked    ~5%   an obstruction only a human can clear -> blocked (parked)
      impossible ~6%   the objective names something the scan does not contain
    """
    layout = make_layout(rng)
    r = rng.random()
    if r < 0.06:
        return impossible_episode(rng, layout)
    if r < 0.11:
        return blocked_episode(rng, layout)
    if r < 0.34:
        return exhausted_episode(rng, layout)
    return _success_episode(rng, layout)


def _success_episode(rng: random.Random, layout) -> str:
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    others = [e for e in layout.elements if e is not target]
    parts = [f"<|drive|>objective: {task}", scan_line(layout)]
    step = 0

    if others and rng.random() < 0.60:
        distractor = rng.choice(others)
        step += 1
        dx, dy = _coords(distractor, rng)
        parts.append(f"<|step|>{step}{_act(distractor, dx, dy)}")
        parts.append(f"<|obs|>{OBS_BY_ROLE[distractor.role].format(label=distractor.label, state='off', pct=0)}")
        parts.append("<|check|>objective not satisfied: wrong element")
        parts.append(f"<|retry|>that was {distractor.label}, not {target.label}. "
                     f"target the {target.role} labelled {target.label}")
    if rng.random() < 0.85:
        why, rung = rng.choice(UI_FAILS[1:])
        step += 1
        if why == "click landed off target":
            nx, ny = _near_miss(target, rng)
        else:
            nx, ny = _coords(target, rng)
        parts.append(f"<|step|>{step}{_act(target, nx, ny)}")
        parts.append(f"<|obs|>{rng.choice(NO_CHANGE) if why == 'click landed off target' else why}")
        parts.append(f"<|check|>objective not satisfied: {why}")
        parts.append(f"<|retry|>aim at the centre of the {target.role}, {target.label} "
                     f"({_scan_centre(target)[0]},{_scan_centre(target)[1]})"
                     if why == "click landed off target" else
                     f"<|retry|>{why}: {LADDER[rung]}, then use the scan coordinate")

    step += 1
    gx, gy = _coords(target, rng)
    parts.append(f"<|step|>{step}{_act(target, gx, gy)}")
    parts.append(f"<|obs|>{OBS_BY_ROLE[target.role].format(label=target.label, state='on', pct=rng.randint(20, 90))}")
    parts.append("<|check|>objective satisfied")
    parts.append("<|done|>success")
    return "".join(parts)


def exhausted_episode(rng: random.Random, layout) -> str:
    """The loop spends its step budget and says so. `<|done|>failed`, never a
    fabricated success -- that is the overclaim the fleet measured at 1.0."""
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    parts = [f"<|drive|>objective: {task}", scan_line(layout)]
    n = rng.randint(2, 4)
    for step in range(1, n + 1):
        other = rng.choice([e for e in layout.elements if e is not target] or layout.elements)
        x, y = _coords(other, rng)
        parts.append(f"<|step|>{step}{_act(other, x, y)}")
        parts.append(f"<|obs|>{rng.choice(NO_CHANGE)}")
        parts.append(f"<|check|>objective not satisfied: {other.label} is not {target.label}")
        if step < n:
            parts.append(f"<|retry|>target the {target.role} labelled {target.label}, not {other.label}")
    parts.append("<|budget|>steps exhausted")
    parts.append("<|done|>failed: retries exhausted, objective not reached")
    return "".join(parts)


def blocked_episode(rng: random.Random, layout) -> str:
    """An obstruction a human has to clear. The honest end is `blocked`, and the
    episode is parked rather than dropped (specs/drive.md, rung 6)."""
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    reason, rung = rng.choice(BLOCKERS)
    parts = [f"<|drive|>objective: {task}", scan_line(layout)]
    x, y = _coords(target, rng)
    parts.append(f"<|step|>1{_act(target, x, y)}")
    parts.append(f"<|obs|>{reason}")
    parts.append(f"<|check|>objective not satisfied: {reason}")
    parts.append(f"<|retry|>{reason}: {LADDER[rung]}")
    x2, y2 = _coords(target, rng)
    parts.append(f"<|step|>2{_act(target, x2, y2)}")
    parts.append("<|obs|>the obstruction is still there")
    parts.append("<|check|>objective not satisfied: blocked")
    parts.append("<|done|>blocked: parked for the next idle window")
    return "".join(parts)


def impossible_episode(rng: random.Random, layout=None) -> str:
    """Objective names something that is not on this screen.

    Two shapes, both honest: read the scan and say so immediately (the behaviour
    the runtime wants), or waste one attempt and then say so. Neither clicks
    around hopefully and neither claims success.
    """
    layout = layout or make_layout(rng)
    present = {e.label for e in layout.elements}
    missing = [l for l in LABELS if l not in present]
    label = rng.choice(missing)
    role = rng.choice(ROLES)
    task = rng.choice(TASKS[role]).format(label=label)
    parts = [f"<|drive|>objective: {task}", scan_line(layout)]
    if rng.random() < 0.30:
        # one honest attempt, no <|retry|> before the terminal (invariant 2)
        other = rng.choice(layout.elements)
        x, y = _coords(other, rng)
        parts.append(f"<|step|>1{_act(other, x, y)}")
        parts.append(f"<|obs|>{rng.choice(NO_CHANGE)}")
        parts.append(f"<|check|>objective not satisfied: {label} is not in the scan")
        parts.append(f"<|step|>2{_act(other, x, y)}")
        parts.append("<|obs|>no visible change")
        parts.append(f"<|check|>{label} is not present on this screen")
    else:
        parts.append(f"<|check|>{label} is not present on this screen")
    parts.append(f"<|done|>impossible: no {role} labelled {label} exists here")
    return "".join(parts)


# ---------------------------------------------------------------- code episode
def _fn(src_sig: str, body: str):
    """Build the reference implementation: the expected value of a test comes
    from executing the *good* body, so a randomised assertion can never lie."""
    ns: dict = {}
    exec(src_sig + "\n" + body, ns)  # noqa: S102 - our own literals only
    name = src_sig.split("(")[0].replace("def ", "").strip()
    return ns[name]


def _ints(rng, k, lo=1, hi=24) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(k)]


def _word(rng) -> str:
    return rng.choice(["alpha", "beta", "delta", "gamma", "omega", "kilo", "sigma",
                       "tango", "zulu", "rust", "forge", "atlas"])


# name -> how to build the call arguments for one randomised test
ARG_GEN = {
    "add": lambda rng: f"{rng.randint(1, 40)}, {rng.randint(1, 40)}",
    "double": lambda rng: f"{rng.randint(1, 50)}",
    "positives": lambda rng: repr(_ints(rng, rng.randint(3, 6), -9, 9)),
    "biggest": lambda rng: repr(_ints(rng, rng.randint(3, 6), 1, 99)),
    "flip": lambda rng: repr("".join(rng.choice("abcdefgh") for _ in range(rng.randint(3, 8)))),
    "count_unique": lambda rng: repr([rng.randint(1, 4) for _ in range(rng.randint(3, 7))]),
    "mean": lambda rng: repr(_ints(rng, rng.randint(3, 6), 1, 20)),
    "first_word": lambda rng: repr(" ".join(_word(rng) for _ in range(rng.randint(2, 4)))),
    "summarize": lambda rng: repr(_ints(rng, rng.randint(2, 5), 0, 9)),
    "keys_positive": lambda rng: repr({_word(rng): rng.randint(-3, 3)
                                       for _ in range(rng.randint(2, 4))}),
    "clamp": lambda rng: f"{rng.randint(-30, 30)}, {rng.randint(-10, 0)}, {rng.randint(1, 10)}",
    "slug": lambda rng: repr(" ".join(_word(rng).capitalize() if rng.random() < 0.5 else _word(rng)
                                      for _ in range(rng.randint(2, 4)))),
    "dedupe": lambda rng: repr([rng.randint(1, 5) for _ in range(rng.randint(2, 6))]),
    "median": lambda rng: repr(_ints(rng, rng.randint(3, 7), 0, 40)),
    "is_even": lambda rng: f"{rng.randint(0, 40)}",
    "pluralise": lambda rng: f"{rng.randint(0, 5)}, {rng.choice(['file', 'item', 'row', 'task'])!r}",
    "pct": lambda rng: f"{rng.randint(0, 40)}, {rng.randint(41, 100)}",
}

BUGS = [
    # (sig, good body, buggy body)
    ("def add(a, b):", "    return a + b", "    return a - b"),
    ("def double(n):", "    return n * 2", "    return n + 2"),
    ("def positives(xs):", "    return [x for x in xs if x > 0]",
     "    return [x for x in xs if x < 0]"),
    ("def biggest(items):", "    return max(items)", "    return min(items)"),
    ("def flip(s):", "    return s[::-1]", "    return s[::2]"),
    ("def count_unique(items):\n    unique = set(items)", "    return len(unique)",
     "    return len(items)"),
    ("def mean(values):\n    total = sum(values)", "    return round(total / len(values), 2)",
     "    return round(total * len(values), 2)"),
    ("def first_word(text):", "    return text.split()[0]", "    return text.split()[-1]"),
    ("def summarize(items):", "    return {'ok': True, 'n': len(items)}",
     "    return {'ok': False, 'n': len(items)}"),
    ("def keys_positive(d):", "    return [k for k in d if d[k] > 0]",
     "    return [k for k in d if d[k] < 0]"),
    ("def clamp(x, lo, hi):", "    return max(lo, min(hi, x))", "    return min(lo, max(hi, x))"),
    ("def slug(text):", "    return '-'.join(text.lower().split())",
     "    return '_'.join(text.lower().split())"),
    ("def dedupe(xs):", "    return list(dict.fromkeys(xs))", "    return list(reversed(xs))"),
    ("def median(xs):\n    s = sorted(xs)", "    return s[len(s) // 2]", "    return s[0]"),
    ("def is_even(n):", "    return n % 2 == 0", "    return n % 2 == 1"),
    ("def pluralise(n, word):", "    return word if n == 1 else word + 's'",
     "    return word + 's' if n == 1 else word"),
    ("def pct(part, whole):", "    return round(100.0 * part / whole, 1)",
     "    return round(part / whole, 1)"),
]

WHY = [
    "wrong operator", "wrong direction of the comparison", "off by one element",
    "early return skips the result", "the branch checks the wrong variable",
    "the loop mutates the list it is scanning", "the guard is inverted",
    "integer division where a float is required", "off by one in the slice",
    "the accumulator is reset inside the loop",
]
ERRORS = ["AssertionError", "TypeError", "ValueError", "IndexError", "NameError"]


SWAPS = [
    (" + ", " - ", "wrong operator: it subtracts where it should add"),
    (" - ", " + ", "wrong operator: it adds where it should subtract"),
    (" > ", " >= ", "the comparison is inclusive where it must be strict"),
    (" < ", " <= ", "the comparison is inclusive where it must be strict"),
    ("max(", "min(", "it takes the minimum while the test asks for the maximum"),
    ("[0]", "[-1]", "off by one in the index: last element instead of first"),
    ("True", "False", "the success flag is inverted"),
    ("**", "*", "exponentiation where multiplication was meant"),
    ("len(", "sum(", "it sums the values instead of counting them"),
]


def _mutate(body: str, rng: random.Random, avoid: set[str] | None = None) -> tuple[str, str]:
    """A plausible wrong variant, plus WHY it is wrong.

    The reason is emitted as the <|think|> line, so the corpus teaches
    "error -> diagnosis -> patch" rather than a random sentence next to a
    random body. Variants already used in this episode are skipped, so attempt
    2 and attempt 3 are different bugs on the same function.
    """
    avoid = avoid or set()
    for a, b, why in rng.sample(SWAPS, k=len(SWAPS)):
        if a in body:
            cand = body.replace(a, b, 1)
            if cand not in avoid:
                return cand, why
    cand = body.replace("return", "return not", 1)
    return cand, "the guard is inverted: it returns the negation"


def code_episode(rng: random.Random) -> str:
    sig, good, bad = rng.choice(BUGS)
    name = sig.split("(")[0].replace("def ", "").strip()
    arg_src = ARG_GEN[name](rng)
    want = repr(_fn(sig, good)(*eval(f"({arg_src},)", {"__builtins__": {}})))  # noqa: S307
    assertion = f"assert {name}({arg_src}) == {want}"

    attempts = rng.randint(2, 4)  # never zero failures: I3 is a hard invariant
    err_kind = rng.choices(ERRORS, weights=[70, 8, 8, 7, 7])[0]
    parts = [f"<|task|>make the tests pass for {name}()"]
    used: set[str] = {bad}
    why = ""
    for attempt in range(1, attempts + 1):
        if attempt == attempts:
            body = good
        elif attempt == 1:
            body = bad
            why = "wrong operator" if body != good else why
        else:
            body, why = _mutate(bad, rng, avoid=used)
            used.add(body)
        parts.append(f"<|try|>{attempt}")
        parts.append(f"<|code|>{sig}\n{body}")
        parts.append("<|run|>pytest -q")
        if attempt < attempts:
            parts.append(f"<|err|>{assertion}  ->  {err_kind}")
            parts.append(f"<|think|>{why or rng.choice(WHY)}")
            parts.append(f"<|retry|>attempt {attempt} failed, patch the logic and rerun")
        else:
            parts.append(f"<|ok|>1 passed in {rng.uniform(0.01, 0.9):.2f}s")
            parts.append("<|check|>objective satisfied")
    parts.append("<|done|>success")
    return "".join(parts)


# ---------------------------------------------------------------- plan episode
DIRS = ["downloads", "desktop", "documents", "pictures", "tmp", "projects"]
EXTS = ["iso", "log", "tmp", "png", "zip", "csv"]
FRAMEWORKS = ["flask", "fastapi", "click", "aiohttp"]
OBJECTIVES = [
    lambda rng: f"delete every {rng.choice(EXTS)} file in {rng.choice(DIRS)}",
    lambda rng: f"tidy the {rng.choice(DIRS)} folder into subfolders by type",
    lambda rng: f"rename {rng.randint(3, 400)} screenshots to the date they were taken",
    lambda rng: f"scaffold a {rng.choice(FRAMEWORKS)} app and run its tests",
    lambda rng: f"close every window that has been idle for {rng.choice([15, 30, 45, 60, 90])} minutes",
    lambda rng: f"export the {rng.choice(['q3', 'budget', 'roster', 'inventory'])} spreadsheet "
                f"to csv and open the result",
    lambda rng: f"move the {rng.randint(2, 40)} newest {rng.choice(EXTS)} files into an archive folder",
]
STEPS = [
    "capture the screen", "list the matching files", "sort by the rule in the objective",
    "apply the change", "verify the change on screen", "write the run log",
    "check free space", "close stale handles",
]
OBSTACLES = [
    (lambda rng: f"file {_word(rng)}.{rng.choice(EXTS)} is locked by pid {rng.randint(1000, 9999)}",
     "close the handle, then continue"),
    (lambda rng: f"permission denied on /{rng.choice(DIRS)}/{_word(rng)}.{rng.choice(EXTS)}",
     "retry with the user's own directory instead"),
    (lambda rng: f"window '{_word(rng).title()} {rng.randint(1, 40)}' did not respond",
     "focus the window, wait, then continue"),
    (lambda rng: "wrong menu opened", "press escape and open the intended menu"),
    (lambda rng: "element scrolled out of view", "scroll to it, then click"),
    (lambda rng: f"disk {rng.choice(['C:', 'D:', '/mnt/data'])} reports "
                 f"{rng.randint(10, 900)} MB free", "free space first, then continue"),
]


def plan_episode(rng: random.Random) -> str:
    layout = make_layout(rng)
    objective = rng.choice(OBJECTIVES)(rng)
    steps = rng.sample(STEPS, k=rng.randint(4, 6))
    n_obs = 1 if rng.random() < 0.55 else 2

    parts = [f"<|drive|>objective: {objective}", scan_line(layout), "<|plan|>"]
    for i, s in enumerate(steps, 1):
        parts.append(f"{i}.{s} ")
    hits = sorted(rng.sample(range(2, max(3, len(steps) + 1)), k=min(n_obs, max(1, len(steps) - 1))))
    for i, s in enumerate(steps, 1):
        el = layout.elements[(i - 1) % len(layout.elements)]
        x, y = _coords(el, rng)
        parts.append(f"<|step|>{i}{_act(el, x, y)}")
        if i in hits:
            obstacle_fn, fix = rng.choice(OBSTACLES)
            obstacle = obstacle_fn(rng)
            parts.append(f"<|obs|>{obstacle}")
            parts.append("<|check|>objective not satisfied: blocked")
            parts.append(f"<|retry|>{fix}")
            fix_tok = "".join(c if c.isalnum() or c == "_" else "_" for c in fix.replace(" ", "_"))
            fx, fy = _coords(el, rng)
            parts.append(f"<|step|>{i}b{_act(el, fx, fy)}")
            parts.append("<|obs|>step completed")
            if len(hits) > 1 and i == hits[0]:
                parts.append(f"<|retry|>{LADDER['tool_switch']}")
                sx, sy = _coords(el, rng)
                parts.append(f"<|step|>{i}c{_act(el, sx, sy)}")
                parts.append("<|obs|>step completed without the GUI")
        else:
            parts.append(f"<|obs|>step completed ({rng.randint(1, 60)} items)")
    parts.append("<|check|>objective satisfied")
    parts.append(f"<|done|>{rng.choices(['success', 'failed'], weights=[90, 10])[0]}")
    return "".join(parts)


# -------------------------------------------------------------------- factory
def drive_corpus(n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        if r < 0.45:
            yield ui_episode(rng)
        elif r < 0.85:
            yield code_episode(rng)
        else:
            yield plan_episode(rng)


def drive_stats(n: int = 2000, seed: int = 0) -> dict:
    """Sanity numbers the forge prints: retries per episode, terminal rate."""
    eps = list(drive_corpus(n, seed))
    retries = sum(e.count("<|retry|>") for e in eps)
    done = sum(e.count("<|done|>") for e in eps)
    fails = sum(1 for e in eps if "<|retry|>" in e)
    terms = {t: 0 for t in SCAN_TERMINALS} | {"other": 0}
    for e in eps:
        tail = e.rsplit("<|done|>", 1)[-1] if "<|done|>" in e else "other"
        word = tail.split(":")[0].strip().split()[0] if tail.strip() else "other"
        terms[word] = terms.get(word, 0) + 1
    return {
        "episodes": len(eps),
        "mean_retries": round(retries / len(eps), 2),
        "terminal_rate": round(done / len(eps), 3),
        "failure_fraction": round(fails / len(eps), 3),
        "terminal_mix": {k: round(v / len(eps), 3) for k, v in terms.items()},
        "mean_chars": round(sum(len(e) for e in eps) / len(eps), 1),
    }
