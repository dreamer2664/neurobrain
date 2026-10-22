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
]


def ui_episode(rng: random.Random) -> str:
    layout = make_layout(rng)
    if rng.random() < 0.18:
        return _exhausted_episode(rng, layout)
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    action = ACTION_OF_ROLE[target.role]
    fails_first = rng.random() < 0.6

    parts = [f"<|drive|>objective: {task}"]
    step = 0

    if fails_first:
        distractor = rng.choice([e for e in layout.elements if e is not target])
        step += 1
        dx, dy = _coords(distractor, rng)
        parts.append(f"<|step|>{step}<|act|>{ACTION_OF_ROLE[distractor.role]}({dx},{dy})")
        parts.append(f"<|obs|>{OBS_BY_ROLE[distractor.role].format(label=distractor.label, state='off', pct=0)}")
        parts.append("<|check|>objective not satisfied: wrong element")
        parts.append(f"<|retry|>that was {distractor.label}, not {target.label}. "
                     f"target the {target.role} labelled {target.label}")

    # one honest near-miss on the right element, then the exact hit
    step += 1
    nx, ny = _near_miss(target, rng)
    parts.append(f"<|step|>{step}<|act|>{action}({nx},{ny})")
    parts.append(f"<|obs|>{rng.choice(NO_CHANGE)}")
    parts.append("<|check|>objective not satisfied: click landed off target")
    parts.append(f"<|retry|>aim at the centre of the {target.role}, {target.label}")

    step += 1
    gx, gy = _coords(target, rng)
    parts.append(f"<|step|>{step}<|act|>{action}({gx},{gy})")
    parts.append(f"<|obs|>{OBS_BY_ROLE[target.role].format(label=target.label, state='on', pct=rng.randint(20, 90))}")
    parts.append("<|check|>objective satisfied")
    parts.append("<|done|>success")
    return "".join(parts)


def _exhausted_episode(rng: random.Random, layout) -> str:
    """A task the model cannot finish: the loop spends its budget and says so.

    Without these, the corpus teaches "every objective ends in success", which
    is the most dangerous thing an unattended machine can believe.
    """
    target = rng.choice(layout.elements)
    task = rng.choice(TASKS[target.role]).format(label=target.label)
    parts = [f"<|drive|>objective: {task}"]
    for step in range(1, rng.randint(3, 5)):
        other = rng.choice(layout.elements)
        x, y = _coords(other, rng)
        parts.append(f"<|step|>{step}<|act|>{ACTION_OF_ROLE[other.role]}({x},{y})")
        parts.append("<|obs|>no visible change")
        parts.append(f"<|check|>objective not satisfied: {other.label} is not {target.label}")
        parts.append(f"<|retry|>target the {target.role} labelled {target.label}, not {other.label}")
    # the invariant is strict on purpose: a retry must always be followed by an
    # attempt, so the last thing before the budget wall is one more real try
    nxt = rng.choice(layout.elements)
    x, y = _coords(nxt, rng)
    parts.append(f"<|step|>{rng.randint(4, 6)}<|act|>{ACTION_OF_ROLE[nxt.role]}({x},{y})")
    parts.append("<|obs|>no visible change")
    parts.append("<|check|>objective not satisfied: still the wrong element")
    parts.append("<|budget|>steps exhausted")
    parts.append("<|done|>failed: retries exhausted, objective not reached")
    return "".join(parts)


def impossible_episode(rng: random.Random) -> str:
    """Objective names something that is not on this screen. Honest answer:
    say impossible, do not click around and do not claim success."""
    layout = make_layout(rng)
    present = {e.label for e in layout.elements}
    missing = [l for l in LABELS if l not in present]
    label = rng.choice(missing)
    role = rng.choice(ROLES)
    task = rng.choice(TASKS[role]).format(label=label)
    parts = [f"<|drive|>objective: {task}",
             "<|scan|>elements on screen: " + ", ".join(sorted(present)),
             f"<|check|>{label} is not present on this screen",
             "<|done|>impossible: no {0} labelled {1} exists here".format(role, label)]
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
]

WHY = [
    "wrong operator", "wrong direction of the comparison", "off by one element",
    "early return skips the result", "the branch checks the wrong variable",
    "the loop mutates the list it is scanning", "the guard is inverted",
    "integer division where a float is required",
]


def _mutate(body: str, rng: random.Random) -> str:
    """A plausible wrong variant, so repeated attempts are not identical."""
    swaps = [(" + ", " - "), (" - ", " + "), (" > ", " >= "), (" < ", " <= "),
             ("max(", "min("), ("[0]", "[-1]"), ("True", "False"), ("**", "*"),
             ("len(", "sum(")]
    for a, b in rng.sample(swaps, k=len(swaps)):
        if a in body:
            return body.replace(a, b, 1)
    return body.replace("return", "return not", 1)


def code_episode(rng: random.Random) -> str:
    sig, good, bad = rng.choice(BUGS)
    name = sig.split("(")[0].replace("def ", "").strip()
    arg_src = ARG_GEN[name](rng)
    want = repr(_fn(sig, good)(*eval(f"({arg_src},)", {"__builtins__": {}})))  # noqa: S307
    assertion = f"assert {name}({arg_src}) == {want}"

    # 15% of code episodes end red: an agent that always reports green is
    # worse than one that reports the failure.
    if rng.random() < 0.15:
        parts = [f"<|task|>make the tests pass for {name}()"]
        for attempt in (1, 2):
            parts.append(f"<|try|>{attempt}")
            parts.append(f"<|code|>{sig}\n{bad}")
            parts.append("<|run|>pytest -q")
            parts.append(f"<|err|>{assertion}  ->  AssertionError")
            parts.append("<|think|>the failing case needs information the prompt does not carry")
            parts.append(f"<|retry|>attempt {attempt} failed, patch the logic and rerun")
        parts.append("<|try|>3")
        parts.append(f"<|code|>{sig}\n{_mutate(bad, rng)}")
        parts.append("<|run|>pytest -q")
        parts.append(f"<|err|>{assertion}  ->  AssertionError")
        parts.append("<|budget|>attempts exhausted")
        parts.append("<|done|>failed: tests still red, 2 attempts")
        return "".join(parts)

    attempts = rng.randint(2, 3)  # never zero failures: I3 is a hard invariant
    parts = [f"<|task|>make the tests pass for {name}()"]
    for attempt in range(1, attempts + 1):
        if attempt == attempts:
            body = good
        elif attempt == 1:
            body = bad
        else:
            body = _mutate(bad, rng)
        parts.append(f"<|try|>{attempt}")
        parts.append(f"<|code|>{sig}\n{body}")
        parts.append("<|run|>pytest -q")
        if attempt < attempts:
            parts.append(f"<|err|>{assertion}  ->  AssertionError")
            parts.append(f"<|think|>{rng.choice(WHY)}")
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
    objective = rng.choice(OBJECTIVES)(rng)
    steps = rng.sample(STEPS, k=rng.randint(4, 6))
    obstacle_fn, fix = rng.choice(OBSTACLES)
    obstacle = obstacle_fn(rng)

    parts = [f"<|drive|>objective: {objective}", "<|plan|>"]
    for i, s in enumerate(steps, 1):
        parts.append(f"{i}.{s} ")
    hit = rng.randint(2, len(steps))
    for i, s in enumerate(steps, 1):
        parts.append(f"<|step|>{i}<|act|>{s.replace(' ', '_')}()")
        if i == hit:
            parts.append(f"<|obs|>{obstacle}")
            parts.append("<|check|>objective not satisfied: blocked")
            parts.append(f"<|retry|>{fix}")
            fix_tok = "".join(c if c.isalnum() or c == "_" else "_" for c in fix.replace(" ", "_"))
            parts.append(f"<|step|>{i}b<|act|>{fix_tok}()")
            parts.append("<|obs|>step completed")
        else:
            parts.append(f"<|obs|>step completed ({rng.randint(1, 60)} items)")
    if rng.random() < 0.15:
        parts.append("<|check|>objective not satisfied: blocked by the environment")
        parts.append("<|done|>blocked: the obstacle needs credentials this machine does not hold")
    else:
        parts.append("<|check|>objective satisfied")
        parts.append("<|done|>success")
    return "".join(parts)


# -------------------------------------------------------------------- factory
def drive_corpus(n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        if r < 0.42:
            yield ui_episode(rng)
        elif r < 0.78:
            yield code_episode(rng)
        elif r < 0.94:
            yield plan_episode(rng)
        else:
            yield impossible_episode(rng)


def drive_stats(n: int = 2000, seed: int = 0) -> dict:
    """Retries per episode, terminal rate, and how the terminals split.

    The split matters more than the rate: a corpus that only ever says
    "success" teaches an unattended machine to overclaim.
    """
    eps = list(drive_corpus(n, seed))
    retries = sum(e.count("<|retry|>") for e in eps)
    done = sum(e.count("<|done|>") for e in eps)
    success = sum("<|done|>success" in e for e in eps)
    failed = sum("<|done|>failed" in e for e in eps)
    blocked = sum("<|done|>blocked" in e for e in eps)
    impossible = sum("<|done|>impossible" in e for e in eps)
    fails = sum(1 for e in eps if "<|retry|>" in e)
    return {
        "episodes": len(eps),
        "mean_retries": round(retries / len(eps), 2),
        "terminal_rate": round(done / len(eps), 3),
        "success": round(success / len(eps), 3),
        "failed": round(failed / len(eps), 3),
        "blocked": round(blocked / len(eps), 3),
        "impossible": round(impossible / len(eps), 3),
        "failure_fraction": round(fails / len(eps), 3),
        "mean_chars": round(sum(len(e) for e in eps) / len(eps), 1),
    }
