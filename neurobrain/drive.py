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
  3. the first attempt fails in >= 50% of UI/code episodes, so "failure" is
     correlated with "next attempt", never with "end"
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
    parts.append("<|obs|>no visible change")
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
BUGS = [
    # (good body, buggy body, error line)
    ("    return a + b", "    return a - b", "assert add(2, 2) == 4"),
    ("    return n * 2", "    return n + 2", "assert double(3) == 6"),
    ("    return [x for x in xs if x > 0]", "    return [x for x in xs if x < 0]",
     "assert positives([-1, 2, 3]) == [2, 3]"),
    ("    return max(items)", "    return min(items)", "assert biggest([4, 9, 2]) == 9"),
    ("    return s[::-1]", "    return s[::2]", "assert flip('abcdef') == 'fedcba'"),
    ("    return len(unique)", "    return len(items)", "assert count_unique([1,2,2,3]) == 3"),
    ("    return round(total / len(values), 2)", "    return round(total * len(values), 2)",
     "assert mean([2, 4, 6]) == 4.0"),
    ("    return text.split()[0]", "    return text.split()[-1]", "assert first_word('hi there') == 'hi'"),
    ("    return {'ok': True, 'n': len(items)}", "    return {'ok': False, 'n': len(items)}",
     "assert summarize([1,2])['ok'] is True"),
    ("    return [k for k in d if d[k] > 0]", "    return [k for k in d if d[k] < 0]",
     "assert keys_positive({'a': 1, 'b': -1}) == ['a']"),
]
SIGS = {
    "add": "def add(a, b):",
    "double": "def double(n):",
    "positives": "def positives(xs):",
    "biggest": "def biggest(items):",
    "flip": "def flip(s):",
    "count_unique": "def count_unique(items):\n    unique = set(items)",
    "mean": "def mean(values):\n    total = sum(values)",
    "first_word": "def first_word(text):",
    "summarize": "def summarize(items):",
    "keys_positive": "def keys_positive(d):",
}


def _mutate(body: str, rng: random.Random) -> str:
    """A plausible wrong variant, so repeated attempts are not identical."""
    swaps = [(" + ", " - "), (" - ", " + "), (" > ", " >= "), (" < ", " <= "),
             ("max(", "min("), ("[0]", "[-1]"), ("True", "False"), ("**", "*")]
    for a, b in rng.sample(swaps, k=len(swaps)):
        if a in body:
            return body.replace(a, b, 1)
    return body.replace("return", "return not", 1)


def code_episode(rng: random.Random) -> str:
    good, bad, assertion = rng.choice(BUGS)
    if rng.random() < 0.15:
        name = assertion.split("(")[0].replace("assert ", "")
        sig = SIGS[name]
        parts = [f"<|task|>make the tests pass for {name}()"]
        for attempt in (1, 2):
            parts.append(f"<|try|>{attempt}")
            parts.append(f"<|code|>{sig}\n{bad}")
            parts.append("<|run|>pytest -q")
            parts.append(f"<|err|>{assertion}  ->  AssertionError")
            parts.append("<|think|>the fix needs information the prompt does not have")
            parts.append(f"<|retry|>attempt {attempt} failed, patch the logic and rerun")
        parts.append("<|budget|>attempts exhausted")
        parts.append("<|done|>failed: tests still red, 2 attempts")
        return "".join(parts)
    name = SIGS and assertion.split("(")[0].replace("assert ", "")
    sig = SIGS[name]
    parts = [f"<|task|>make the tests pass for {name}()"]
    attempts = rng.randint(1, 3)
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
            why = rng.choice([
                "wrong operator", "wrong direction of the comparison",
                "off by one element", "early return skips the result",
                "the branch checks the wrong variable",
            ])
            parts.append(f"<|think|>{why}")
            parts.append(f"<|retry|>attempt {attempt} failed, patch the logic and rerun")
        else:
            parts.append("<|ok|>1 passed")
            parts.append("<|check|>objective satisfied")
    parts.append("<|done|>success")
    return "".join(parts)


# ---------------------------------------------------------------- plan episode
OBSTACLES = [
    ("file is locked by another process", "close the handle, then continue"),
    ("permission denied", "retry with the user's own directory instead"),
    ("window did not respond", "focus the window, wait, then continue"),
    ("wrong menu opened", "press escape and open the intended menu"),
    ("element scrolled out of view", "scroll to it, then click"),
]


def plan_episode(rng: random.Random) -> str:
    objective = rng.choice([
        "tidy the desktop into folders by type",
        "delete every iso file in downloads",
        "scaffold a flask app and run its tests",
        "rename screenshots to the date they were taken",
        "close every window that has been idle for an hour",
        "export the spreadsheet to csv and open the result",
    ])
    steps = rng.sample([
        "capture the screen", "list the matching files", "sort by the rule in the objective",
        "apply the change", "verify the change on screen", "write the run log",
    ], k=4)
    obstacle, fix = rng.choice(OBSTACLES)

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
            parts.append("<|obs|>step completed")
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
    return {
        "episodes": len(eps),
        "mean_retries": round(retries / len(eps), 2),
        "terminal_rate": round(done / len(eps), 3),
        "success": round(success / len(eps), 3),
        "failed": round(failed / len(eps), 3),
        "blocked": round(blocked / len(eps), 3),
        "impossible": round(impossible / len(eps), 3),
        "mean_chars": round(sum(len(e) for e in eps) / len(eps), 1),
    }
