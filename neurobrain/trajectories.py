"""Trajectory forge: multi-step tasks for the orchestrator (M5).

One JSON object per line, exactly the schema in ``specs/task_schema.md``:

    id, instruction, created_at, status, plan[], history[], result, budget

with one addition that the spec implies but does not name: a ``branches`` list
of the failure paths the task may take (wrong element, blocked dialog, test
error) each with the recovery that was actually recorded in ``history``. A
trainer can therefore ask "given this prefix, which recovery?" instead of only
"what came next".

Rules the generator enforces (and ``dataforge stats`` re-checks on the bytes):

1. every action is preceded and followed by an observation (no blind action);
2. every ``verdict`` is one of progress / no-op / wrong / fatal;
3. ``status`` ends in exactly one of done / failed (spec lifecycle);
4. histories are append-only and step numbers are non-decreasing (a retry of
   the same plan step keeps its number, which is how the spec example reads);
5. >= 50% of tasks contain at least one failure verdict and a recovery.

Deterministic: ids are ``t-<zero-padded index>``, ``created_at`` is derived
from the seed (never the clock).
"""

from __future__ import annotations

import random

from .ui_world_hard import CANVAS, sample_example

VERDICTS = ["progress", "no-op", "wrong", "fatal"]

CODE_TASKS = [
    ("add a retry with backoff to fetch()", "def fetch(url):", "return get(url)"),
    ("make the parser handle empty input", "def parse(text):", "return text.split(',')"),
    ("fix the off-by-one in the paginator", "def page(items, n):", "return items[n * 10:n * 10 + 10]"),
    ("handle missing keys in the config loader", "def load(path):", "return json.load(open(path))"),
    ("stop the upload from blocking the ui", "def upload(path):", "return post(path)"),
    ("cap the log file at ten megabytes", "def rotate(path):", "open(path, 'a')"),
    ("make the cache expire after an hour", "def get(key):", "return store[key]"),
    ("dedupe the event stream", "def events(xs):", "return [x for x in xs]"),
]
PLAN_TASKS = [
    "tidy the downloads folder by file type",
    "export every invoice from the last quarter to csv",
    "rename the screenshots by the date they were taken",
    "close idle windows and free memory",
    "scaffold a flask app and run its tests",
    "move finished projects into an archive folder",
    "sync the notes folder to the backup drive",
    "flatten the nested folders into one directory",
]
UI_DONE = ["done", "done-partial", "failed"]


VALID_ACTIONS = {"click", "type", "key", "drag", "scroll", "toggle", "wait"}
STATUSES = {"pending", "planning", "acting", "verifying", "done", "failed"}


def validate_trajectory(rec: dict) -> list[str]:
    """Strict check against specs/task_schema.md. Empty list == valid."""
    errs: list[str] = []
    for key in ("id", "instruction", "created_at", "status", "plan", "history",
                "result", "budget"):
        if key not in rec:
            errs.append(f"missing {key}")
    if errs:
        return errs
    if rec["status"] not in STATUSES:
        errs.append(f"bad status {rec['status']!r}")
    if not isinstance(rec["instruction"], str) or not rec["instruction"].strip():
        errs.append("empty instruction")
    if not isinstance(rec["created_at"], (int, float)):
        errs.append("created_at not a number")
    b = rec["budget"]
    for k in ("max_steps", "max_seconds", "max_ram_mb"):
        if k not in b or not isinstance(b[k], (int, float)) or b[k] <= 0:
            errs.append(f"budget.{k} missing or non-positive")
    if not isinstance(rec["plan"], list) or not rec["plan"]:
        errs.append("plan empty")
    for p in rec["plan"]:
        if "step" not in p or "intent" not in p or "action" not in p:
            errs.append("plan entry missing step/intent/action")
            continue
        if p["action"].get("type") not in VALID_ACTIONS:
            errs.append(f"plan action type {p['action'].get('type')!r}")
    prev = -1
    for h in rec["history"]:
        if h.get("step", -1) < prev:
            errs.append(f"history step went backwards at {h.get('step')}")
        prev = h.get("step", prev)
        if "observation_before" not in h or "observation_after" not in h:
            errs.append("history entry without observations (blind action)")
        if h.get("verdict") not in VERDICTS:
            errs.append(f"bad verdict {h.get('verdict')!r}")
        if h.get("action", {}).get("type") not in VALID_ACTIONS:
            errs.append(f"history action type {h.get('action', {}).get('type')!r}")
    r = rec["result"]
    if "success" not in r or "summary" not in r:
        errs.append("result missing success/summary")
    if rec["status"] == "done" and not r.get("success"):
        errs.append("status done but result.success false")
    return errs


FAIL_VERDICTS = ("no-op", "wrong", "fatal")


def has_failure(rec: dict) -> bool:
    """A failed attempt is any verdict that is not progress: 'no-op' means the
    action did nothing, which is exactly a miss the loop must recover from."""
    return any(h.get("verdict") in FAIL_VERDICTS for h in rec.get("history", []))


def has_recovery(rec: dict) -> bool:
    hist = rec.get("history", [])
    for i, h in enumerate(hist):
        if h.get("verdict") in FAIL_VERDICTS:
            if any(h2.get("verdict") == "progress" for h2 in hist[i + 1:]):
                return True
    return False


def _stamp(seed: int, index: int) -> float:
    """Deterministic created_at: 2026-01-01 + index seconds + seed jitter."""
    return 1767225600.0 + index * 37.0 + (seed % 1000)


# ------------------------------------------------------------------ ui tasks
def ui_task(rng: random.Random, index: int, seed: int) -> dict:
    ex = sample_example(rng)
    tgt = ex.target
    task_id = f"t-{index:06d}"
    window = f"{ex.layout.title}"

    def shot(step: int) -> dict:
        return {
            "screenshot": f"s3://local/{task_id}/{step:03d}.png",
            "window": window,
            "elements": [{"role": tgt.role, "label": tgt.label,
                          "box": [round(tgt.x, 1), round(tgt.y, 1),
                                  round(tgt.w, 1), round(tgt.h, 1)]}],
        }

    plan = [
        {"step": 1, "intent": "look at the screen and find the target",
         "action": {"type": "key", "keys": ["ctrl", "f"]}},
        {"step": 2, "intent": f"act on the {tgt.role} labelled {tgt.label}",
         "action": {"type": "click", "x": ex.click[0] * 15, "y": ex.click[1] * 15,
                    "button": "left"}},
        {"step": 3, "intent": "verify the change on screen",
         "action": {"type": "key", "keys": ["ctrl", "shift", "s"]}},
    ]

    history = []
    n_fail = rng.choices([0, 1, 2], weights=[0.25, 0.55, 0.20])[0]
    step = 0
    for f in range(n_fail):
        step += 1
        side = rng.choice(["up", "down", "left", "right"])
        dx = rng.randint(-14, 14)
        dy = rng.randint(-14, 14)
        history.append({
            "step": step,
            "observation_before": shot(step),
            "action": {"type": "click", "x": int(tgt.cx + dx) * 15,
                       "y": int(tgt.cy + dy) * 15, "button": "left"},
            "observation_after": {**shot(step + 1), "changed": False},
            "verdict": rng.choice(["no-op", "wrong"]),
            "note": f"missed {side} of the {tgt.role}, {abs(dx) + abs(dy)} px off",
        })
        history.append({
            "step": step + 1,
            "observation_before": shot(step + 1),
            "action": {"type": "click", "x": int(tgt.cx) * 15, "y": int(tgt.cy) * 15,
                       "button": "left"},
            "observation_after": {**shot(step + 2), "changed": True},
            "verdict": "progress",
            "note": "recovery: aimed at the widget centre",
        })
        step += 1

    step += 1
    history.append({
        "step": step,
        "observation_before": shot(step),
        "action": {"type": "click", "x": ex.click[0] * 15, "y": ex.click[1] * 15,
                   "button": "left"},
        "observation_after": {**shot(step + 1), "changed": True},
        "verdict": "progress",
        "note": f"{tgt.role} {tgt.label} responded",
    })

    success = rng.random() < 0.86
    return {
        "id": task_id,
        "instruction": ex.task,
        "created_at": _stamp(seed, index),
        "status": "done" if success else "failed",
        "plan": plan,
        "history": history,
        "result": {"success": success,
                   "summary": (f"{ex.action} on {tgt.label} completed" if success
                               else f"budget exhausted before {tgt.label} responded"),
                   "artifacts": []},
        "budget": {"max_steps": rng.choice([20, 40, 60]),
                   "max_seconds": rng.choice([120, 300, 600]),
                   "max_ram_mb": rng.choice([2048, 4096])},
        "branches": [
            {"when": "click lands outside the widget", "recovery": "resample inside the box",
             "verdict": "no-op"},
            {"when": "another window takes focus", "recovery": "focus the window, re-observe",
             "verdict": "wrong"},
        ],
        "source": "ui",
        "canvas": CANVAS,
    }


# ---------------------------------------------------------------- code tasks
def code_task(rng: random.Random, index: int, seed: int) -> dict:
    title, sig, body = rng.choice(CODE_TASKS)
    task_id = f"t-{index:06d}"
    step = 0
    history = []
    n_fail = rng.choices([1, 1, 2, 3], weights=[0.55, 0.2, 0.15, 0.10])[0]
    for a in range(1, n_fail + 1):
        step += 1
        history.append({
            "step": step,
            "observation_before": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                                   "window": "Editor", "elements": []},
            "action": {"type": "type", "text": f"{sig}\n    {body}  # attempt {a}"},
            "observation_after": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                                  "changed": True},
            "verdict": "progress",
            "note": f"attempt {a} written",
        })
        step += 1
        history.append({
            "step": step,
            "observation_before": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                                   "window": "Terminal", "elements": []},
            "action": {"type": "key", "keys": ["ctrl", "enter"]},
            "observation_after": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                                  "changed": True},
            "verdict": "wrong" if a < n_fail else "progress",
            "note": rng.choice(["AssertionError: expected 4, got -4",
                                "TypeError: unsupported operand",
                                "IndexError: list index out of range",
                                "1 failed, 0 passed"]),
        })
    step += 1
    history.append({
        "step": step,
        "observation_before": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                               "window": "Terminal", "elements": []},
        "action": {"type": "key", "keys": ["ctrl", "enter"]},
        "observation_after": {"screenshot": f"s3://local/{task_id}/{step:03d}.png",
                              "changed": True},
        "verdict": "progress",
        "note": "1 passed in 0.04s",
    })
    return {
        "id": task_id,
        "instruction": title,
        "created_at": _stamp(seed, index),
        "status": "done",
        "plan": [{"step": i + 1, "intent": t, "action": {"type": "key", "keys": ["ctrl", "enter"]}}
                 for i, t in enumerate(["read the failing test", "patch the function",
                                        "rerun the suite", "verify green"])],
        "history": history,
        "result": {"success": True, "summary": f"{title}: tests green after {n_fail} failures",
                   "artifacts": ["patch.diff"]},
        "budget": {"max_steps": 40, "max_seconds": 300, "max_ram_mb": 4096},
        "branches": [{"when": "test still fails after the patch",
                      "recovery": "read the first error line and patch again",
                      "verdict": "wrong"},
                     {"when": "import error before the test runs",
                      "recovery": "add the missing import", "verdict": "fatal"}],
        "source": "code",
    }


# ---------------------------------------------------------------- plan tasks
def plan_task(rng: random.Random, index: int, seed: int) -> dict:
    objective = rng.choice(PLAN_TASKS)
    task_id = f"t-{index:06d}"
    steps = rng.sample(["capture the screen", "list the matching files",
                        "apply the change", "verify on screen", "write the run log",
                        "check free space"], k=rng.randint(3, 5))
    hit = rng.randint(1, len(steps))
    history = []
    for i, s in enumerate(steps, 1):
        history.append({
            "step": i,
            "observation_before": {"screenshot": f"s3://local/{task_id}/{i:03d}.png",
                                   "window": "Desktop", "elements": []},
            "action": {"type": "key" if "screen" in s else "click",
                       "keys": ["ctrl", "shift", "p"]} if "screen" in s else
                      {"type": "click", "x": rng.randint(20, 1800), "y": rng.randint(20, 1000),
                       "button": "left"},
            "observation_after": {"screenshot": f"s3://local/{task_id}/{i:03d}.png",
                                  "changed": i != hit},
            "verdict": "wrong" if i == hit else "progress",
            "note": rng.choice(["a lock dialog appeared", "permission denied",
                                "the list was empty", "the window is not responding"])
            if i == hit else f"{s} completed",
        })
        if i == hit:
            history.append({
                "step": i,
                "observation_before": {"screenshot": f"s3://local/{task_id}/{i:03d}.png",
                                       "window": "Desktop", "elements": []},
                "action": {"type": "key", "keys": ["escape"]},
                "observation_after": {"screenshot": f"s3://local/{task_id}/{i:03d}.png",
                                      "changed": True},
                "verdict": "progress",
                "note": "replan: dismiss the obstruction, then repeat the step",
            })
    return {
        "id": task_id,
        "instruction": objective,
        "created_at": _stamp(seed, index),
        "status": "done",
        "plan": [{"step": i, "intent": s, "action": {"type": "key", "keys": ["ctrl", "enter"]}}
                 for i, s in enumerate(steps, 1)],
        "history": history,
        "result": {"success": True, "summary": f"{objective}: {len(steps)} steps, 1 replan",
                   "artifacts": ["run.log"]},
        "budget": {"max_steps": 60, "max_seconds": 600, "max_ram_mb": 4096},
        "branches": [{"when": "a step blocks on a lock or permission",
                      "recovery": "dismiss, re-observe, repeat the step",
                      "verdict": "wrong"}],
        "source": "plan",
    }


# ------------------------------------------------------------------ factory
def trajectory_corpus(n: int, seed: int = 0, mix=(0.45, 0.40, 0.15)):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        if r < mix[0]:
            yield ui_task(rng, i, seed)
        elif r < mix[0] + mix[1]:
            yield code_task(rng, i, seed)
        else:
            yield plan_task(rng, i, seed)
