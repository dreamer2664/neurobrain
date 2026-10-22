"""The drive loop (specs/drive.md): plan -> ground -> act -> observe ->
verify -> retry ladder -> terminal record. Nothing exits silently.

Terminal guarantee: every settled task ends `done`, `done-partial` or
`failed`. The ladder per step (each rung tried twice before moving on):

    1 retry exact   2 retry shifted (re-ground)   3 replan
    4 decompose     5 tool switch (shell path)    6 escalate (park + record)

Parked tasks are resumed once by `run_until_settled` (the "next idle
window"); if they park again they are closed terminally by measured
progress. Budgets (steps/seconds) are hard limits checked before every
attempt. History is persisted after every attempt, so a crash mid-task
leaves a readable trace (M6 builds on this).

Worlds are pluggable. `SimulatedDesktop` is a reactive ui_world desktop
(verification = world state delta, as in scripts/e2e_demo.py); `ShellWorld`
executes file operations in a workspace with checks, deletes going through
brainbox.trash. Grounders are pluggable too: `OracleGrounder` (loop
mechanics, labelled as such) or `RealGrounder` (the GripNet /ground path).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import time

from .governor import Governor, GovernorRefused
from .trash import brainbox_home, trash_path

DEFAULT_BUDGET = {"max_steps": 60, "max_seconds": 600, "max_ram_mb": 4096}
TERMINAL = ("done", "done-partial", "failed")


# --------------------------------------------------------------------- store
def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


class TaskStore:
    """JSON task files under <brainbox home>/tasks, run records under runs/."""

    def __init__(self, home: str | None = None):
        self.home = brainbox_home(home)
        self.tasks_dir = os.path.join(self.home, "tasks")
        self.runs_dir = os.path.join(self.home, "runs")
        os.makedirs(self.tasks_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)

    def _path(self, task_id: str) -> str:
        return os.path.join(self.tasks_dir, f"{task_id}.json")

    def create(self, instruction: str, budget: dict | None = None,
               checks: list | None = None, workspace: str | None = None) -> dict:
        n = 1
        existing = self.list()
        if existing:
            n = max(int(t["id"].split("-")[1]) for t in existing
                    if t["id"].split("-")[1].isdigit()) + 1
        task = {
            "id": f"t-{n:04d}", "instruction": instruction,
            "created_at": time.time(), "status": "pending",
            "plan": [], "plan_index": 0, "history": [], "result": None,
            "budget": {**DEFAULT_BUDGET, **(budget or {})},
            "checks": checks or [], "workspace": workspace,
            "blocked_records": [],
            "metrics": {"steps_used": 0, "max_rung": 0, "failures": 0,
                        "recovered": False},
        }
        self.save(task)
        return task

    def save(self, task: dict) -> None:
        _atomic_write(self._path(task["id"]), task)

    def load(self, task_id: str) -> dict:
        with open(self._path(task_id)) as f:
            return json.load(f)

    def list(self) -> list[dict]:
        out = []
        if os.path.isdir(self.tasks_dir):
            for name in sorted(os.listdir(self.tasks_dir)):
                if name.endswith(".json"):
                    try:
                        with open(os.path.join(self.tasks_dir, name)) as f:
                            out.append(json.load(f))
                    except Exception:
                        continue
        return out

    def write_run_record(self, task: dict) -> str:
        rec = {**task, "closed_at": time.time(),
               "last_five_actions": [h.get("action") for h in task["history"][-5:]]}
        path = os.path.join(self.runs_dir, f"{task['id']}.json")
        _atomic_write(path, rec)
        return path


# ------------------------------------------------------------------ planners
class ScriptedPlanner:
    """Deterministic planner for scripted tasks (M3 acceptance, tests)."""

    def __init__(self, script, replan_script=None, shell_steps=None,
                 decompose_steps=None):
        self.script = script
        self.replan_script = replan_script
        self.shell_steps = shell_steps
        self.decompose_steps = decompose_steps
        self._replanned = False
        self._decomposed = False

    def plan(self, task):
        return json.loads(json.dumps(self.script))

    def replan(self, task, evidence):
        if self.replan_script is None or self._replanned:
            return None
        self._replanned = True
        return json.loads(json.dumps(self.replan_script))

    def decompose(self, step, task):
        if self.decompose_steps is None or self._decomposed:
            return None
        self._decomposed = True
        return json.loads(json.dumps(self.decompose_steps))

    def tool_switch(self, step, task):
        if self.shell_steps is None:
            return None
        return json.loads(json.dumps(self.shell_steps))


class LMPlanner:
    """Plans with the LM when possible; falls back to grounding the raw
    instruction. The fallback is deliberate: a 4M model's free text often
    contains no parseable action, and the loop must not stall on parsing."""

    def __init__(self, host):
        self.host = host

    def plan(self, task):
        instruction = task["instruction"]
        steps = []
        try:
            out = self.host.generate(
                f"task: {instruction}\nplan:\nclick the ",
                max_tokens=48, temperature=0.7)
            task["planner_raw"] = out["text"]
            import re

            for m in re.finditer(r"click the ([a-z0-9 -]{2,40})", out["text"].lower()):
                what = m.group(1).strip()
                steps.append({"intent": f"click the {what}",
                              "ground": f"click the {what}",
                              "action": {"type": "click"}})
                if len(steps) >= 4:
                    break
        except Exception as e:
            task["planner_raw"] = f"<lm planner failed: {e}>"
        if not steps:
            steps = [{"intent": instruction, "ground": instruction,
                      "action": {"type": "click"}}]
        return steps

    def replan(self, task, evidence):
        return None

    def decompose(self, step, task):
        return None

    def tool_switch(self, step, task):
        return None


# ----------------------------------------------------------------- grounders
class OracleGrounder:
    """Returns the scripted target's true centre. Loop mechanics only —
    always labelled, never presented as perception."""

    name = "oracle"

    def __init__(self, world):
        self.world = world

    def locate(self, world, step, obs):
        return world.element_center(step.get("target") or {})


class RealGrounder:
    """GripNet via brainbox.grip (the same code path as POST /ground)."""

    name = "gripnet"

    def __init__(self, grounder):
        self.grounder = grounder

    def locate(self, world, step, obs):
        frame = world.frame()
        if frame is None:
            return None
        import numpy as np
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
        r = self.grounder.ground(buf.getvalue(), step.get("ground") or step["intent"])
        return r["x"], r["y"]


# -------------------------------------------------------------------- worlds
class SimulatedDesktop:
    """Reactive ui_world desktop. Clicking a widget activates it (drawn as a
    marker); verification compares activated sets before/after the action.
    Chaos knobs reproduce the failure classes the ladder exists for."""

    name = "sim"

    def __init__(self, example, ignore_first_correct_clicks: int = 0,
                 relocate_on_first_act: bool = False, unreachable: bool = False):
        self.example = example
        self.layout = example.layout
        self.required = [example.target]
        self.activated: set[int] = set()
        self.ignore_left = ignore_first_correct_clicks
        self.relocate_pending = relocate_on_first_act
        self.unreachable = unreachable
        self.unreachable_labels: set[str] = set()

    # helpers ---------------------------------------------------------------
    def _idx(self, element) -> int:
        return self.layout.elements.index(element)

    def require(self, element) -> None:
        if element not in self.required:
            self.required.append(element)

    def step_for(self, element) -> dict:
        from neurobrain.ui_world import TASKS

        text = TASKS[element.role][0].format(label=element.label)
        return {"intent": text, "ground": text, "action": {"type": "click"},
                "target": {"label": element.label, "role": element.role,
                           "index": self._idx(element)}}

    def element_center(self, target: dict):
        idx = target.get("index")
        if idx is None or not (0 <= idx < len(self.layout.elements)):
            for e in self.layout.elements:
                if e.label == target.get("label") and e.role == target.get("role"):
                    return e.cx, e.cy
            return None
        e = self.layout.elements[idx]
        return e.cx, e.cy

    # world protocol ------------------------------------------------------------
    def frame(self):
        import numpy as np
        from PIL import Image, ImageDraw

        from neurobrain.ui_world import render

        img = Image.fromarray(np.asarray(render(self.layout), dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        for idx in self.activated:
            e = self.layout.elements[idx]
            draw.rectangle([e.x, e.y, e.x + e.w - 1, e.y + e.h - 1],
                           outline=(255, 255, 255), width=2)
        return np.asarray(img)

    def observe(self, task=None) -> dict:
        # centres are part of the observation on purpose: a real desktop agent
        # gets bounding boxes from the accessibility tree, and the LM policy
        # needs the same coordinate space the text protocol uses
        return {"kind": "sim", "window": getattr(self.layout, "title", "sim"),
                "activated": sorted(self.layout.elements[i].label
                                    for i in self.activated),
                "elements": [{"label": e.label, "role": e.role,
                              "cx": e.cx, "cy": e.cy,
                              "x": e.x, "y": e.y, "w": e.w, "h": e.h}
                             for e in self.layout.elements]}

    def act(self, action, point, task=None) -> dict:
        if (action or {}).get("type") != "click":
            return {"ok": False, "detail": f"sim world only clicks, got {action!r}"}
        if point is None:
            return {"ok": False, "detail": "no point grounded"}
        if self.relocate_pending:  # UI reflows between screenshot and click
            self.relocate_pending = False
            t = self.required[0]
            others = [e for e in self.layout.elements if e is not t]
            if others:
                o = others[0]
                t.x, o.x = o.x, t.x
                t.y, o.y = o.y, t.y
        px, py = point
        hit = None
        for e in self.layout.elements:
            if e.contains(px, py):
                hit = e
        if hit is None:
            return {"ok": True, "detail": "clicked empty desktop"}
        if self.unreachable or hit.label in self.unreachable_labels:
            return {"ok": True, "detail": f"{hit.label} did not react (disabled)"}
        if hit in self.required and self.ignore_left > 0:
            self.ignore_left -= 1
            return {"ok": True, "detail": f"{hit.label} ignored the click (flaky)"}
        self.activated.add(self._idx(hit))
        return {"ok": True, "detail": f"activated {hit.label}"}

    def verify(self, before, after, step, act_result):
        # The verifier judges against the task's required elements (ground
        # truth), never against where the planner *believed* the target was.
        new = set(after["activated"]) - set(before["activated"])
        want = {e.label for e in self.required}
        if new & want:
            return "progress", f"activated required {sorted(new & want)}"
        if new:
            return "wrong", f"activated {sorted(new)}, none of them required"
        return "no-op", act_result.get("detail", "nothing changed")

    def progress(self) -> float:
        got = sum(1 for e in self.required if self._idx(e) in self.activated)
        return got / max(1, len(self.required))

    def snapshot(self) -> dict:
        return {"activated": sorted(self.activated),
                "positions": [[e.x, e.y] for e in self.layout.elements],
                "ignore_left": self.ignore_left,
                "relocate_pending": self.relocate_pending,
                "required_idx": [self._idx(e) for e in self.required],
                "unreachable": self.unreachable,
                "unreachable_labels": sorted(self.unreachable_labels)}

    def restore(self, snap: dict) -> None:
        if not snap:
            return
        self.activated = set(snap.get("activated", []))
        for e, (x, y) in zip(self.layout.elements, snap.get("positions", [])):
            e.x, e.y = x, y
        self.ignore_left = snap.get("ignore_left", self.ignore_left)
        self.relocate_pending = snap.get("relocate_pending", False)
        idxs = snap.get("required_idx")
        if idxs:
            self.required = [self.layout.elements[i] for i in idxs]
        self.unreachable = snap.get("unreachable", self.unreachable)
        self.unreachable_labels = set(snap.get("unreachable_labels", []))


class ShellWorld:
    """File operations in a workspace, verified against declarative checks.
    Deletes go through brainbox.trash (undoable), enforced here."""

    name = "shell"

    def __init__(self, workspace: str, checks: list, home: str | None = None):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.checks = checks
        self.home = home
        self.task_id = None

    def frame(self):
        return None

    def _safe(self, rel: str) -> str:
        p = os.path.abspath(os.path.join(self.workspace, rel))
        if not p.startswith(self.workspace + os.sep) and p != self.workspace:
            raise ValueError(f"path escapes workspace: {rel!r}")
        return p

    def _check(self, c: dict) -> bool:
        if "path_exists" in c:
            return os.path.exists(self._safe(c["path_exists"]))
        if "path_absent" in c:
            return not os.path.exists(self._safe(c["path_absent"]))
        if "file_contains" in c:
            spec = c["file_contains"]
            p = self._safe(spec["path"])
            if not os.path.exists(p):
                return False
            with open(p) as fh:
                return spec["text"] in fh.read()
        return False

    def checks_state(self) -> list[bool]:
        return [self._check(c) for c in self.checks]

    def observe(self, task=None) -> dict:
        files = {}
        for root, dirs, names in os.walk(self.workspace):
            dirs.sort()
            for name in sorted(names):
                fp = os.path.join(root, name)
                rel = os.path.relpath(fp, self.workspace)
                with open(fp, "rb") as fh:
                    files[rel] = hashlib.sha256(fh.read()).hexdigest()[:12]
        return {"kind": "shell", "workspace": self.workspace, "files": files,
                "checks": self.checks_state()}

    def element_center(self, target):  # shell world has no geometry
        return None

    def act(self, action, point, task=None) -> dict:
        a = action or {}
        if a.get("type") != "shell":
            return {"ok": False, "detail": "shell world needs shell actions"}
        op = a.get("op")
        try:
            if op == "write_file":
                if a.get("delay_ms"):
                    import time as _time

                    _time.sleep(int(a["delay_ms"]) / 1000)  # slow FS/device
                p = self._safe(a["path"])
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w") as f:
                    f.write(a.get("text", ""))
                return {"ok": True, "detail": f"wrote {a['path']}"}
            if op == "delete":
                p = self._safe(a["path"])
                trash_path(self.task_id or "shell", p, home=self.home)
                return {"ok": True, "detail": f"trashed {a['path']} (undoable)"}
            if op == "mkdir":
                os.makedirs(self._safe(a["path"]), exist_ok=True)
                return {"ok": True, "detail": f"mkdir {a['path']}"}
            if op == "move":
                shutil.move(self._safe(a["src"]), self._safe(a["dst"]))
                return {"ok": True, "detail": f"moved {a['src']} -> {a['dst']}"}
        except Exception as e:
            return {"ok": False, "detail": f"{op} failed: {e}"}
        return {"ok": False, "detail": f"unknown shell op {op!r}"}

    def verify(self, before, after, step, act_result):
        b = sum(before["checks"])
        a = sum(after["checks"])
        if a > b:
            return "progress", f"checks {b}->{a}"
        if a < b:
            return "wrong", f"checks regressed {b}->{a}"
        if not act_result.get("ok"):
            return "no-op", act_result.get("detail", "act failed")
        if before.get("files") != after.get("files"):
            return "progress", "workspace state changed toward the step intent"
        return "no-op", "no check changed"

    def progress(self) -> float:
        state = self.checks_state()
        return sum(state) / max(1, len(state)) if state else 0.0

    def snapshot(self) -> dict:
        return {}  # the workspace filesystem is the state

    def restore(self, snap: dict) -> None:
        pass


def build_scripted_run(store: TaskStore, instruction: str, spec: dict,
                       budget: dict | None = None, inbox_file: str | None = None):
    """(task, world, planner, grounder) from a script/inbox JSON spec."""
    planner = ScriptedPlanner(spec.get("steps", []),
                              replan_script=spec.get("replan_steps"),
                              shell_steps=spec.get("shell_steps"),
                              decompose_steps=spec.get("decompose_steps"))
    task = store.create(instruction, budget=budget,
                        checks=spec.get("checks"), workspace=spec.get("workspace"))
    task["spec"] = spec
    if inbox_file:
        task["inbox_file"] = inbox_file
    if spec.get("world", "shell") == "shell":
        world = ShellWorld(spec["workspace"], spec.get("checks", []),
                           home=store.home)
        grounder = None
        task["world"] = "shell workspace"
    else:
        import random as _random

        from neurobrain import ui_world

        ex = ui_world.sample_example(_random.Random(spec.get("seed", 0)))
        world = SimulatedDesktop(ex, **spec.get("world_options", {}))
        for s in spec.get("steps", []):
            t = s.get("target") or {}
            for e in world.layout.elements:
                if e.label == t.get("label") and e.role == t.get("role"):
                    world.require(e)
        grounder = OracleGrounder(world)
        task["world"] = "simulated (scripted, oracle grounder)"
    store.save(task)
    return task, world, planner, grounder


# ---------------------------------------------------------------------- loop
class DriveLoop:
    def __init__(self, store: TaskStore, governor: Governor | None = None,
                 rng_seed: int = 0):
        self.store = store
        self.governor = governor or Governor()
        self.rng = random.Random(rng_seed)

    # -- public ---------------------------------------------------------------
    def run_until_settled(self, task, world, planner, grounder=None) -> dict:
        """Run, resume parked once (next idle window), then close terminally."""
        if hasattr(world, "task_id"):
            world.task_id = task["id"]
        self._run(task, world, planner, grounder)
        if task["status"] == "parked":
            self._run(task, world, planner, grounder, resume=True)
        if task["status"] not in TERMINAL:
            self._close(task, world)
        return task

    def idle_cycle(self) -> dict:
        """Never-idle maintenance pass: verify last 10 done tasks that carry
        filesystem checks, prune old screenshots, self-check. Recorded."""
        report = {"verified_ok": 0, "verified_drift": 0, "verified_skipped": 0,
                  "pruned_files": 0}
        done = [t for t in self.store.list() if t["status"] == "done"][-10:]
        for t in done:
            if t.get("checks") and t.get("workspace"):
                w = ShellWorld(t["workspace"], t["checks"],
                               home=os.path.dirname(self.store.tasks_dir))
                if w.progress() >= 1.0:
                    report["verified_ok"] += 1
                else:
                    report["verified_drift"] += 1
            else:
                report["verified_skipped"] += 1
        cutoff = time.time() - 7 * 86400
        for name in os.listdir(self.store.runs_dir):
            d = os.path.join(self.store.runs_dir, name)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    p = os.path.join(d, f)
                    if f.endswith(".png") and os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        report["pruned_files"] += 1
        report["governor_ok"] = self.governor.can_work()
        _atomic_write(os.path.join(self.store.runs_dir, "idle_last.json"), report)
        return report

    # -- internals ---------------------------------------------------------------
    def _close(self, task, world) -> None:
        progress = world.progress()
        if progress >= 1.0:
            status, success = "done", True
        elif progress > 0:
            status, success = "done-partial", False
        else:
            status, success = "failed", False
        task["status"] = status
        task["result"] = {"success": success, "progress": progress,
                          "summary": f"{task['metrics']['steps_used']} attempts, "
                                     f"max rung {task['metrics']['max_rung']}, "
                                     f"progress {progress:.2f}",
                          "artifacts": []}
        task["metrics"]["recovered"] = bool(
            task["metrics"]["failures"] > 0 and status == "done")
        self.store.save(task)
        self.store.write_run_record(task)

    def _park(self, task, step_idx, reason, note) -> None:
        task["status"] = "parked"
        task["blocked_records"].append(
            {"at": time.time(), "step": step_idx + 1, "reason": reason,
             "note": note,
             "last_verdicts": [h["verdict"] for h in task["history"][-5:]]})
        self.store.save(task)

    def _run(self, task, world, planner, grounder, resume: bool = False) -> None:
        started = time.time()
        budget = task["budget"]
        if not task["plan"]:
            task["status"] = "planning"
            plan = planner.plan(task)
            task["plan"] = [{"step": i + 1, "intent": s.get("intent", ""),
                             "ground": s.get("ground", s.get("intent", "")),
                             "target": s.get("target"),
                             "action": s.get("action", {"type": "click"})}
                            for i, s in enumerate(plan)]
            task["plan_index"] = 0
        task["status"] = "acting"
        if task.get("world_state") and hasattr(world, "restore"):
            world.restore(task["world_state"])
        self.store.save(task)

        while task["plan_index"] < len(task["plan"]):
            idx = task["plan_index"]
            step = task["plan"][idx]
            outcome = self._run_step(task, world, planner, grounder, idx,
                                     started)
            if outcome == "done":
                task["plan_index"] += 1
                self.store.save(task)
            elif outcome == "replanned":
                continue  # plan list was replaced for the remaining steps
            elif outcome == "decomposed":
                continue  # substeps inserted at idx
            elif outcome in ("parked", "budget"):
                if outcome == "budget":
                    self._close(task, world)
                return  # parked: resume/close handled by caller
        self._close(task, world)

    def _attempt(self, task, world, grounder, idx, rung, point, started):
        """One observe->act->verify cycle. Returns verdict or 'budget'."""
        m = task["metrics"]
        if m["steps_used"] >= task["budget"]["max_steps"] or \
                time.time() - started > task["budget"]["max_seconds"]:
            return "budget"
        try:
            self.governor.check_or_raise()
        except GovernorRefused:
            raise
        step = task["plan"][idx]
        action = dict(step.get("action") or {"type": "click"})
        before = world.observe(task)
        act_point = point
        if action.get("type") == "click":
            if act_point is None and grounder is not None:
                act_point = grounder.locate(world, step, before)
        else:
            act_point = None
        result = world.act(action, act_point, task)
        after = world.observe(task)
        task["status"] = "verifying"
        verdict, note = world.verify(before, after, step, result)
        task["status"] = "acting"
        m["steps_used"] += 1
        if verdict != "progress":
            m["failures"] += 1
        obs_after = dict(after)
        frame = world.frame() if hasattr(world, "frame") else None
        if frame is not None:
            rel = self._save_frame(task, frame, idx, rung, m["steps_used"])
            obs_after["screenshot"] = rel
        if hasattr(world, "snapshot"):
            task["world_state"] = world.snapshot()
        task["history"].append({
            "step": idx + 1, "rung": rung, "intent": step.get("intent", ""),
            "observation_before": before,
            "action": {**action,
                       **({"x": act_point[0], "y": act_point[1]}
                          if act_point else {})},
            "observation_after": obs_after,
            "verdict": verdict, "note": note or result.get("detail", ""),
        })
        self.store.save(task)  # crash-safe trace
        return verdict

    def _save_frame(self, task, frame, idx, rung, n) -> str:
        import numpy as np
        from PIL import Image

        d = os.path.join(self.store.runs_dir, task["id"])
        os.makedirs(d, exist_ok=True)
        rel = f"{task['id']}/step{idx + 1}_rung{rung}_{n:03d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
            os.path.join(self.store.runs_dir, rel))
        return rel

    def _run_step(self, task, world, planner, grounder, idx, started) -> str:
        m = task["metrics"]

        def go(rung, point=None):
            m["max_rung"] = max(m["max_rung"], rung)
            try:
                return self._attempt(task, world, grounder, idx, rung, point,
                                     started)
            except GovernorRefused:
                return "ram"

        # plain attempt: ground once, keep the point for exact retries
        step = task["plan"][idx]
        point = None
        if (step.get("action") or {}).get("type", "click") == "click" \
                and grounder is not None:
            point = grounder.locate(world, step, world.observe(task))
        v = go(0, point)
        if v == "progress":
            return "done"
        if v in ("budget", "ram", "fatal"):
            return self._stop(task, idx, v)

        for _ in range(2):  # rung 1: retry exact
            v = go(1, point)
            if v == "progress":
                return "done"
            if v in ("budget", "ram", "fatal"):
                return self._stop(task, idx, v)

        for _ in range(2):  # rung 2: re-ground + jitter inside the target area
            p = None
            if grounder is not None and \
                    (step.get("action") or {}).get("type", "click") == "click":
                p = grounder.locate(world, step, world.observe(task))
                if p is not None:
                    p = (p[0] + self.rng.randint(-3, 3),
                         p[1] + self.rng.randint(-3, 3))
            v = go(2, p)
            if v == "progress":
                return "done"
            if v in ("budget", "ram", "fatal"):
                return self._stop(task, idx, v)

        # rung 3: replan (remaining steps replaced)
        m["max_rung"] = max(m["max_rung"], 3)
        remaining = planner.replan(task, task["history"][-5:])
        if remaining:
            cur = task["plan"][:idx]
            new = [{"step": i + 1, "intent": s.get("intent", ""),
                    "ground": s.get("ground", s.get("intent", "")),
                    "target": s.get("target"),
                    "action": s.get("action", {"type": "click"})}
                   for i, s in enumerate(cur + remaining)]
            task["plan"] = new
            self.store.save(task)
            return "replanned"

        # rung 4: decompose this step into substeps
        m["max_rung"] = max(m["max_rung"], 4)
        subs = planner.decompose(step, task)
        if subs:
            ins = [{"step": 0, "intent": s.get("intent", ""),
                    "ground": s.get("ground", s.get("intent", "")),
                    "target": s.get("target"),
                    "action": s.get("action", {"type": "click"})}
                   for s in subs]
            task["plan"][idx:idx + 1] = ins
            for i, s in enumerate(task["plan"]):
                s["step"] = i + 1
            self.store.save(task)
            return "decomposed"

        # rung 5: tool switch (shell path)
        m["max_rung"] = max(m["max_rung"], 5)
        shell_steps = planner.tool_switch(step, task)
        if shell_steps:
            for ss in shell_steps:
                sub = {"intent": ss.get("intent", ""),
                       "ground": ss.get("ground", ""),
                       "target": ss.get("target"),
                       "action": ss.get("action")}
                saved = task["plan"][idx]
                task["plan"][idx] = sub
                v = go(5)
                task["plan"][idx] = saved
                if v == "progress":
                    return "done"
                if v in ("budget", "ram", "fatal"):
                    return self._stop(task, idx, v)

        # rung 6: escalate — park with a blocked record, never drop silently
        m["max_rung"] = max(m["max_rung"], 6)
        self._park(task, idx, "ladder-exhausted",
                   f"step {idx + 1} ({step.get('intent', '')}) failed rungs 0-5")
        return "parked"

    def _stop(self, task, idx, why) -> str:
        if why == "ram":
            self._park(task, idx, "ram-governor", "RSS over governor cap")
            return "parked"
        if why == "fatal":
            self._park(task, idx, "fatal-verdict", "verifier reported fatal")
            return "parked"
        return "budget"
