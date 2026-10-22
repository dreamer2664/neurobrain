"""M3 acceptance: the drive loop with its ladder on 10 scripted tasks.

Each task runs twice: with an ORACLE grounder (loop mechanics isolated) and
with the REAL GripNet grounder (the /ground code path, step-750 checkpoint).
Shell tasks need no grounder and run once per mode identically. Metrics are
the ones specs/drive.md names: terminal_rate (done/all), retry_depth (mean
max rung on non-done), recovery_rate (failed-then-done / ever-failed), plus
closed_rate (reached any terminal state) which the loop must keep at 1.0.

    python3 scripts/drive_suite.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brainbox.drive import (  # noqa: E402
    DriveLoop, OracleGrounder, RealGrounder, ScriptedPlanner, ShellWorld,
    SimulatedDesktop, TaskStore)
from brainbox.grip import Grounder  # noqa: E402
from neurobrain import ui_world  # noqa: E402


def sim_world(seed, extra_targets=0, **kw):
    ex = ui_world.sample_example(random.Random(seed))
    world = SimulatedDesktop(ex, **kw)
    targets = [ex.target]
    others = [e for e in ex.layout.elements if e is not ex.target]
    for e in others[:extra_targets]:
        targets.append(e)
        world.require(e)
    steps = [world.step_for(t) for t in targets]
    return world, steps, ex


def build_tasks(base):
    """Returns list of (name, builder) where builder(store) -> (task, world, planner)."""
    tasks = []

    def simple(seed):
        def b(store):
            world, steps, ex = sim_world(seed)
            return store.create(ex.task), world, ScriptedPlanner(steps)
        return b

    def chaos(seed, **kw):
        def b(store):
            world, steps, ex = sim_world(seed, **kw)
            return store.create(ex.task), world, ScriptedPlanner(steps)
        return b

    def wrong_plan(seed):
        def b(store):
            world, steps, ex = sim_world(seed)
            others = [e for e in world.layout.elements
                      if e is not world.required[0]]
            wrong = world.step_for(others[0])
            wrong["intent"] = steps[0]["intent"]
            planner = ScriptedPlanner([wrong], replan_script=steps)
            return store.create(ex.task), world, planner
        return b

    def two_targets(seed, second_unreachable=False):
        def b(store):
            world, steps, ex = sim_world(seed, extra_targets=1)
            if second_unreachable:
                world.unreachable_labels.add(world.required[1].label)
            return store.create(ex.task), world, ScriptedPlanner(steps)
        return b

    def shell_write(store):
        ws = os.path.join(base, "ws_write")
        checks = [{"path_exists": "report.txt"},
                  {"file_contains": {"path": "report.txt", "text": "quarterly"}}]
        world = ShellWorld(ws, checks, home=base)
        planner = ScriptedPlanner(
            [{"intent": "create the report", "ground": "click the new button",
              "action": {"type": "click"}}],
            shell_steps=[{"intent": "write report via shell",
                          "action": {"type": "shell", "op": "write_file",
                                     "path": "report.txt",
                                     "text": "quarterly numbers\n"}}])
        task = store.create("create report.txt", checks=checks, workspace=ws)
        return task, world, planner

    def shell_delete(store):
        ws = os.path.join(base, "ws_delete")
        os.makedirs(ws, exist_ok=True)
        for name in ("a.tmp", "b.tmp", "keep.txt"):
            with open(os.path.join(ws, name), "w") as f:
                f.write(name)
        checks = [{"path_absent": "a.tmp"}, {"path_absent": "b.tmp"},
                  {"path_exists": "keep.txt"}]
        world = ShellWorld(ws, checks, home=base)
        planner = ScriptedPlanner(
            [{"intent": "delete the tmp files",
              "ground": "click the delete button", "action": {"type": "click"}}],
            decompose_steps=[
                {"intent": "delete a.tmp",
                 "action": {"type": "shell", "op": "delete", "path": "a.tmp"}},
                {"intent": "delete b.tmp",
                 "action": {"type": "shell", "op": "delete", "path": "b.tmp"}}])
        task = store.create("delete the tmp files", checks=checks, workspace=ws)
        return task, world, planner

    def budget_starved(seed):
        def b(store):
            world, steps, ex = sim_world(seed, unreachable=True)
            task = store.create(ex.task,
                                budget={"max_steps": 3, "max_seconds": 600})
            return task, world, ScriptedPlanner(steps)
        return b

    tasks = [
        ("01-click-target", simple(21)),
        ("02-flaky-click", chaos(22, ignore_first_correct_clicks=1)),
        ("03-widget-relocates", chaos(23, relocate_on_first_act=True)),
        ("04-wrong-plan-replan", wrong_plan(24)),
        ("05-two-targets", two_targets(25)),
        ("06-second-unreachable", two_targets(26, second_unreachable=True)),
        ("07-unreachable", chaos(27, unreachable=True)),
        ("08-shell-write-toolswitch", shell_write),
        ("09-shell-delete-decompose", shell_delete),
        ("10-budget-starved", budget_starved(28)),
    ]
    return tasks


def metrics(rows):
    done = [r for r in rows if r["status"] == "done"]
    nondone = [r for r in rows if r["status"] != "done"]
    ever_failed = [r for r in rows if r["failures"] > 0]
    recovered = [r for r in ever_failed if r["status"] == "done"]
    return {
        "terminal_rate (done/all)": round(len(done) / len(rows), 3),
        "closed_rate (any terminal/all)": round(
            sum(1 for r in rows if r["status"] in ("done", "done-partial", "failed"))
            / len(rows), 3),
        "retry_depth (mean max rung, non-done)": round(
            sum(r["max_rung"] for r in nondone) / len(nondone), 2) if nondone else 0,
        "recovery_rate": round(len(recovered) / len(ever_failed), 3)
        if ever_failed else None,
        "mean_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 1),
    }


def run_mode(mode, base, real_grounder):
    # Fully separate state per mode: shared workspaces/trash between modes
    # pre-satisfy checks and collide on trash slots (found the hard way).
    mode_base = os.path.join(base, mode)
    os.makedirs(mode_base, exist_ok=True)
    store = TaskStore(home=os.path.join(mode_base, "home"))
    loop = DriveLoop(store)
    rows = []
    for name, builder in build_tasks(mode_base):
        task, world, planner = builder(store)
        grounder = None
        if isinstance(world, SimulatedDesktop):
            grounder = (OracleGrounder(world) if mode == "oracle"
                        else RealGrounder(real_grounder))
        t0 = time.time()
        out = loop.run_until_settled(task, world, planner, grounder)
        m = out["metrics"]
        rows.append({"task": name, "id": out["id"], "status": out["status"],
                     "progress": (out["result"] or {}).get("progress"),
                     "attempts": m["steps_used"], "max_rung": m["max_rung"],
                     "failures": m["failures"], "recovered": m["recovered"],
                     "wall_s": round(time.time() - t0, 2),
                     "parked_times": len(out["blocked_records"])})
        print(f"[{mode}] {name:<26} {out['status']:<13} attempts={m['steps_used']:<3} "
              f"max_rung={m['max_rung']} progress={rows[-1]['progress']}")
    idle = loop.idle_cycle()
    return {"mode": mode, "rows": rows, "metrics": metrics(rows), "idle": idle}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the sandbox state")
    args = ap.parse_args()

    base = tempfile.mkdtemp(prefix="bb_drive_suite_")
    real = Grounder.find(None)  # checkpoints/grip (M2) or legacy
    print(f"real grounder: step {real.step} from {real.source}")
    report = {"milestone": "M3", "grip_step": real.step, "base": base,
              "modes": {}}
    try:
        for mode in ("oracle", "real"):
            report["modes"][mode] = run_mode(mode, base, real)
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
            report["base"] = "(removed; rerun with --keep to inspect)"
    print(json.dumps({m: report["modes"][m]["metrics"] for m in report["modes"]},
                     indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    ok = all(report["modes"][m]["metrics"]["closed_rate (any terminal/all)"] == 1.0
             for m in report["modes"])
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
