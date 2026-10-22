"""M8: 20-task end-to-end suite, specs/eval_suite.md shape.

PROVENANCE: the milestone says "from agent 03"; as of 2026-10-22 their
60-task Xvfb suite is still pending (their M7), so this suite is authored by
agent 01 against the same spec (ids/categories/success-as-code/scoreboard)
to avoid blocking. It runs the full runtime stack — real GripNet grounding,
drive-loop ladder, shell worlds on a real filesystem, LM in the loop for
the code tasks — but on brainbox's simulated desktop rather than a real X
display (this sandbox has none; the coordinator's scripts/e2e_demo.py
covers real-pixel grounding: 0.55 hit there). Swap in 03's suite when it
lands; the scoreboard format is the spec's.

    python3 scripts/e2e_suite.py --out scoreboard.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brainbox.drive import (  # noqa: E402
    DriveLoop, LMPlanner, RealGrounder, ScriptedPlanner, ShellWorld,
    SimulatedDesktop, TaskStore)
from brainbox.grip import Grounder  # noqa: E402
from brainbox.host import BrainHost  # noqa: E402
from neurobrain import ui_world  # noqa: E402
from scripts.drive_suite import sim_world  # noqa: E402

CATEGORIES = ("file_manager", "editor", "browser", "settings", "multi_app",
              "recovery", "code")


def gui(seed, extra=0, wrong=False, **kw):
    def make(store, base, ctx):
        world, steps, ex = sim_world(seed, extra_targets=extra, **kw)
        planner_steps, replan = steps, None
        if wrong:
            others = [e for e in world.layout.elements
                      if e is not world.required[0]]
            w = world.step_for(others[0])
            w["intent"] = steps[0]["intent"]
            planner_steps, replan = [w], steps
        task = store.create(ex.task)
        return task, world, ScriptedPlanner(planner_steps, replan_script=replan), \
            RealGrounder(ctx["grip"]), None
    return make


def gui_unreachable_second(seed):
    def make(store, base, ctx):
        world, steps, ex = sim_world(seed, extra_targets=1)
        world.unreachable_labels.add(world.required[1].label)
        task = store.create(ex.task)
        return task, world, ScriptedPlanner(steps), RealGrounder(ctx["grip"]), None
    return make


def shell(setup):
    def make(store, base, ctx):
        ws = os.path.join(base, "ws_" + setup["name"])
        os.makedirs(ws, exist_ok=True)
        for name, text in setup.get("seed_files", {}).items():
            p = os.path.join(ws, name)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
        world = ShellWorld(ws, setup["checks"], home=base)
        planner = ScriptedPlanner(setup["steps"],
                                  shell_steps=setup.get("shell_steps"),
                                  decompose_steps=setup.get("decompose_steps"))
        task = store.create(setup["instruction"], checks=setup["checks"],
                            workspace=ws)
        return task, world, planner, None, setup.get("post")
    return make


def lm_code(store, base, ctx):
    task = store.create("complete the add function")
    return task, None, None, None, "lm"


def lm_planner_gui(store, base, ctx):
    world, steps, ex = sim_world(51)
    task = store.create(ex.task)
    return task, world, LMPlanner(ctx["lm"]), RealGrounder(ctx["grip"]), None


GUI_STEP = {"intent": "use the app UI", "ground": "click the button",
            "action": {"type": "click"}}

SUITE = [
    ("ed-001", "editor", gui(31)),
    ("st-001", "settings", gui(32)),
    ("br-001", "browser", gui(33)),
    ("fm-001", "file_manager", gui(34)),
    ("rc-001", "recovery", gui(35, ignore_first_correct_clicks=1)),
    ("rc-002", "recovery", gui(36, relocate_on_first_act=True)),
    ("rc-003", "recovery", gui(37, wrong=True)),
    ("ma-001", "multi_app", gui(38, extra=1)),
    ("ma-002", "multi_app", gui(39, extra=1)),
    ("st-002", "settings", gui(40, ignore_first_correct_clicks=2)),
    ("st-003", "settings", gui_unreachable_second(41)),
    ("br-002", "browser", gui(42, relocate_on_first_act=True)),
    ("fm-002", "file_manager", shell({
        "name": "report", "instruction": "create report.txt with the quarterly text",
        "checks": [{"file_contains": {"path": "report.txt", "text": "quarterly"}}],
        "steps": [GUI_STEP],
        "shell_steps": [{"intent": "write report",
                         "action": {"type": "shell", "op": "write_file",
                                    "path": "report.txt", "text": "quarterly\n"}}]})),
    ("fm-003", "file_manager", shell({
        "name": "downloads", "instruction": "delete every .iso file in Downloads",
        "seed_files": {"a.iso": "A", "b.iso": "B", "c.iso": "C",
                       "keep.txt": "K", "keep.zip": "Z"},
        "checks": [{"path_absent": "a.iso"}, {"path_absent": "b.iso"},
                   {"path_absent": "c.iso"}, {"path_exists": "keep.txt"},
                   {"path_exists": "keep.zip"}],
        "steps": [GUI_STEP],
        "decompose_steps": [
            {"intent": f"delete {n}", "action": {"type": "shell", "op": "delete",
                                                 "path": n}}
            for n in ("a.iso", "b.iso", "c.iso")],
        "post": "fm003"})),
    ("fm-004", "file_manager", shell({
        "name": "rename", "instruction": "move draft.txt to final/report.txt",
        "seed_files": {"draft.txt": "body"},
        "checks": [{"path_exists": "final/report.txt"}, {"path_absent": "draft.txt"}],
        "steps": [{"intent": "move it", "action": {"type": "shell", "op": "move",
                                                   "src": "draft.txt",
                                                   "dst": "final/report.txt"}}]})),
    ("ed-002", "editor", shell({
        "name": "proj", "instruction": "scaffold the project files",
        "checks": [{"path_exists": "proj/main.py"}, {"path_exists": "proj/README.md"}],
        "steps": [GUI_STEP],
        "decompose_steps": [
            {"intent": "main.py", "action": {"type": "shell", "op": "write_file",
                                             "path": "proj/main.py", "text": "print('x')\n"}},
            {"intent": "readme", "action": {"type": "shell", "op": "write_file",
                                            "path": "proj/README.md", "text": "# proj\n"}}]})),
    ("co-001", "code", shell({
        "name": "addfn", "instruction": "write an add function to add.py",
        "checks": [{"file_contains": {"path": "add.py", "text": "def add"}}],
        "steps": [{"intent": "write add.py",
                   "action": {"type": "shell", "op": "write_file", "path": "add.py",
                              "text": "def add(a, b):\n    return a + b\n"}}],
        "post": "co001"})),
    ("rc-004", "recovery", shell({
        "name": "deltwice", "instruction": "delete a.tmp and missing.tmp",
        "seed_files": {"a.tmp": "t"},
        "checks": [{"path_absent": "a.tmp"}, {"path_absent": "missing.tmp"}],
        "steps": [GUI_STEP],
        "decompose_steps": [
            {"intent": "delete a.tmp", "action": {"type": "shell", "op": "delete",
                                                  "path": "a.tmp"}},
            {"intent": "delete missing.tmp", "action": {"type": "shell",
                                                         "op": "delete",
                                                         "path": "missing.tmp"}}]})),
    ("co-002", "code", lm_code),
    ("co-003", "code", lm_planner_gui),
]

assert len(SUITE) == 20, len(SUITE)


def post_check(tag, out, base, ctx):
    """Extra success-as-code checks the loop's world checks cannot express."""
    if tag == "co001":
        p = os.path.join(base, "ws_addfn", "add.py")
        try:
            src = open(p).read()
            tree = ast.parse(src)
            ns = {}
            exec(compile(tree, p, "exec"), ns)  # suite's own generated fixture
            return ns["add"](2, 3) == 5, "add(2,3)==5"
        except Exception as e:
            return False, f"exec failed: {e}"
    if tag == "fm003":
        from brainbox.trash import list_trash

        held = list_trash(home=base)
        return any(t["task_id"] == out["id"] and t["entries"] == 3
                   for t in held), "3 isos held in trash (undoable)"
    return True, ""


def run_lm_code_task(task, ctx):
    from brainbox.drive import _atomic_write  # noqa: F401  (keep import surface)

    out_gen = ctx["lm"].generate("def add(a, b):\n", max_tokens=32,
                                  temperature=0.7, seed=0)
    text = out_gen["text"]
    ok, why = False, ""
    try:
        ast.parse(text)
        ok = "return" in text
        why = "parses" if ok else "parses but no return"
    except SyntaxError as e:
        why = f"syntax error: {e}"
    task["status"] = "done" if ok else "failed"
    task["result"] = {"success": ok, "progress": 1.0 if ok else 0.0,
                      "summary": f"lm completion check: {why}; "
                                 f"generated={text[:120]!r}", "artifacts": []}
    task["metrics"]["steps_used"] = 1
    ctx["store"].save(task)
    ctx["store"].write_run_record(task)
    return task


def scoreboard(rows, commit, lm_step, grip_step, ram_peak_mb):
    by = {}
    for r in rows:
        b = by.setdefault(r["category"], {"n": 0, "success": 0})
        b["n"] += 1
        b["success"] += bool(r["success"])
    for b in by.values():
        b["rate"] = round(b["success"] / b["n"], 3)
    return {
        "commit": commit,
        "model": {"lm_step": lm_step, "grip_step": grip_step},
        "n": len(rows),
        "success": round(sum(bool(r["success"]) for r in rows) / len(rows), 3),
        "by_category": by,
        "median_steps": statistics.median([r["attempts"] for r in rows]),
        "median_seconds": round(statistics.median([r["wall_s"] for r in rows]), 2),
        "ram_peak_mb": round(ram_peak_mb, 1),
        "failures": [{"id": r["id"], "why": r["why"]} for r in rows
                     if not r["success"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames-dir", default=None,
                    help="copy failure screenshot strips here")
    args = ap.parse_args()

    base = tempfile.mkdtemp(prefix="bb_e2e_")
    store = TaskStore(home=os.path.join(base, "home"))
    lm = BrainHost(checkpoint=os.path.join(ROOT, "checkpoints", "lm"), threads=2)
    lm_info = lm.boot()
    grip = Grounder.find(os.path.join(ROOT, "checkpoints", "grip"))
    ctx = {"lm": lm, "grip": grip, "store": store}
    loop = DriveLoop(store, rng_seed=1)

    rows = []
    for spec_id, category, builder in SUITE:
        assert category in CATEGORIES
        task, world, planner, grounder, post = builder(store, base, ctx)
        t0 = time.time()
        if world is None:  # lm code task runs through the LM directly
            out = run_lm_code_task(task, ctx)
        else:
            out = loop.run_until_settled(task, world, planner, grounder)
        wall = time.time() - t0
        success = out["status"] == "done"
        why = "" if success else (
            f"{out['status']} progress={(out['result'] or {}).get('progress')} "
            f"max_rung={out['metrics']['max_rung']}; "
            f"last={out['history'][-1]['verdict'] if out['history'] else '-'}:"
            f"{out['history'][-1]['note'][:80] if out['history'] else (out['result'] or {}).get('summary','')[:80]}")
        if post and post != "lm":
            ok, note = post_check(post, out, base, ctx)
            if not ok:
                success, why = False, f"post-check failed: {note}"
        rows.append({"id": spec_id, "category": category, "task": out["id"],
                     "status": out["status"], "success": success,
                     "attempts": out["metrics"]["steps_used"],
                     "max_rung": out["metrics"]["max_rung"],
                     "wall_s": round(wall, 2), "why": why})
        print(f"{spec_id:<8} {category:<13} {out['status']:<13} "
              f"{'PASS' if success else 'FAIL'} attempts={rows[-1]['attempts']}")
        if not success and args.frames_dir:
            src_dir = os.path.join(store.runs_dir, out["id"])
            if os.path.isdir(src_dir):
                dst = os.path.join(args.frames_dir, spec_id)
                os.makedirs(dst, exist_ok=True)
                for f in sorted(os.listdir(src_dir))[-5:]:
                    shutil.copyfile(os.path.join(src_dir, f),
                                    os.path.join(dst, f))

    with open("/proc/self/status") as fh:
        ram = next(int(l.split()[1]) / 1024 for l in fh if l.startswith("VmHWM:"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    sb = scoreboard(rows, commit, lm_info["step"], grip.step, ram)
    sb["provenance"] = ("agent-01-authored stand-in for agent 03's pending "
                        "60-task Xvfb suite (their M7); same spec shape, "
                        "simulated desktop (no X server in this sandbox)")
    sb["rows"] = rows
    sb["suite_definition"] = [{"id": i, "category": c} for i, c, _ in SUITE]
    print(json.dumps({k: sb[k] for k in ("n", "success", "by_category",
                                         "median_steps", "median_seconds",
                                         "ram_peak_mb", "failures")}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(sb, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
