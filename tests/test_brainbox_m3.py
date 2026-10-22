"""M3 tests: drive loop — ladder, terminal states, budgets, task store.

Worlds are the scripted SimulatedDesktop (ui_world renders + pixel-diff
verification) with an oracle grounder: loop mechanics under test here, not
GripNet. The acceptance suite (scripts/drive_suite.py) runs the same tasks
with the real grounder too.
"""

from __future__ import annotations

import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class DriveFixture:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m3_")
        self.home = os.path.join(self.tmp, "home")

        from brainbox.drive import DriveLoop, OracleGrounder, ScriptedPlanner, TaskStore

        self.store = TaskStore(home=self.home)
        self.loop = DriveLoop(self.store)
        self.OracleGrounder = OracleGrounder
        self.ScriptedPlanner = ScriptedPlanner

    def sim_task(self, seed=7, world_kw=None, planner_kw=None, budget=None,
                 wrong_first=False, extra_targets=0):
        from brainbox.drive import SimulatedDesktop

        from neurobrain import ui_world

        rng = random.Random(seed)
        ex = ui_world.sample_example(rng)
        world = SimulatedDesktop(ex, **(world_kw or {}))
        targets = [ex.target]
        others = [e for e in ex.layout.elements if e is not ex.target]
        for e in others[:extra_targets]:
            targets.append(e)
            world.require(e)
        steps = [world.step_for(t) for t in targets]
        script = steps
        replan = None
        if wrong_first and others:
            wrong = world.step_for(others[0])
            wrong["intent"] = steps[0]["intent"]
            script = [wrong] + steps[1:]
            replan = steps
        planner = self.ScriptedPlanner(script, replan_script=replan,
                                        **(planner_kw or {}))
        task = self.store.create(ex.task, budget=budget)
        return task, world, planner

    def run(self, task, world, planner):
        return self.loop.run_until_settled(task, world, planner,
                                           self.OracleGrounder(world))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestDriveLoop(unittest.TestCase):
    def setUp(self):
        self.fx = DriveFixture()

    def tearDown(self):
        self.fx.close()

    def test_simple_task_done_no_retry(self):
        task, world, planner = self.fx.sim_task(seed=7)
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "done")
        self.assertTrue(out["result"]["success"])
        self.assertEqual(out["metrics"]["max_rung"], 0)
        self.assertGreaterEqual(len(out["history"]), 1)
        for h in out["history"]:
            self.assertIn("observation_before", h)
            self.assertIn("observation_after", h)
            self.assertIn("verdict", h)
        # persisted, append-only trace survives reload
        again = self.fx.store.load(task["id"])
        self.assertEqual(again["history"], out["history"])

    def test_flaky_click_recovers_at_rung1(self):
        task, world, planner = self.fx.sim_task(
            seed=8, world_kw={"ignore_first_correct_clicks": 1})
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["metrics"]["max_rung"], 1)
        self.assertTrue(out["metrics"]["recovered"])

    def test_relocated_widget_recovers_at_rung2(self):
        task, world, planner = self.fx.sim_task(
            seed=9, world_kw={"relocate_on_first_act": True})
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["metrics"]["max_rung"], 2)

    def test_wrong_plan_recovers_at_rung3_replan(self):
        task, world, planner = self.fx.sim_task(seed=10, wrong_first=True)
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["metrics"]["max_rung"], 3)

    def test_unreachable_fails_terminally_after_ladder(self):
        task, world, planner = self.fx.sim_task(
            seed=11, world_kw={"unreachable": True})
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "failed")
        self.assertFalse(out["result"]["success"])
        self.assertEqual(out["metrics"]["max_rung"], 6)
        self.assertTrue(out["blocked_records"], "escalate must leave a blocked record")

    def test_partial_credit_done_partial(self):
        task, world, planner = self.fx.sim_task(seed=12, extra_targets=1)
        world.unreachable_labels.add(world.required[1].label)
        out = self.fx.run(task, world, planner)
        self.assertEqual(out["status"], "done-partial")
        self.assertAlmostEqual(out["result"]["progress"], 0.5)

    def test_step_budget_forces_terminal(self):
        task, world, planner = self.fx.sim_task(
            seed=13, world_kw={"unreachable": True},
            budget={"max_steps": 3, "max_seconds": 600})
        out = self.fx.run(task, world, planner)
        self.assertIn(out["status"], ("failed", "done-partial"))
        self.assertLessEqual(out["metrics"]["steps_used"], 3)

    def test_shell_tool_switch_and_trash_delete(self):
        from brainbox.drive import ScriptedPlanner, ShellWorld
        from brainbox.trash import list_trash, restore

        ws = os.path.join(self.fx.tmp, "ws")
        os.makedirs(ws)
        victim = os.path.join(ws, "junk.tmp")
        with open(victim, "wb") as f:
            f.write(b"temporary")
        world = ShellWorld(ws, checks=[{"path_absent": "junk.tmp"}],
                           home=self.fx.home)
        planner = ScriptedPlanner(
            [{"intent": "delete junk.tmp", "ground": "click the delete button",
              "action": {"type": "click"}}],
            shell_steps=[{"intent": "delete junk.tmp via shell",
                          "action": {"type": "shell", "op": "delete",
                                     "path": "junk.tmp"}}])
        task = self.fx.store.create("delete the tmp files")
        out = self.fx.loop.run_until_settled(task, world, planner, grounder=None)
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["metrics"]["max_rung"], 5)  # tool switch
        self.assertFalse(os.path.exists(victim))
        held = list_trash(home=self.fx.home)
        self.assertEqual([t["task_id"] for t in held], [task["id"]])
        restore(task["id"], home=self.fx.home)  # and it is undoable
        with open(victim, "rb") as f:
            self.assertEqual(f.read(), b"temporary")

    def test_idle_cycle_verifies_done_and_reports(self):
        from brainbox.drive import ScriptedPlanner, ShellWorld

        ws = os.path.join(self.fx.tmp, "ws2")
        os.makedirs(ws)
        world = ShellWorld(ws, checks=[{"path_exists": "note.txt"}],
                           home=self.fx.home)
        planner = ScriptedPlanner(
            [{"intent": "write note", "action": {"type": "shell",
              "op": "write_file", "path": "note.txt", "text": "hi"}}])
        task = self.fx.store.create("write a note")
        task["checks"] = [{"path_exists": "note.txt"}]
        task["workspace"] = ws
        self.fx.store.save(task)
        out = self.fx.loop.run_until_settled(task, world, planner, grounder=None)
        self.assertEqual(out["status"], "done")
        rep = self.fx.loop.idle_cycle()
        self.assertGreaterEqual(rep["verified_ok"], 1)
        self.assertEqual(rep["verified_drift"], 0)


class TestDriveCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m3cli_")
        os.environ["BRAINBOX_HOME"] = os.path.join(self.tmp, "home")
        self.addCleanup(lambda: os.environ.pop("BRAINBOX_HOME", None))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_scripted_shell_task_then_tasks_and_logs(self):
        from brainbox.cli import main

        ws = os.path.join(self.tmp, "ws")
        os.makedirs(ws)
        script = {"world": "shell", "workspace": ws,
                  "checks": [{"path_exists": "out.txt"}],
                  "steps": [{"intent": "write out",
                             "action": {"type": "shell", "op": "write_file",
                                        "path": "out.txt", "text": "done"}}]}
        spath = os.path.join(self.tmp, "task.json")
        with open(spath, "w") as f:
            json.dump(script, f)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["run", "write out.txt", "--script", spath, "--json"])
        self.assertEqual(code, 0)
        rec = json.loads(buf.getvalue())
        self.assertEqual(rec["status"], "done")
        self.assertTrue(os.path.exists(os.path.join(ws, "out.txt")))

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["tasks", "--json"])
        self.assertEqual(code, 0)
        tasks = json.loads(buf.getvalue())
        self.assertEqual(tasks[0]["status"], "done")

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["logs", rec["id"], "--json"])
        self.assertEqual(code, 0)
        log = json.loads(buf.getvalue())
        self.assertEqual(log["id"], rec["id"])
        self.assertGreaterEqual(len(log["history"]), 1)


if __name__ == "__main__":
    unittest.main()


class TestDecompose(unittest.TestCase):
    def test_decompose_rung4(self):
        from brainbox.drive import DriveLoop, ScriptedPlanner, ShellWorld, TaskStore

        tmp = tempfile.mkdtemp(prefix="bb_m3dec_")
        try:
            ws = os.path.join(tmp, "ws")
            os.makedirs(ws)
            world = ShellWorld(ws, checks=[{"path_exists": "a/one.txt"},
                                           {"path_exists": "a/two.txt"}],
                               home=tmp)
            planner = ScriptedPlanner(
                [{"intent": "prepare the workspace",
                  "ground": "click the prepare button",
                  "action": {"type": "click"}}],
                decompose_steps=[
                    {"intent": "make dir + first file",
                     "action": {"type": "shell", "op": "write_file",
                                "path": "a/one.txt", "text": "1"}},
                    {"intent": "second file",
                     "action": {"type": "shell", "op": "write_file",
                                "path": "a/two.txt", "text": "2"}}])
            store = TaskStore(home=tmp)
            task = store.create("prepare the workspace")
            out = DriveLoop(store).run_until_settled(task, world, planner)
            self.assertEqual(out["status"], "done")
            self.assertEqual(out["metrics"]["max_rung"], 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
