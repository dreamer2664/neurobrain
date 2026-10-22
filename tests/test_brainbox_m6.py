"""M6 tests: watch mode, autostart generation, crash recovery, kill."""

from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def shell_spec(ws, fname="out.txt", text="hi"):
    return {"instruction": f"write {fname}", "world": "shell", "workspace": ws,
            "checks": [{"path_exists": fname}],
            "steps": [{"intent": "write it",
                       "action": {"type": "shell", "op": "write_file",
                                  "path": fname, "text": text}}]}


class TestWorldSnapshot(unittest.TestCase):
    def test_sim_snapshot_restore(self):
        from brainbox.drive import SimulatedDesktop

        from neurobrain import ui_world

        ex = ui_world.sample_example(random.Random(3))
        w = SimulatedDesktop(ex, relocate_on_first_act=True)
        step = w.step_for(ex.target)
        w.act({"type": "click"}, (ex.target.cx, ex.target.cy))  # relocates first
        snap = w.snapshot()
        ex2 = ui_world.sample_example(random.Random(3))
        w2 = SimulatedDesktop(ex2)
        w2.restore(snap)
        self.assertEqual(w2.progress(), w.progress())
        self.assertEqual(
            [(e.x, e.y) for e in w2.layout.elements],
            [(e.x, e.y) for e in w.layout.elements])


class TestRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m6_")
        self.home = os.path.join(self.tmp, "home")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_active_task_recovered_as_failed_parked_kept(self):
        from brainbox.drive import TaskStore
        from brainbox.watch import Watcher

        store = TaskStore(home=self.home)
        active = store.create("was mid flight")
        active["status"] = "acting"
        active["plan"] = [{"step": 1, "intent": "x", "action": {"type": "click"}}]
        active["history"] = [{"step": 1, "rung": 0, "verdict": "no-op",
                              "intent": "x", "observation_before": {},
                              "action": {}, "observation_after": {}, "note": ""}]
        store.save(active)
        parked = store.create("waiting")
        parked["status"] = "parked"
        store.save(parked)
        done = store.create("finished")
        done["status"] = "done"
        store.save(done)

        inbox = os.path.join(self.home, "inbox")
        watcher = Watcher(store, inbox)
        rep = watcher.recover()
        self.assertEqual(rep["recovered_failed"], [active["id"]])
        t = store.load(active["id"])
        self.assertEqual(t["status"], "failed")
        self.assertIn("interrupted", t["result"]["summary"])
        self.assertEqual(t["history"][-1]["verdict"], "fatal")
        self.assertTrue(os.path.exists(
            os.path.join(store.runs_dir, f"{active['id']}.json")))
        self.assertEqual(store.load(parked["id"])["status"], "parked")
        self.assertEqual(store.load(done["id"])["status"], "done")

    def test_watch_once_processes_inbox(self):
        from brainbox.drive import TaskStore
        from brainbox.watch import Watcher

        store = TaskStore(home=self.home)
        inbox = os.path.join(self.home, "inbox")
        os.makedirs(inbox)
        ws = os.path.join(self.tmp, "ws")
        for name in ("a.json", "b.json"):
            with open(os.path.join(inbox, name), "w") as f:
                json.dump(shell_spec(ws, fname=name.replace(".json", ".txt")), f)
        watcher = Watcher(store, inbox)
        rep = watcher.run(max_cycles=1)
        self.assertEqual(rep["processed"], 2)
        statuses = sorted(t["status"] for t in store.list())
        self.assertEqual(statuses, ["done", "done"])
        self.assertEqual(sorted(os.listdir(inbox)),
                         ["done", "failed", "processing"])
        done_dir = os.path.join(inbox, "done")
        self.assertEqual(len([f for f in os.listdir(done_dir)
                              if f.endswith(".result.json")]), 2)
        self.assertTrue(os.path.exists(os.path.join(ws, "a.txt")))

    def test_runner_lock_excludes_second_watcher(self):
        from brainbox.drive import TaskStore
        from brainbox.watch import AlreadyRunning, RunnerLock

        lock1 = RunnerLock(self.home)
        lock1.acquire()
        try:
            with self.assertRaises(AlreadyRunning):
                RunnerLock(self.home).acquire()
        finally:
            lock1.release()
        lock2 = RunnerLock(self.home).acquire()  # reusable after release
        lock2.release()


class TestAutostart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m6auto_")
        os.environ["XDG_CONFIG_HOME"] = os.path.join(self.tmp, "config")
        self.addCleanup(lambda: os.environ.pop("XDG_CONFIG_HOME", None))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_systemd_unit_and_install_cycle(self):
        from brainbox import autostart

        unit = autostart.systemd_unit("/usr/bin/python3 -m brainbox",
                                      "/home/u/.brainbox/inbox")
        self.assertIn("ExecStart=/usr/bin/python3 -m brainbox run --watch "
                      "/home/u/.brainbox/inbox", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("WantedBy=default.target", unit)
        path = autostart.install_systemd("/usr/bin/python3 -m brainbox",
                                         "/home/u/.brainbox/inbox")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(autostart.status()["systemd_unit"])
        self.assertTrue(autostart.uninstall_systemd())
        self.assertFalse(os.path.exists(path))

    def test_schtasks_command_shape(self):
        from brainbox import autostart

        cmd = autostart.schtasks_create_command(
            r"C:\venv\Scripts\brainbox.exe", r"C:\Users\u\.brainbox\inbox")
        self.assertIn("schtasks", cmd)
        self.assertIn("/Create", cmd)
        self.assertIn('/TN "brainbox"', cmd)
        self.assertIn("ONLOGON", cmd)
        self.assertIn("run --watch", cmd)


class TestKillCLI(unittest.TestCase):
    def test_kill_stops_watcher_subprocess(self):
        tmp = tempfile.mkdtemp(prefix="bb_m6kill_")
        home = os.path.join(tmp, "home")
        inbox = os.path.join(home, "inbox")
        os.makedirs(inbox)
        env = {**os.environ, "BRAINBOX_HOME": home}
        proc = subprocess.Popen(
            [sys.executable, "-m", "brainbox", "run", "--watch", inbox,
             "--poll", "30"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        try:
            deadline = time.time() + 30
            lock = os.path.join(home, "runner.lock")
            while not os.path.exists(lock) and time.time() < deadline:
                self.assertIsNone(proc.poll(), "watcher died at startup")
                time.sleep(0.1)
            self.assertTrue(os.path.exists(lock))
            from brainbox.cli import main

            old = os.environ.get("BRAINBOX_HOME")
            os.environ["BRAINBOX_HOME"] = home
            try:
                with redirect_stdout(io.StringIO()):
                    code = main(["kill"])
            finally:
                if old is None:
                    os.environ.pop("BRAINBOX_HOME", None)
                else:
                    os.environ["BRAINBOX_HOME"] = old
            self.assertEqual(code, 0)
            proc.wait(timeout=10)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
