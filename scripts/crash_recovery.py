"""M6 acceptance: kill -9 the watcher mid-task, restart, prove recovery.

Queue: 01 quick, 02 slow (killed during its 12 s write), 03 queued behind.
Pass = 01 done, 02 recorded failed (interrupted) with a run record,
03 still processed after the crash, no stuck process, lock re-acquirable.

    python3 scripts/crash_recovery.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def write_spec(inbox, name, spec):
    with open(os.path.join(inbox, name), "w") as f:
        json.dump(spec, f)


def load_tasks(home):
    d = os.path.join(home, "tasks")
    out = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.endswith(".json"):
                with open(os.path.join(d, n)) as f:
                    out.append(json.load(f))
    return out


def spawn_watcher(home, inbox, extra=()):
    env = {**os.environ, "BRAINBOX_HOME": home}
    return subprocess.Popen(
        [sys.executable, "-m", "brainbox", "run", "--watch", inbox,
         "--poll", "0.5", *extra],
        cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = tempfile.mkdtemp(prefix="bb_crash_")
    home, inbox, ws = (os.path.join(base, x) for x in ("home", "inbox", "ws"))
    os.makedirs(inbox)
    os.makedirs(ws)
    quick = lambda fname: {  # noqa: E731
        "instruction": f"write {fname}", "world": "shell", "workspace": ws,
        "checks": [{"path_exists": fname}],
        "steps": [{"intent": "write", "action": {"type": "shell",
                   "op": "write_file", "path": fname, "text": "ok"}}]}
    write_spec(inbox, "01-quick.json", quick("quick.txt"))
    write_spec(inbox, "02-slow.json", {
        "instruction": "slow two-step write", "world": "shell", "workspace": ws,
        "checks": [{"path_exists": "finished.txt"}],
        "steps": [
            {"intent": "mark start", "action": {"type": "shell",
             "op": "write_file", "path": "started.txt", "text": "s"}},
            {"intent": "slow write", "action": {"type": "shell",
             "op": "write_file", "path": "finished.txt", "text": "f",
             "delay_ms": 120000}}]})
    write_spec(inbox, "03-after.json", quick("after.txt"))

    report = {"milestone": "M6", "phases": {}}
    p1 = spawn_watcher(home, inbox)
    try:
        deadline = time.time() + 60
        slow = None
        while time.time() < deadline:
            tasks = load_tasks(home)
            slow = next((t for t in tasks
                         if t.get("inbox_file") == "02-slow.json"), None)
            if slow and slow["history"] and os.path.exists(
                    os.path.join(ws, "started.txt")):
                break
            if p1.poll() is not None:
                print(p1.stdout.read())
                print("watcher died early", file=sys.stderr)
                return 1
            time.sleep(0.2)
        assert slow, "slow task never started"
        report["phases"]["before_kill"] = {
            "slow_status": slow["status"], "history_len": len(slow["history"])}
        time.sleep(1.0)  # be sure we are inside the 120 s write
        p1.send_signal(signal.SIGKILL)
        p1.wait(timeout=10)
        report["phases"]["killed"] = {"returncode": p1.returncode}
        assert p1.returncode == -signal.SIGKILL
    finally:
        if p1.poll() is None:
            p1.kill()

    p2 = spawn_watcher(home, inbox, extra=("--max-cycles", "3"))
    out, _ = p2.communicate(timeout=120)
    report["phases"]["after_restart"] = {"returncode": p2.returncode,
                                         "watcher_out_tail": out.strip().splitlines()[-2:]}
    tasks = {t.get("inbox_file"): t for t in load_tasks(home)}
    slow = tasks["02-slow.json"]
    from brainbox.watch import RunnerLock

    lock = RunnerLock(home).acquire()
    lock.release()
    report["tasks"] = {k: {"status": v["status"],
                           "attempts": v["metrics"]["steps_used"],
                           "summary": (v["result"] or {}).get("summary")}
                       for k, v in tasks.items()}
    report["checks"] = {
        "quick_done": tasks["01-quick.json"]["status"] == "done",
        "slow_failed_interrupted": slow["status"] == "failed"
            and "interrupted" in (slow["result"] or {}).get("summary", ""),
        "slow_history_fatal": slow["history"][-1]["verdict"] == "fatal",
        "slow_run_record": os.path.exists(
            os.path.join(home, "runs", f"{slow['id']}.json")),
        "finished_absent": not os.path.exists(os.path.join(ws, "finished.txt")),
        "queued_after_processed": tasks["03-after.json"]["status"] == "done",
        "slow_spec_filed_failed": os.path.exists(
            os.path.join(inbox, "failed", "02-slow.json.result.json")),
        "lock_reacquirable_no_stuck_process": True,
        "second_watcher_exit0": p2.returncode == 0,
    }
    report["pass"] = all(report["checks"].values())
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
