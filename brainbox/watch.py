"""Watch mode: an inbox directory is the durable task queue (M6).

    brainbox run --watch ~/.brainbox/inbox

Drop a JSON task spec (same shape as `brainbox run --script`, plus an
"instruction" field) into the inbox; the watcher moves it to `processing/`,
runs it through the drive loop, then files the spec + a `<name>.result.json`
under `done/` or `failed/`. Files left in `processing/` at startup mean the
previous runner died mid-task: those tasks are recorded as failed (with a
fatal history entry and a run record), orphans with no task record are
re-queued, and untouched inbox specs simply process as usual. The queue
therefore survives `kill -9` and reboots intact.

One runner per brainbox home, enforced by an flock'd `runner.lock` pidfile;
`brainbox kill` signals whatever holds it (SIGTERM, then SIGKILL).
"""

from __future__ import annotations

import json
import os
import signal
import time

from .drive import TERMINAL, DriveLoop, TaskStore, _atomic_write, build_scripted_run
from .trash import brainbox_home

ACTIVE = ("planning", "acting", "verifying")


class AlreadyRunning(RuntimeError):
    pass


class RunnerLock:
    """flock'd pidfile at <home>/runner.lock (single runner per home)."""

    def __init__(self, home: str | None = None):
        self.home = brainbox_home(home)
        self.path = os.path.join(self.home, "runner.lock")
        self._fd = None

    def acquire(self):
        import fcntl

        os.makedirs(self.home, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = self.read_pid()
            os.close(fd)
            raise AlreadyRunning(f"another brainbox runner holds the lock "
                                 f"(pid {pid})")
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self):
        if self._fd is not None:
            try:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(self._fd)
            self._fd = None
            try:
                os.remove(self.path)
            except OSError:
                pass

    def read_pid(self) -> int | None:
        try:
            with open(self.path) as f:
                return int(f.read().strip() or 0) or None
        except Exception:
            return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            if f.read().rsplit(")", 1)[-1].split()[0] == "Z":
                return False  # zombie: exited, not yet reaped by its parent
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def kill_runner(home: str | None = None, timeout: float = 5.0) -> dict:
    """SIGTERM the lock holder, escalate to SIGKILL. Honest about misses."""
    lock = RunnerLock(home)
    pid = lock.read_pid()
    if not pid_alive(pid):
        stale = os.path.exists(lock.path)
        return {"killed": False, "pid": pid, "stale_lock": stale,
                "detail": "no live runner (stale lock)" if stale else "not running"}
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.05)
    how = "SIGTERM"
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        how = "SIGKILL"
        while time.time() < deadline + 2 and pid_alive(pid):
            time.sleep(0.05)
    killed = not pid_alive(pid)
    return {"killed": killed, "pid": pid, "via": how,
            "detail": f"killed via {how}" if killed else "process still alive"}


class Watcher:
    def __init__(self, store: TaskStore, inbox: str, poll_s: float = 2.0):
        self.store = store
        self.inbox = os.path.abspath(inbox)
        self.poll_s = poll_s
        self.stop = False
        self.loop = DriveLoop(store)
        for sub in ("", "processing", "done", "failed"):
            os.makedirs(os.path.join(self.inbox, sub), exist_ok=True)

    def request_stop(self):
        self.stop = True

    # ------------------------------------------------------------- recovery
    def recover(self) -> dict:
        """Startup sweep after any unclean exit. Idempotent."""
        rep = {"recovered_failed": [], "requeued": []}
        tasks = self.store.list()
        by_inbox = {t.get("inbox_file"): t for t in tasks if t.get("inbox_file")}
        for task in tasks:
            if task["status"] in ACTIVE:
                task["status"] = "failed"
                progress = (task.get("plan_index", 0) / len(task["plan"])
                            if task.get("plan") else 0.0)
                task["result"] = {
                    "success": False, "progress": progress,
                    "summary": f"interrupted: runner process died mid-task; "
                               f"recorded as failed at recovery "
                               f"({task['metrics']['steps_used']} attempts made)",
                    "artifacts": []}
                task["history"].append({
                    "step": len(task["plan"]) or 1, "rung": 6,
                    "intent": "recovery", "observation_before": {},
                    "action": {}, "observation_after": {},
                    "verdict": "fatal",
                    "note": "recovered after crash: task was in an active "
                            "state with no live runner"})
                self.store.save(task)
                self.store.write_run_record(task)
                rep["recovered_failed"].append(task["id"])
        proc_dir = os.path.join(self.inbox, "processing")
        for name in sorted(os.listdir(proc_dir)):
            src = os.path.join(proc_dir, name)
            task = by_inbox.get(name)
            if task is not None and task["status"] in TERMINAL:
                self._file_result(src, "failed",
                                  {"task_id": task["id"], "status": task["status"],
                                   "recovered_after_crash": True,
                                   "result": task["result"]})
            elif task is None:
                os.replace(src, os.path.join(self.inbox, name))  # never drop work
                rep["requeued"].append(name)
            # task exists but non-terminal (parked): leave in processing; its
            # resume is the drive loop's business, the spec file is evidence.
        return rep

    # ------------------------------------------------------------ processing
    def _file_result(self, spec_path: str, bucket: str, result: dict) -> None:
        dest_dir = os.path.join(self.inbox, bucket)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(spec_path))
        os.replace(spec_path, dest)
        _atomic_write(dest + ".result.json", result)

    def process_file(self, name: str) -> dict:
        src = os.path.join(self.inbox, name)
        processing = os.path.join(self.inbox, "processing", name)
        os.replace(src, processing)
        try:
            with open(processing) as f:
                spec = json.load(f)
            instruction = spec.get("instruction") or f"inbox task {name}"
            task, world, planner, grounder = build_scripted_run(
                self.store, instruction, spec, inbox_file=name)
            out = self.loop.run_until_settled(task, world, planner, grounder)
            result = {"task_id": out["id"], "status": out["status"],
                      "result": out["result"], "metrics": out["metrics"]}
            self._file_result(processing,
                              "done" if out["status"] == "done" else "failed",
                              result)
            return result
        except Exception as e:
            result = {"task_id": None, "status": "failed",
                      "error": f"{type(e).__name__}: {e}"}
            self._file_result(processing, "failed", result)
            return result

    def run(self, max_cycles: int | None = None) -> dict:
        rep = {"processed": 0, "cycles": 0, "recovered": self.recover()}
        while not self.stop:
            rep["cycles"] += 1
            names = sorted(f for f in os.listdir(self.inbox)
                           if f.endswith(".json")
                           and os.path.isfile(os.path.join(self.inbox, f)))
            for name in names:
                if self.stop:
                    break
                self.process_file(name)
                rep["processed"] += 1
            if max_cycles is not None and rep["cycles"] >= max_cycles:
                break
            if not names:
                self.loop.idle_cycle()
                slept = 0.0
                while slept < self.poll_s and not self.stop:
                    time.sleep(min(0.2, self.poll_s - slept))
                    slept += 0.2
        return rep
