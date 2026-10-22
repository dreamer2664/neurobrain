"""brainbox CLI: doctor | bench | serve | run | tasks | logs | kill | undo.

Implemented: doctor, bench, serve, run, tasks, logs, undo. Remaining verbs print the
milestone that lands them and exit 2. Nothing is faked.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

_T0 = time.time()  # process-start marker, before any torch import

from . import __version__  # noqa: E402

_NOT_YET = {
}


def _mem_total_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def cmd_doctor(args) -> int:
    from . import checkpoints as ckmod
    from .governor import peak_rss_mb, rss_mb

    versions = {"brainbox": __version__,
                "python": platform.python_version(),
                "platform": platform.platform()}
    t = time.time()
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_threads"] = torch.get_num_threads()
        torch_ok = True
    except Exception as e:  # pragma: no cover - only on a broken stack
        versions["torch"] = f"IMPORT FAILED: {e}"
        torch_ok = False
    torch_import_s = round(time.time() - t, 3)

    tok_path = ckmod.default_tokenizer_path()
    tokenizer = {"found": False}
    if tok_path:
        from neurobrain.tokenizer import BPETokenizer

        tok = BPETokenizer.load(tok_path)
        tokenizer = {"found": True, "path": tok_path,
                     "vocab_size": tok.vocab_size,
                     "sha256": ckmod.sha256_file(tok_path)}

    found = ckmod.find_checkpoint()
    if found and found[1] == "dir":
        checkpoint = {"found": True, "format": "dir",
                      "verified": ckmod.verify_checkpoint_dir(found[0]),
                      **{k: v for k, v in ckmod.inspect_checkpoint_dir(found[0]).items()
                         if k != "config"}}
    elif found:
        checkpoint = {"found": True, "format": "legacy-pt", "path": found[0],
                      "sha256": ckmod.sha256_file(found[0]), "stub": False,
                      "verified": None}
    else:
        checkpoint = {"found": False, "stub": True, "verified": None,
                      "note": "no checkpoint published; bench/serve fall back to a "
                              "seeded stub of the real architecture (see card.md)"}

    try:
        gfound = ckmod.find_grip_checkpoint()
    except ckmod.CheckpointError as e:
        gfound = None
        grip_checkpoint = {"found": False, "verified": None, "note": str(e)}
    if gfound and gfound[1] == "dir":
        grip_checkpoint = {"found": True, "format": "dir",
                           "verified": ckmod.verify_checkpoint_dir(gfound[0]),
                           **{k: v for k, v in ckmod.inspect_checkpoint_dir(gfound[0]).items()
                              if k != "config"}}
    elif gfound:
        grip_checkpoint = {"found": True, "format": "legacy-pt", "path": gfound[0],
                           "sha256": ckmod.sha256_file(gfound[0]), "verified": None}
    elif "grip_checkpoint" not in dir():
        grip_checkpoint = {"found": False, "verified": None}

    disk = shutil.disk_usage(os.path.expanduser("~"))
    intact = checkpoint.get("verified") is not False and \
        grip_checkpoint.get("verified") is not False
    report = {
        "ok": bool(torch_ok and tokenizer["found"] and intact),
        "versions": versions,
        "torch_import_s": torch_import_s,
        "cpu_count": os.cpu_count(),
        "mem_total_mb": _mem_total_mb(),
        "disk_free_mb": disk.free // (1024 * 1024),
        "rss_mb": round(rss_mb(), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "tokenizer": tokenizer,
        "checkpoint": checkpoint,
        "grip_checkpoint": grip_checkpoint,
        "uptime_s": round(time.time() - _T0, 3),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"brainbox {__version__}  python {versions['python']}  "
              f"torch {versions.get('torch')}")
        print(f"platform   {versions['platform']}  cpus={report['cpu_count']}  "
              f"mem_total={report['mem_total_mb']} MB  disk_free={report['disk_free_mb']} MB")
        print(f"tokenizer  {'ok vocab=' + str(tokenizer.get('vocab_size')) + ' ' + str(tokenizer.get('path')) if tokenizer['found'] else 'MISSING'}")
        if checkpoint["found"]:
            print(f"checkpoint {checkpoint['path']} step={checkpoint.get('step')} "
                  f"stub={checkpoint.get('stub')} verified={checkpoint.get('verified')}")
        else:
            print(f"checkpoint NONE FOUND — stub fallback active")
        if grip_checkpoint["found"]:
            print(f"grip       {grip_checkpoint['path']} step={grip_checkpoint.get('step')} "
                  f"verified={grip_checkpoint.get('verified')}")
        else:
            print("grip       NONE FOUND (/ground will 503)")
        print(f"rss        {report['rss_mb']} MB (peak {report['peak_rss_mb']} MB)")
        print("doctor:", "OK" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


def cmd_bench(args) -> int:
    t_start = time.time()
    from .governor import peak_rss_mb, rss_mb
    from .host import BrainHost

    host = BrainHost(checkpoint=args.checkpoint, threads=args.threads)
    info = host.boot()
    boot_s = time.time() - t_start

    host.generate(args.prompt, max_tokens=4, seed=0)  # warmup, untimed
    gen = host.generate(args.prompt, max_tokens=args.tokens,
                        temperature=args.temperature, seed=args.seed)
    gen_s = gen["ms"] / 1000.0
    report = {
        "boot_s": round(boot_s, 3),
        "gen_tokens": gen["tokens"],
        "gen_s": round(gen_s, 3),
        "tok_s": round(gen["tokens"] / gen_s, 2) if gen_s > 0 else None,
        "ms_per_token": round(gen["ms"] / max(1, gen["tokens"]), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rss_mb": round(rss_mb(), 1),
        "threads": args.threads,
        "prompt": args.prompt,
        "checkpoint": {k: info[k] for k in ("path", "kind", "step", "stub", "params")},
        "vocab_size": info["vocab_size"],
        "sample_text": gen["text"],
        "brainbox": __version__,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        ck = report["checkpoint"]
        print(f"checkpoint {ck['path']} step={ck['step']} stub={ck['stub']} "
              f"params={ck['params']:,}")
        print(f"boot       {report['boot_s']} s (torch import + load, {args.threads} threads)")
        print(f"generate   {report['gen_tokens']} tokens in {report['gen_s']} s  "
              f"({report['tok_s']} tok/s, {report['ms_per_token']} ms/tok)")
        print(f"peak RSS   {report['peak_rss_mb']} MB")
        print(f"sample     {gen['text']!r}")
    return 0


def cmd_serve(args) -> int:
    from .governor import Governor, rss_mb
    from .host import BrainHost
    from .server import make_server

    governor = Governor(cap_mb=args.ram_cap_mb) if args.ram_cap_mb else None
    host = BrainHost(checkpoint=args.checkpoint,
                     grip_checkpoint=args.grip_checkpoint,
                     threads=args.threads, governor=governor)
    info = host.boot()
    grip_desc = "grip=unavailable (/ground -> 503)"
    try:
        g = host.boot_grip()
        grip_desc = f"grip step={g['step']} stub={g['stub']}"
    except Exception as e:
        print(f"brainbox serve: grounder not loaded ({e})", file=sys.stderr)
    server = make_server(host, port=args.port)
    print(f"brainbox {__version__} serve http://127.0.0.1:{args.port} "
          f"lm step={info['step']} stub={info['stub']} params={info['params']:,} "
          f"{grip_desc} boot={info['boot_s']}s rss={rss_mb():.0f}MB", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbrainbox serve: stopped")
    finally:
        server.server_close()
    return 0


def cmd_undo(args) -> int:
    from .trash import TrashError, list_trash, restore

    if args.list:
        listing = list_trash()
        if args.json:
            print(json.dumps(listing, indent=2))
        elif listing:
            for t in listing:
                print(f"{t['task_id']}  entries={t['entries']}  bytes={t['bytes']}")
        else:
            print("trash is empty")
        return 0
    if not args.task_id:
        print("usage: brainbox undo <task-id> | brainbox undo --list", file=sys.stderr)
        return 2
    try:
        rep = restore(args.task_id, overwrite=args.force)
    except TrashError as e:
        print(f"brainbox undo: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        for p in rep["restored"]:
            print(f"restored {p}")
        for f in rep["failures"]:
            print(f"FAILED {f['path']}: {f['error']}", file=sys.stderr)
        print(f"undo {args.task_id}: {len(rep['restored'])} restored, "
              f"{len(rep['failures'])} failed")
    return 0 if not rep["failures"] else 1


def cmd_run(args) -> int:
    import random as _random

    from .drive import (DriveLoop, LMPlanner, OracleGrounder, RealGrounder,
                        ScriptedPlanner, ShellWorld, SimulatedDesktop, TaskStore)

    store = TaskStore()
    if args.watch:
        import signal as _signal

        from .watch import AlreadyRunning, RunnerLock, Watcher

        lock = RunnerLock()
        try:
            lock.acquire()
        except AlreadyRunning as e:
            print(f"brainbox run --watch: {e}", file=sys.stderr)
            return 1
        watcher = Watcher(store, args.watch, poll_s=args.poll)
        for sig in (_signal.SIGTERM, _signal.SIGINT):
            _signal.signal(sig, lambda *_: watcher.request_stop())
        print(f"brainbox {__version__} watching {args.watch} "
              f"(home {store.home}, pid {os.getpid()})", flush=True)
        try:
            rep = watcher.run(max_cycles=args.max_cycles)
        finally:
            lock.release()
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            print(f"watch stopped: processed={rep['processed']} "
                  f"cycles={rep['cycles']} "
                  f"recovered_failed={rep['recovered']['recovered_failed']}")
        return 0
    if not args.instruction:
        print("usage: brainbox run <instruction> | brainbox run --watch <dir>",
              file=sys.stderr)
        return 2
    budget = {}
    if args.max_steps:
        budget["max_steps"] = args.max_steps
    if args.max_seconds:
        budget["max_seconds"] = args.max_seconds

    grounder = None
    if args.script:
        from .drive import build_scripted_run

        with open(args.script) as f:
            spec = json.load(f)
        task, world, planner, grounder = build_scripted_run(
            store, args.instruction, spec, budget=budget or None)
    else:
        from neurobrain import ui_world

        from .host import BrainHost

        host = BrainHost(checkpoint=args.checkpoint, threads=args.threads)
        host.boot()
        planner = LMPlanner(host)
        task = store.create(args.instruction, budget=budget or None)
        ex = ui_world.sample_example(_random.Random(0))
        world = SimulatedDesktop(ex)
        task["world"] = "simulated desktop (no real display attached)"
        try:
            from .grip import Grounder

            grounder = RealGrounder(Grounder.find(args.grip_checkpoint))
        except Exception as e:
            print(f"brainbox run: no grip checkpoint ({e}); oracle grounder",
                  file=sys.stderr)
            grounder = OracleGrounder(world)

    out = DriveLoop(store).run_until_settled(task, world, planner, grounder)
    rec = {"id": out["id"], "status": out["status"], "result": out["result"],
           "metrics": out["metrics"], "world": out.get("world"),
           "history_len": len(out["history"])}
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        m = rec["metrics"]
        print(f"{rec['id']}: {rec['status']}  progress="
              f"{(rec['result'] or {}).get('progress')}  attempts={m['steps_used']}  "
              f"max_rung={m['max_rung']}  recovered={m['recovered']}")
        if rec["world"]:
            print(f"world: {rec['world']}")
    return 0 if out["status"] == "done" else 1


def cmd_tasks(args) -> int:
    from .drive import TaskStore

    tasks = TaskStore().list()
    rows = [{"id": t["id"], "status": t["status"],
             "instruction": t["instruction"],
             "attempts": t["metrics"]["steps_used"],
             "max_rung": t["metrics"]["max_rung"]} for t in tasks]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['id']}  {r['status']:<12}  attempts={r['attempts']}  "
                  f"rung<={r['max_rung']}  {r['instruction'][:60]}")
        if not rows:
            print("no tasks yet")
    return 0


def cmd_logs(args) -> int:
    from .drive import TaskStore

    store = TaskStore()
    if not args.task_id:
        names = sorted(f[:-5] for f in os.listdir(store.runs_dir)
                       if f.endswith(".json"))
        print(json.dumps(names, indent=2) if args.json else "\n".join(names))
        return 0
    run_path = os.path.join(store.runs_dir, f"{args.task_id}.json")
    try:
        if os.path.exists(run_path):
            with open(run_path) as f:
                rec = json.load(f)
        else:
            rec = store.load(args.task_id)
    except Exception as e:
        print(f"brainbox logs: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(f"{rec['id']}: {rec['status']} — {rec['instruction']}")
        for h in rec["history"]:
            print(f"  step {h['step']} rung {h['rung']} {h['verdict']}: "
                  f"{h['intent'][:50]} ({h['note'][:60]})")
    return 0


def cmd_kill(args) -> int:
    from .watch import kill_runner

    rep = kill_runner()
    if args.json:
        print(json.dumps(rep, indent=2))
    elif rep["killed"]:
        print(f"killed brainbox runner pid {rep['pid']} via {rep.get('via')}")
    else:
        print(f"brainbox kill: {rep['detail']}")
    return 0 if rep["killed"] else 1


def cmd_autostart(args) -> int:
    import platform
    import subprocess

    from . import autostart as A
    from .trash import brainbox_home

    watch = args.watch or os.path.join(brainbox_home(), "inbox")
    action = args.action
    if action == "status":
        st = A.status()
        print(json.dumps(st, indent=2) if args.json else
              f"platform={st['platform']} systemd_unit={st['systemd_unit']} "
              f"({st['systemd_path']})")
        return 0
    system = platform.system()
    if action == "install":
        if system == "Windows":
            exe = os.path.join(os.path.dirname(sys.executable), "Scripts",
                               "brainbox.exe")
            cmd = A.schtasks_create_command(exe, watch)
            r = subprocess.run(cmd, shell=True)
            return r.returncode
        exe = A.default_exec()
        path = A.install_systemd(exe, watch, home=brainbox_home())
        print(f"wrote {path}")
        enable = "systemctl --user daemon-reload && systemctl --user enable --now brainbox"
        if args.enable:
            r = subprocess.run(enable, shell=True)
            if r.returncode != 0:
                print("enable failed here (no user systemd?); unit file is in "
                      "place, run on the target:\n  " + enable)
            return 0
        print("enable with:\n  " + enable)
        if system == "Windows":
            pass
        return 0
    if action == "uninstall":
        if system == "Windows":
            return subprocess.run(A.schtasks_delete_command(), shell=True).returncode
        removed = A.uninstall_systemd()
        print("removed systemd unit" if removed else "no systemd unit found")
        return 0 if removed else 1
    return 2


def _not_yet(verb: str):
    def handler(_args) -> int:
        print(f"brainbox {verb}: not implemented yet — lands in {_NOT_YET[verb]}")
        return 2
    return handler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brainbox",
        description="neurobrain runtime host: loads from-scratch checkpoints, "
                    "serves them locally, runs tasks under a 4 GB budget.")
    p.add_argument("--version", action="version", version=f"brainbox {__version__}")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("doctor", help="verify stack, tokenizer, checkpoint; report versions")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_doctor)

    b = sub.add_parser("bench", help="boot + generate N tokens; print boot time and peak RSS")
    b.add_argument("--checkpoint", default=None,
                   help="checkpoint dir or legacy .pt (default: search, else stub)")
    b.add_argument("--tokens", type=int, default=20)
    b.add_argument("--prompt", default="# quicksort\n")
    b.add_argument("--temperature", type=float, default=0.8)
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json", action="store_true")
    b.add_argument("--out", default=None, help="also write the JSON report here")
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("serve", help="local API on 127.0.0.1 (specs/model_io.md)")
    s.add_argument("--port", type=int,
                   default=int(os.environ.get("BRAINBOX_PORT", "8321")),
                   help="port on 127.0.0.1 (default 8321, env BRAINBOX_PORT)")
    s.add_argument("--checkpoint", default=None,
                   help="LM checkpoint dir or legacy .pt (default: search)")
    s.add_argument("--grip-checkpoint", default=None,
                   help="GripNet checkpoint dir or legacy .pt (default: search)")
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--ram-cap-mb", type=float, default=None,
                   help="governor hard cap (default 3800, env BRAINBOX_RAM_CAP_MB)")
    s.set_defaults(fn=cmd_serve)

    u = sub.add_parser("undo", help="restore files trashed by a task (M4)")
    u.add_argument("task_id", nargs="?")
    u.add_argument("--list", action="store_true", help="list tasks holding trash")
    u.add_argument("--force", action="store_true",
                   help="overwrite paths that exist again")
    u.add_argument("--json", action="store_true")
    u.set_defaults(fn=cmd_undo)

    r = sub.add_parser("run", help="run a task through the drive loop (M3)")
    r.add_argument("instruction", nargs="?")
    r.add_argument("--watch", default=None, metavar="DIR",
                   help="watch an inbox dir of task specs (M6; survives reboot/kill -9)")
    r.add_argument("--poll", type=float, default=2.0)
    r.add_argument("--max-cycles", type=int, default=None,
                   help="stop after N watch cycles (default: run forever)")
    r.add_argument("--script", default=None,
                   help="JSON scripted task {world, workspace, checks, steps...}")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--grip-checkpoint", default=None)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--max-seconds", type=int, default=None)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("tasks", help="list tasks and their status (M3)")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=cmd_tasks)

    lg = sub.add_parser("logs", help="show a task's run record / history (M3)")
    lg.add_argument("task_id", nargs="?")
    lg.add_argument("--json", action="store_true")
    lg.set_defaults(fn=cmd_logs)

    k = sub.add_parser("kill", help="stop the watch runner holding the lock (M6)")
    k.add_argument("--json", action="store_true")
    k.set_defaults(fn=cmd_kill)

    a = sub.add_parser("autostart", help="register brainbox at login (M6)")
    a.add_argument("action", choices=("install", "uninstall", "status"))
    a.add_argument("--watch", default=None, metavar="DIR")
    a.add_argument("--enable", action="store_true",
                   help="also run systemctl --user enable --now (Linux)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_autostart)

    for verb, milestone in _NOT_YET.items():
        s = sub.add_parser(verb, help=f"({milestone})")
        s.add_argument("rest", nargs="*")
        s.set_defaults(fn=_not_yet(verb))
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
