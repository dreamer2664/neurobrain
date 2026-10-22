"""brainbox CLI: doctor | bench | serve | run | tasks | logs | kill | undo.

Implemented: doctor (M0), bench (M0), serve (M1). Remaining verbs print the
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
    "run": "M3 (drive loop per specs/drive.md)",
    "tasks": "M3 (task store per specs/task_schema.md)",
    "logs": "M3 (run records under ~/.brainbox/runs/)",
    "kill": "M6 (watch/autostart supervision)",
    "undo": "M4 (trash/restore under ~/.brainbox/trash/)",
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
                      **{k: v for k, v in ckmod.inspect_checkpoint_dir(found[0]).items()
                         if k != "config"}}
    elif found:
        checkpoint = {"found": True, "format": "legacy-pt", "path": found[0],
                      "sha256": ckmod.sha256_file(found[0]), "stub": False}
    else:
        checkpoint = {"found": False, "stub": True,
                      "note": "no checkpoint published; bench/serve fall back to a "
                              "seeded stub of the real architecture (see card.md)"}

    disk = shutil.disk_usage(os.path.expanduser("~"))
    report = {
        "ok": bool(torch_ok and tokenizer["found"]),
        "versions": versions,
        "torch_import_s": torch_import_s,
        "cpu_count": os.cpu_count(),
        "mem_total_mb": _mem_total_mb(),
        "disk_free_mb": disk.free // (1024 * 1024),
        "rss_mb": round(rss_mb(), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "tokenizer": tokenizer,
        "checkpoint": checkpoint,
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
                  f"stub={checkpoint.get('stub')}")
        else:
            print(f"checkpoint NONE FOUND — stub fallback active")
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
