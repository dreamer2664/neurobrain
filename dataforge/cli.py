"""dataforge CLI.

    python -m dataforge build  --out DIR --profile {smoke,standard,max}
    python -m dataforge stats  --dir DIR [--json PATH]
    python -m dataforge verify --dir DIR
    python -m dataforge link   --dir DIR --to data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__, manifest as M, profiles as P
from . import stats as S, verify as V
from .generators import IMPLEMENTED, TODO


def _log(quiet: bool):
    if quiet:
        return lambda *a, **k: None
    return lambda msg: print(msg, flush=True)


def _apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """--set dataset.key=value, e.g. --set hard_ui.samples=10000."""
    cfg = {k: dict(v) for k, v in cfg.items()}
    for item in sets or []:
        if "=" not in item or "." not in item.split("=")[0]:
            raise SystemExit(f"dataforge: --set wants dataset.key=value, got {item!r}")
        key, val = item.split("=", 1)
        ds, knob = key.split(".", 1)
        if ds not in cfg:
            raise SystemExit(f"dataforge: --set unknown dataset {ds!r}")
        try:
            num = int(val)
        except ValueError:
            try:
                num = float(val)
            except ValueError:
                num = val
        cfg[ds][knob] = num
    return cfg


def cmd_build(args) -> int:
    cfg = _apply_overrides(P.resolve(args.profile), args.set)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    log = _log(args.quiet)
    from .stats import _load_tok, find_tokenizer
    tok_path = find_tokenizer(out, args.tokenizer)
    tok = _load_tok(tok_path)

    want = args.datasets.split(",") if args.datasets else list(P.BUILD_ORDER)
    unknown = [w for w in want if w not in P.DATASETS]
    if unknown:
        print(f"dataforge: unknown datasets {unknown}; known: {list(P.DATASETS)}", file=sys.stderr)
        return 2

    log(f"[build] profile={args.profile} seed={args.seed} out={out} tokenizer={tok_path}")
    t0 = time.time()
    index: dict = {"profile": args.profile, "seed": args.seed,
                   "datasets": {}, "skipped": {}, "build_seconds": None}
    failed = False

    for name in want:
        if name not in IMPLEMENTED:
            index["skipped"][name] = TODO.get(name, "not implemented")
            log(f"[build] skip {name}: {TODO.get(name, 'not implemented')}")
            continue
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        t = time.time()
        try:
            body = IMPLEMENTED[name](d, cfg[name], args.seed, tok, log=log)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: build raised {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        body.update({"profile": args.profile, "seed": args.seed})
        M.write_manifest(d, body)
        index["datasets"][name] = {
            "n": body["n"], "units": body["units"], "kind": body["kind"],
            "shards": [{"name": s["name"], "sha256": s["sha256"], "bytes": s["bytes"]}
                       for s in body["shards"]],
            "seconds": round(time.time() - t, 2),
        }
        log(f"[build] {name}: {body['n']:,} {body['units']} in {time.time()-t:.1f}s")

    index["build_seconds"] = round(time.time() - t0, 2)
    M.write_manifest(out, {"version": M.VERSION, "generated_by": M.GENERATOR,
                           "profile": args.profile, "seed": args.seed,
                           "datasets": index["datasets"], "skipped": index["skipped"],
                           "build_seconds": index["build_seconds"]})

    if args.no_stats:
        log(f"[build] done in {time.time()-t0:.1f}s (stats skipped)")
        return 0

    log("[build] running stats (out-of-range numbers fail the build)")
    report = S.stats_dir(out, tokenizer=tok_path, sample_docs=args.sample_docs)
    print(S.render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(S.json_view(report), f, indent=1, sort_keys=True)
    if not report["ok"]:
        print("FAIL: out-of-range numbers, build rejected", file=sys.stderr)
        failed = True
    log(f"[build] total {time.time()-t0:.1f}s")
    return 1 if failed else 0


def cmd_stats(args) -> int:
    report = S.stats_dir(args.dir, tokenizer=args.tokenizer, sample_docs=args.sample_docs,
                         max_image_shards=args.image_shards)
    print(S.render(report))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(S.json_view(report), f, indent=1, sort_keys=True)
        print(f"[stats] wrote {args.json}")
    return 0 if report["ok"] else 1


def cmd_verify(args) -> int:
    ok, problems, summary = V.verify_dir(args.dir, tokenizer=args.tokenizer)
    for p in problems:
        print(p)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
    if ok:
        ds = ", ".join(f"{k}:{v['shards']}" for k, v in summary["datasets"].items())
        print(f"verify OK  dir={summary['dir']}  shards({ds})  sha256 all match")
        return 0
    print(f"verify FAILED  dir={summary['dir']}  {len(problems)} problem(s)")
    return 1


def cmd_link(args) -> int:
    """Symlink the legacy flat layout (data/code_train.bin ...) onto a forge dir."""
    to = os.path.abspath(args.to)
    os.makedirs(to, exist_ok=True)
    n = 0
    for d, mf in M.dataset_dirs(args.dir):
        for sh in mf.get("shards", []):
            src = os.path.join(d, sh["name"])
            dst = os.path.join(to, sh["name"])
            if os.path.islink(dst) or os.path.isfile(dst):
                os.unlink(dst)
            os.symlink(src, dst)
            n += 1
    tok_src = S.find_tokenizer(args.dir, args.tokenizer)
    tok_dst = os.path.join(to, "tokenizer.json")
    if not os.path.exists(tok_dst):
        os.symlink(tok_src, tok_dst)
    print(f"[link] {n} shards + tokenizer -> {to}")
    return 0


def cmd_index(args) -> int:
    """Rebuild the top-level manifest from the per-dataset manifests on disk.

    Needed after a partial rebuild (`--datasets drive`), which would otherwise
    leave the index listing only what that one command produced.
    """
    root = os.path.abspath(args.dir)
    datasets, skipped = {}, {}
    seed = profile = None
    for d, mf in M.dataset_dirs(root):
        name = mf.get("dataset") or os.path.basename(d)
        datasets[name] = {
            "n": mf.get("n"), "units": mf.get("units"), "kind": mf.get("kind"),
            "shards": [{"name": s["name"], "sha256": s["sha256"], "bytes": s["bytes"]}
                       for s in list(mf.get("shards", [])) + list(mf.get("records", []) or [])],
        }
        seed = mf.get("seed", seed)
        profile = mf.get("profile", profile)
    M.write_manifest(root, {"version": M.VERSION, "generated_by": M.GENERATOR,
                            "profile": profile, "seed": seed,
                            "datasets": datasets, "skipped": skipped})
    print(f"[index] {len(datasets)} datasets: " + ", ".join(
        f"{k}({v['n']})" for k, v in sorted(datasets.items())))
    return 0


def cmd_diff(args) -> int:
    """Determinism proof: same seed + profile => same sha256 for every shard."""
    def hashes(root):
        h = {}
        for d, mf in M.dataset_dirs(root):
            name = mf.get("dataset") or os.path.basename(d)
            for sh in mf.get("shards", []):
                h[f"{name}/{sh['name']}"] = sh["sha256"]
        return h

    a, b = hashes(args.a), hashes(args.b)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    for k in only_a:
        print(f"only in A: {k}")
    for k in only_b:
        print(f"only in B: {k}")
    for k in diff:
        print(f"HASH DIFF: {k}  {a[k][:12]}.. != {b[k][:12]}..")
    if only_a or only_b or diff:
        print(f"determinism FAILED  {len(diff)} hash differences, {len(only_a)}+{len(only_b)} missing")
        return 1
    print(f"determinism OK  {len(a)} shards identical between {args.a} and {args.b}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dataforge", description=f"corpus factory v{__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate every dataset for a profile")
    b.add_argument("--out", required=True)
    b.add_argument("--profile", default="smoke", choices=sorted(P.PROFILES))
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--datasets", default=None, help="comma list; default all")
    b.add_argument("--tokenizer", default=None)
    b.add_argument("--sample-docs", type=int, default=3000)
    b.add_argument("--json", default=None)
    b.add_argument("--set", action="append", metavar="DS.KEY=VAL",
                   help="override a profile knob, e.g. --set hard_ui.samples=10000")
    b.add_argument("--no-stats", action="store_true")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("stats", help="health report, recomputed from the shards")
    s.add_argument("--dir", required=True)
    s.add_argument("--json", default=None)
    s.add_argument("--tokenizer", default=None)
    s.add_argument("--sample-docs", type=int, default=5000)
    s.add_argument("--image-shards", type=int, default=4)
    s.set_defaults(func=cmd_stats)

    v = sub.add_parser("verify", help="sha256 every shard + re-check semantics")
    v.add_argument("--dir", required=True)
    v.add_argument("--tokenizer", default=None)
    v.add_argument("--json", default=None)
    v.set_defaults(func=cmd_verify)

    ix = sub.add_parser("index", help="rebuild the top-level manifest from dataset manifests")
    ix.add_argument("--dir", required=True)
    ix.set_defaults(func=cmd_index)

    df = sub.add_parser("diff", help="prove two builds produced identical shards")
    df.add_argument("--a", required=True)
    df.add_argument("--b", required=True)
    df.set_defaults(func=cmd_diff)

    l = sub.add_parser("link", help="symlink shards into the legacy flat layout")
    l.add_argument("--dir", required=True)
    l.add_argument("--to", required=True)
    l.add_argument("--tokenizer", default=None)
    l.set_defaults(func=cmd_link)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
