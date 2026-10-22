"""Health report for a forge directory, recomputed from the bytes on disk.

Never trusts the manifest's own counters: duplicates, label balance,
coordinate spread, image entropy and the drive invariants are all derived from
the shards themselves. ``check_dataset`` turns those numbers into a verdict;
a single violation makes ``build`` (and ``stats``) exit non-zero, so a bad
shard never reaches a trainer.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from . import checks, drive_rules, manifest as M, tokenscan
from .profiles import PROFILES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------- helpers
def find_tokenizer(root: str, explicit: str | None = None) -> str:
    cands = [explicit] if explicit else []
    cands += [os.path.join(root, "tokenizer.json"),
              os.path.join(os.path.dirname(os.path.abspath(root)), "tokenizer.json"),
              os.path.join(ROOT, "data", "tokenizer.json"),
              os.path.join(os.getcwd(), "data", "tokenizer.json")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("dataforge: no tokenizer.json found; pass --tokenizer")


def _load_tok(path: str):
    from neurobrain.tokenizer import BPETokenizer
    return BPETokenizer.load(path)


def _norm(text: str) -> bytes:
    return " ".join(text.split()).encode("utf-8", "replace")


def _hist_stats(counts) -> dict:
    counts = [int(c) for c in counts]
    total = sum(counts) or 1
    n_labels = len(counts)
    return {
        "counts": counts,
        "n_labels": n_labels,
        "entropy": round(checks.entropy(counts), 4),
        "entropy_ratio": round(checks.entropy_ratio(counts), 4),
        "min_share": round(min(counts) / total, 5),
        "max_share": round(max(counts) / total, 5),
        "balanced": round(checks.balanced(counts), 4),
    }


# ------------------------------------------------------------- token streams
def token_stream_stats(path: str, tok, sample_docs: int = 5000, scan_episodes: bool = False) -> dict:
    arr = np.memmap(path, dtype="<u2", mode="r")
    n_tokens = int(len(arr))
    if n_tokens == 0:
        return {"tokens": 0, "docs": 0, "error": "empty stream"}

    # EOT boundaries (chunked; the only full pass we need for counts)
    eot = tokenscan.eot_id(tok)
    eots = []
    vocab_counts = np.zeros(max(65536, tok.vocab_size), dtype=np.int64)
    for s, e in tokenscan._iter_chunks(arr, 32 << 20):
        chunk = np.asarray(arr[s:e])
        eots.append(np.flatnonzero(chunk == eot).astype(np.int64) + s)
        vocab_counts += np.bincount(chunk, minlength=vocab_counts.size)
    eots_arr = np.concatenate(eots) if eots else np.empty(0, np.int64)
    bounds = tokenscan.episode_bounds(eots_arr, n_tokens)
    n_docs = int(len(bounds))

    # sampled docs -> duplicates + length distribution
    k = min(sample_docs, n_docs)
    idx = np.linspace(0, n_docs - 1, k).astype(np.int64) if n_docs else np.empty(0, np.int64)
    digests = []
    for i in idx:
        s, e = int(bounds[i, 0]), int(bounds[i, 1])
        digests.append(hashlib.blake2b(_norm(tok.decode(arr[s:e])), digest_size=16).digest())
    exact_rate = 0.0
    if digests:
        exact_rate = round(1.0 - len(set(digests)) / len(digests), 6)
    near = round(1.0 - len(set(d[:8] for d in digests)) / len(digests), 6) if digests else 0.0

    lens = (bounds[:, 1] - bounds[:, 0]) if n_docs else np.zeros(1, np.int64)
    doc_len = {
        "mean": round(float(lens.mean()), 1),
        "p50": int(np.percentile(lens, 50)),
        "p99": int(np.percentile(lens, 99)),
        "max": int(lens.max()),
    }

    used = int(np.count_nonzero(vocab_counts))
    top = np.argsort(vocab_counts)[-8:][::-1]
    top_tokens = [{"id": int(i), "text": tok.decode([int(i)])[:24],
                   "share": round(float(vocab_counts[i]) / n_tokens, 5)} for i in top]

    out = {
        "kind": "token_streams",
        "path": os.path.basename(path),
        "tokens": n_tokens,
        "docs": n_docs,
        "vocab_used": used,
        "balancing": round(tokenscan.token_classes(tok, arr)["_share"]["special"], 5),
        "balance": tokenscan.token_classes(tok, arr),
        "top_tokens": top_tokens,
        "duplicates": {"sampled": int(k), "exact_rate": exact_rate, "near_rate": near},
        "doc_len": doc_len,
    }
    if scan_episodes:
        s = tokenscan.scan(arr, tok)
        s["ranges_ok"] = not drive_rules.check(s) if "episodes" in s else False
        out["invariants"] = s
    return out


# --------------------------------------------------------------- jsonl tasks
def jsonl_stats(dirpath: str, manifest: dict, name: str, sample: int = 4000) -> dict:
    """Re-validate sampled trajectories from the gzip on disk."""
    import gzip
    import json as _json
    from neurobrain.trajectories import (FAIL_VERDICTS, has_failure, has_recovery,
                                         validate_trajectory)

    shards = manifest.get("shards", [])
    n = sum(s.get("n", 0) for s in shards)
    seen = 0
    invalid = 0
    failures = recoveries = 0
    verdicts: dict[str, int] = {}
    steps = hist = 0
    ids: set[str] = set()
    dup_ids = 0
    per = max(1, sample // max(1, len(shards)))
    for sh in shards:
        p = os.path.join(dirpath, sh["name"])
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                rec = _json.loads(line)
                if seen < sample * len(shards) and seen % max(1, (sh["n"] // per or 1)) == 0:
                    errs = validate_trajectory(rec)
                    if errs:
                        invalid += 1
                    if rec["id"] in ids:
                        dup_ids += 1
                    ids.add(rec["id"])
                    steps += len(rec["plan"])
                    hist += len(rec["history"])
                    for h in rec["history"]:
                        verdicts[h["verdict"]] = verdicts.get(h["verdict"], 0) + 1
                    if has_failure(rec):
                        failures += 1
                        if has_recovery(rec):
                            recoveries += 1
                seen += 1
    checked = len(ids)
    return {
        "kind": "jsonl",
        "dataset": name,
        "units": manifest.get("units", "tasks"),
        "n": n,
        "shards": len(shards),
        "checked": checked,
        "invalid": invalid,
        "dup_ids": dup_ids,
        "duplicates": {"sampled": checked, "exact_rate": round(dup_ids / max(1, checked), 6),
                       "near_rate": 0.0},
        "invariants": {
            "terminal_rate": 1.0 if invalid == 0 else 0.0,
            "failure_fraction": round(failures / max(1, checked), 4),
            "recovery_fraction": round(recoveries / max(1, failures), 4) if failures else None,
            "plan_steps_per_task": round(steps / max(1, checked), 2),
            "history_per_task": round(hist / max(1, checked), 2),
            "verdicts": {k: round(v / max(1, sum(verdicts.values())), 4)
                         for k, v in sorted(verdicts.items())},
        },
    }


# ----------------------------------------------------------------- ui shards
def ui_shard_stats(dirpath: str, manifest: dict, max_image_shards: int = 4) -> dict:
    from .generators.ui_seed import dhash, image_entropy

    shards = manifest.get("shards", [])
    label_hist: np.ndarray | None = None
    role_hist = action_hist = theme_hist = None
    heat_hist = np.zeros(1024, dtype=np.int64)
    offsets = []
    task_pad = task_len = 0
    n_total = 0
    digests: list[bytes] = []
    near: dict[int, int] = {}
    entropies: list[float] = []
    black = 0
    pixels = 0

    for si, sh in enumerate(shards):
        p = os.path.join(dirpath, sh["name"])
        z = np.load(p, allow_pickle=False)
        n = int(z["images"].shape[0])
        n_total += n
        labels = z["labels"] if "labels" in z.files else None
        if labels is not None:
            lh = np.bincount(labels, minlength=24)
            label_hist = lh if label_hist is None else label_hist + lh
        rh = np.bincount(z["roles"], minlength=7)
        role_hist = rh if role_hist is None else role_hist + rh
        ah = np.bincount(z["actions"], minlength=4)
        action_hist = ah if action_hist is None else action_hist + ah
        th = np.bincount(z["themes"], minlength=3)
        theme_hist = th if theme_hist is None else theme_hist + th

        heat = z["heat"].astype(np.float32)
        arg = heat.reshape(n, -1).argmax(axis=1)
        heat_hist += np.bincount(arg, minlength=1024)
        offsets.append(z["offsets"].astype(np.float32))
        tasks = z["tasks"]
        task_pad += int(np.sum(tasks == 0))
        task_len += int(tasks.size)

        if si < max_image_shards:
            # bind once: on a compressed npz every z["images"] call would
            # decompress the whole 245 MB member again
            images = z["images"]
            take = np.linspace(0, n - 1, min(500, n)).astype(int)
            for i in take:
                img = images[i]
                digests.append(hashlib.blake2b(img.tobytes(), digest_size=16).digest())
                h = dhash(img)
                near[h] = near.get(h, 0) + 1
                entropies.append(image_entropy(img))
                pixels += img.size
                black += int(np.sum(img.max(axis=2) < 16))
            del images
        z.close()

    offs = np.concatenate(offsets) if offsets else np.zeros((1, 2), np.float32)
    cell_counts = heat_hist
    off_abs = np.abs(offs)
    coord = {
        "cells_used": int(np.count_nonzero(cell_counts)),
        "max_cell_share": round(float(cell_counts.max()) / max(1, cell_counts.sum()), 5),
        "entropy": round(checks.entropy(cell_counts.tolist()), 4),
        "entropy_ratio": round(checks.entropy_ratio(cell_counts.tolist()), 4),
        "offset_mean": [round(float(offs[:, 0].mean()), 4), round(float(offs[:, 1].mean()), 4)],
        "offset_std": [round(float(offs[:, 0].std()), 4), round(float(offs[:, 1].std()), 4)],
        "offset_mean_abs": round(float(off_abs.mean()), 4),
        "offset_clipped": round(float(np.mean(np.any(np.abs(offs) >= 0.999, axis=1))), 5),
    }
    dup_exact = round(1.0 - len(set(digests)) / len(digests), 6) if digests else 0.0
    dup_near = round(sum(c - 1 for c in near.values()) / len(digests), 6) if digests else 0.0
    n_pix = max(1, pixels)

    return {
        "kind": "ui_shards",
        "dataset": manifest.get("dataset", os.path.basename(dirpath)),
        "n": n_total,
        "shards": len(shards),
        "label_balance": _hist_stats(label_hist if label_hist is not None else []),
        "role_balance": _hist_stats(role_hist if role_hist is not None else []),
        "action_balance": _hist_stats(action_hist if action_hist is not None else []),
        "theme_balance": _hist_stats(theme_hist if theme_hist is not None else []),
        "coordinate": coord,
        "duplicates": {"sampled": len(digests), "exact_rate": dup_exact, "near_rate": dup_near},
        "image_entropy": {
            "mean": round(float(np.mean(entropies)) if entropies else 0.0, 4),
            "std": round(float(np.std(entropies)) if entropies else 0.0, 4),
            "black_share": round(black / n_pix, 4),
            "sampled": len(entropies),
        },
        "instruction": {
            "padding_share": round(task_pad / max(1, task_len), 4),
            "mean_used_chars": round((task_len - task_pad) / max(1, n_total), 1),
        },
    }


# -------------------------------------------------------------------- directory
def stats_dir(root: str, tokenizer: str | None = None, sample_docs: int = 5000,
              max_image_shards: int = 4) -> dict:
    tok_path = find_tokenizer(root, tokenizer)
    tok = _load_tok(tok_path)
    report: dict = {"dir": os.path.abspath(root), "tokenizer": tok_path,
                    "vocab_size": tok.vocab_size, "datasets": {}}

    for d, mf in M.dataset_dirs(root):
        name = mf.get("dataset") or os.path.basename(d)
        if mf.get("kind") == "token_streams":
            per_file = []
            cache: dict[str, object] = {}
            for sh in mf.get("shards", []):
                p = os.path.join(d, sh["name"])
                stok = tok
                if sh.get("tokenizer"):
                    tpath = sh["tokenizer"]
                    if tpath not in cache:
                        cache[tpath] = _load_tok(os.path.join(d, tpath))
                    stok = cache[tpath]
                st = token_stream_stats(p, stok, sample_docs=sample_docs,
                                        scan_episodes=(name == "drive"))
                st["tokenizer"] = sh.get("tokenizer") or os.path.basename(tok_path)
                st["role"] = sh.get("role", "primary")
                per_file.append(st)
            agg = _merge_streams(per_file)
            agg.update({"dataset": name, "profile": mf.get("profile"), "seed": mf.get("seed"),
                        "content": mf.get("content", {})})
            agg["ranges"] = {"ok": not checks.check_dataset(agg),
                             "violations": checks.check_dataset(agg),
                             "overrides": checks.DATASET_OVERRIDES.get(name, {})}
            report["datasets"][name] = agg
        elif mf.get("kind") == "jsonl":
            st = jsonl_stats(d, mf, name)
            st.update({"profile": mf.get("profile"), "seed": mf.get("seed")})
            st["ranges"] = {"ok": not checks.check_dataset(st),
                            "violations": checks.check_dataset(st),
                            "overrides": checks.DATASET_OVERRIDES.get(name, {})}
            report["datasets"][name] = st
        elif mf.get("kind") == "ui_shards":
            st = ui_shard_stats(d, mf, max_image_shards=max_image_shards)
            st.update({"profile": mf.get("profile"), "seed": mf.get("seed")})
            st["ranges"] = {"ok": not checks.check_dataset(st),
                            "violations": checks.check_dataset(st),
                            "overrides": checks.DATASET_OVERRIDES.get(name, {})}
            report["datasets"][name] = st

    ok = all(d["ranges"]["ok"] for d in report["datasets"].values()) if report["datasets"] else False
    report["ok"] = ok
    report["profile"] = next((d.get("profile") for d in report["datasets"].values() if d.get("profile")), None)
    report["seed"] = next((d.get("seed") for d in report["datasets"].values() if d.get("seed") is not None), None)
    return report


def _merge_streams(parts: list[dict]) -> dict:
    """Merge per-file stats. Alternate tokenizer encodings of the same documents
    are reported separately instead of double-counting the corpus."""
    prim = [p for p in parts if p.get("role", "primary") in ("primary", "instruction_pairs")]
    alt = [p for p in parts if p.get("role") == "alt_tokenizer"]
    toks = sum(p["tokens"] for p in prim)
    docs = sum(p["docs"] for p in prim)
    bal = {k: 0 for k in parts[0]["balance"]} if parts else {}
    for p in parts:
        for k, v in p["balance"].items():
            if k == "_share":
                continue
            bal[k] += v
    tot = sum(v for k, v in bal.items()) or 1
    bal["_share"] = {k: round(v / tot, 5) for k, v in bal.items()}
    vc = max((p["vocab_used"] for p in parts), default=0)
    out = {
        "kind": "token_streams",
        "n": docs,
        "units": "docs",
        "tokens": toks,
        "docs": docs,
        "vocab_used": vc,
        "balance": bal,
        "top_tokens": max(parts, key=lambda p: p["tokens"])["top_tokens"] if parts else [],
        "duplicates": {
            "sampled": sum(p["duplicates"]["sampled"] for p in parts),
            "exact_rate": round(max((p["duplicates"]["exact_rate"] for p in parts), default=0.0), 6),
            "near_rate": round(max((p["duplicates"]["near_rate"] for p in parts), default=0.0), 6),
        },
        "doc_len": max(parts, key=lambda p: p["tokens"])["doc_len"] if parts else {},
        "files": [p["path"] for p in parts],
        "alt_encodings": [{"file": p["path"], "tokens": p["tokens"], "tokenizer": p.get("tokenizer")}
                          for p in alt],
    }
    inv = [p["invariants"] for p in parts if "invariants" in p]
    if inv:
        out["invariants"] = _merge_invariants(inv)
    return out


def _merge_invariants(invs: list[dict]) -> dict:
    eps = sum(i["episodes"] for i in invs)
    merged = {
        "episodes": eps,
        "tokens": sum(i["tokens"] for i in invs),
        "tokens_per_episode": round(sum(i["tokens_per_episode"] * i["episodes"] for i in invs) / max(1, eps), 1),
        "terminal_rate": round(sum(i["terminal_rate"] * i["episodes"] for i in invs) / max(1, eps), 6),
        "retry_after_failure_rate": round(
            sum(i["retry_after_failure_rate"] * max(1, i["retries"]) for i in invs)
            / max(1, sum(max(1, i["retries"]) for i in invs)), 6),
        "failure_fraction": round(sum(i["failure_fraction"] * i["episodes"] for i in invs) / max(1, eps), 6),
        "mean_retries": round(sum(i["mean_retries"] * i["episodes"] for i in invs) / max(1, eps), 3),
        "mix": {k: round(sum(i["mix"][k] * i["episodes"] for i in invs) / max(1, eps), 4)
                for k in ("ui", "code", "plan")},
        "terminal_missing": sum(i.get("terminal_missing", 0) for i in invs),
        "terminal_over": sum(i.get("terminal_over", 0) for i in invs),
        "bad_retries": sum(i.get("bad_retries", 0) for i in invs),
        "terminal_mix": {k: round(sum(i["terminal_mix"].get(k, 0) * i["episodes"] for i in invs)
                                  / max(1, eps), 4)
                         for k in ("success", "success-partial", "budget-exhausted", "failed")},
        "failures": {"terminal": [], "retry_gap": []},
    }
    merged["violations"] = drive_rules.check(merged)
    return merged


# ----------------------------------------------------------------------- print
def json_view(report: dict) -> dict:
    """Flat, consumer-friendly view: one top-level key per dataset + ``_meta``."""
    out = {"_meta": {k: report[k] for k in ("dir", "tokenizer", "vocab_size", "ok", "profile", "seed")}}
    out.update(report["datasets"])
    return out


def render(report: dict) -> str:
    lines = []
    lines.append(f"== dataforge stats ==  dir={report['dir']}  profile={report.get('profile')}  "
                 f"seed={report.get('seed')}  vocab={report['vocab_size']}")
    head = f"{'dataset':<14}{'kind':<13}{'n':>14}{'tokens':>10}{'dup%':>8}  {'balance':<32}{'ranges'}"
    lines.append(head)
    lines.append("-" * len(head))
    for name, d in report["datasets"].items():
        if d["kind"] == "token_streams":
            n = f"{d['docs']:,} {d['units']}"
            toks = f"{d['tokens']/1e6:.2f}M"
            sh = d["balance"]["_share"]
            bal = f"a/d/s {sh['alpha']:.2f}/{sh['digit']:.2f}/{sh['space']:.2f} sp {sh['special']:.2f}"
        elif d["kind"] == "ui_shards":
            n = f"{d['n']:,} samples"
            toks = "-"
            lb = d["label_balance"]
            bal = f"label H/Hmax {lb['entropy_ratio']:.2f} ({lb['n_labels']} cls)"
        else:  # jsonl tasks
            n = f"{d['n']:,} {d.get('units', 'tasks')}"
            toks = "-"
            inv = d["invariants"]
            bal = f"failure {100*inv['failure_fraction']:.0f}% recov {100*(inv['recovery_fraction'] or 0):.0f}%"
        dup = f"{100*d['duplicates']['exact_rate']:.2f}"
        verdict = "OK" if d["ranges"]["ok"] else "FAIL"
        lines.append(f"{name:<14}{d['kind']:<13}{n:>14}{toks:>10}{dup:>8}  {bal:<32}{verdict}")

    ui = [d for d in report["datasets"].values() if d["kind"] == "ui_shards"]
    st = [d for d in report["datasets"].values() if d["kind"] == "token_streams"]
    lines.append("")
    if st:
        worst = max(report["datasets"].items(), key=lambda kv: kv[1]["duplicates"]["exact_rate"]
                    if kv[1]["kind"] == "token_streams" else -1)
        sampled = sum(d["duplicates"]["sampled"] for d in st)
        lines.append(f"duplicates      exact {100*worst[1]['duplicates']['exact_rate']:.2f}% worst "
                     f"({worst[0]}), near {100*max(d['duplicates']['near_rate'] for d in st):.2f}%, "
                     f"sampled {sampled:,} documents")
        lens = report["datasets"][worst[0]]["doc_len"]
        lines.append(f"tokens          {sum(d['tokens'] for d in st)/1e6:.2f}M over "
                     f"{sum(d['docs'] for d in st):,} documents; worst doc_len "
                     f"mean {lens['mean']:.0f} p99 {lens['p99']}")
    for name, d in report["datasets"].items():
        if "invariants" in d and d["kind"] != "jsonl":
            inv = d["invariants"]
            mix = inv["mix"]
            lines.append(f"drive invariants  terminal_rate={inv['terminal_rate']:.3f}  "
                         f"retry_after_failure={inv['retry_after_failure_rate']:.3f}  "
                         f"failure_frac={inv['failure_fraction']:.3f}  retry/ep={inv['mean_retries']:.2f}  "
                         f"mix ui/code/plan {mix['ui']:.2f}/{mix['code']:.2f}/{mix['plan']:.2f}  "
                         f"terminals ok/partial/exhausted/failed "
                         f"{inv['terminal_mix']['success']:.2f}/"
                         f"{inv['terminal_mix']['success-partial']:.2f}/"
                         f"{inv['terminal_mix']['budget-exhausted']:.2f}/"
                         f"{inv['terminal_mix'].get('failed', 0):.2f}  "
                         f"{'OK' if not inv['violations'] else 'FAIL'}")
    if ui:
        co = ui[0]["coordinate"]
        e = ui[0]["image_entropy"]
        lb = ui[0]["label_balance"]
        du = ui[0]["duplicates"]
        lines.append(f"label balance   H/Hmax {lb['entropy_ratio']:.3f} over {lb['n_labels']} classes, "
                     f"min share {100*lb['min_share']:.2f}% (uniform {100/lb['n_labels']:.2f}%), "
                     f"action H/Hmax {ui[0]['action_balance']['entropy_ratio']:.2f}")
        lines.append(f"coordinate      heat argmax H/Hmax {co['entropy_ratio']:.2f}, max cell "
                     f"{100*co['max_cell_share']:.1f}%, cells {co['cells_used']}/1024, "
                     f"offset |m| {co['offset_mean_abs']:.2f}, clipped {100*co['offset_clipped']:.1f}%")
        lines.append(f"entropy         {e['mean']:.2f}±{e['std']:.2f} bits over {e['sampled']} images, "
                     f"black share {100*e['black_share']:.1f}%")
        lines.append(f"duplicates img  exact {100*du['exact_rate']:.2f}%, near {100*du['near_rate']:.2f}%, "
                     f"{du['sampled']:,} sampled")
        lines.append(f"action balance  " + " ".join(
            f"{n} {100*c/max(1,sum(ui[0]['action_balance']['counts'])):.1f}%"
            for n, c in zip(["click", "type", "toggle", "drag"], ui[0]["action_balance"]["counts"]))
            + f"   themes " + " ".join(
            f"{n} {100*c/max(1,sum(ui[0]['theme_balance']['counts'])):.0f}%"
            for n, c in zip(["light", "dark", "contrast"], ui[0]["theme_balance"]["counts"])))

    for name, d in report["datasets"].items():
        if d["kind"] == "jsonl":
            inv = d["invariants"]
            lines.append(f"trajectories    {d['n']:,} tasks, checked {d['checked']:,}, invalid {d['invalid']}, "
                         f"failure {100*inv['failure_fraction']:.1f}%, "
                         f"recovery {100*(inv['recovery_fraction'] or 0):.1f}%, "
                         f"plan {inv['plan_steps_per_task']} steps, history {inv['history_per_task']}")
            lines.append(f"verdicts        " + " ".join(
                f"{k} {100*v:.0f}%" for k, v in inv["verdicts"].items()))
        c = d.get("content") or {}
        if name == "instructions" and c:
            fam = c.get("family", {})
            var = c.get("variant", {})
            lines.append(f"instructions    {c.get('unique',0):,} unique, reject {100*c.get('reject_rate',0):.2f}%, "
                         f"mean {c.get('chars_mean',0)} chars, multi {100*fam.get('multi',0):.1f}% "
                         f"relative {100*fam.get('relative',0):.1f}% sequence {100*fam.get('sequence',0):.1f}%")
            lines.append(f"surface forms   " + " ".join(f"{k} {100*v:.1f}%" for k, v in var.items()))
    ov = {n: d["ranges"]["overrides"] for n, d in report["datasets"].items() if d["ranges"].get("overrides")}
    if ov:
        lines.append("relaxed bounds  " + "  ".join(
            f"{n}: " + ",".join(f"{k}>{v}" for k, v in o.items()) for n, o in ov.items()))
    viol = [(n, d["ranges"]["violations"]) for n, d in report["datasets"].items() if d["ranges"]["violations"]]
    lines.append("")
    if viol:
        for n, v in viol:
            for item in v:
                lines.append(f"FAIL {n}: {item}")
    else:
        lines.append("ranges          all within bounds  OK")
    return "\n".join(lines)
