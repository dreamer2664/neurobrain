"""Verify a forge directory: hashes first, semantics second.

1. every shard listed in a manifest exists and its sha256 matches
2. token streams are valid uint16 LE and the drive invariants hold **when the
   bytes on disk are re-scanned** (not the counters the generator wrote)
3. UI shards carry the exact keys, dtypes and shapes of specs/data_formats.md

A single failure prints ``FAIL <dataset>: ...`` and exits 1.
"""

from __future__ import annotations

import os

import numpy as np

from . import drive_rules, manifest as M, tokenscan
from .stats import _load_tok, find_tokenizer

SPEC_SHAPES = {
    "images": (np.uint8, (None, 128, 128, 3)),
    "tasks": (np.int64, (None, 48)),
    "heat": (np.float16, (None, 32, 32)),
    "offsets": (np.float32, (None, 2)),
    "actions": (np.int64, (None,)),
}


def _match(shape, spec_shape) -> bool:
    if len(shape) != len(spec_shape):
        return False
    return all(s is None or s == t for s, t in zip(spec_shape, shape))


def verify_dir(root: str, tokenizer: str | None = None) -> tuple[bool, list[str], dict]:
    problems: list[str] = []
    summary: dict = {"dir": os.path.abspath(root), "datasets": {}}

    dirs = list(M.dataset_dirs(root))
    if not dirs:
        return False, [f"no dataset manifests under {root}"], summary

    tok_path = find_tokenizer(root, tokenizer)
    tok = _load_tok(tok_path)

    for d, mf in dirs:
        name = mf.get("dataset") or os.path.basename(d)
        info = {"shards": 0, "bytes": 0, "hashes_ok": True}
        for sh in list(mf.get("shards", [])) + list(mf.get("records", []) or []):
            p = os.path.join(d, sh["name"])
            info["shards"] += 1
            if not os.path.isfile(p):
                problems.append(f"FAIL {name}: missing shard {sh['name']}")
                info["hashes_ok"] = False
                continue
            got = M.sha256_file(p)
            size = os.path.getsize(p)
            info["bytes"] += size
            if got != sh["sha256"]:
                problems.append(f"FAIL {name}: sha256 mismatch on {sh['name']} "
                                f"(manifest {sh['sha256'][:12]}.. got {got[:12]}..)")
                info["hashes_ok"] = False
            if "bytes" in sh and size != sh["bytes"]:
                problems.append(f"FAIL {name}: size mismatch on {sh['name']} "
                                f"(manifest {sh['bytes']} got {size})")

        if mf.get("kind") == "jsonl":
            problems += _verify_jsonl(d, name, mf, info)
        elif mf.get("kind") == "token_streams":
            problems += _verify_streams(d, name, mf, tok, info)
        elif mf.get("kind") == "ui_shards":
            problems += _verify_shards(d, name, mf, info)
        summary["datasets"][name] = info

    summary["ok"] = not problems
    return not problems, problems, summary


def _verify_streams(d: str, name: str, mf: dict, tok, info: dict) -> list[str]:
    problems = []
    scans = []
    for sh in mf.get("shards", []):
        p = os.path.join(d, sh["name"])
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        if size % 2:
            problems.append(f"FAIL {name}: {sh['name']} size {size} not divisible by 2 (uint16)")
            continue
        arr = np.memmap(p, dtype="<u2", mode="r")
        if name == "drive":
            scan = tokenscan.scan(arr, tok)
            bad = drive_rules.check(scan)
            for b in bad:
                problems.append(f"FAIL {name}: {sh['name']} invariant {b}")
            # per-shard numbers beat the dataset aggregate: a 60-episode val
            # shard has its own failure fraction, and comparing it to the
            # aggregate (or vice versa) would be a false alarm
            declared = sh.get("invariants") or (
                (mf.get("content") or {}).get("invariants", {})
                if len(mf.get("shards", [])) == 1 else {})
            for k in ("terminal_rate", "retry_after_failure_rate", "failure_fraction"):
                if k in declared and abs(float(declared[k]) - float(scan[k])) > 1e-6:
                    problems.append(f"FAIL {name}: manifest {k}={declared[k]} but bytes say {scan[k]}")
            scans.append(scan)
        del arr
    if scans:
        from .stats import _merge_invariants
        merged = _merge_invariants(scans)
        info["invariants"] = {k: merged[k] for k in
                              ("episodes", "terminal_rate", "retry_after_failure_rate",
                               "failure_fraction", "mix", "mean_retries")}
    return problems


def _verify_jsonl(d: str, name: str, mf: dict, info: dict, sample_per_shard: int = 200) -> list[str]:
    import gzip
    import json as _json
    from neurobrain.trajectories import has_failure, has_recovery, validate_trajectory
    problems = []
    checked = invalid = fails = recs_with_fail = 0
    for sh in mf.get("shards", []):
        p = os.path.join(d, sh["name"])
        if not os.path.isfile(p):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = _json.loads(line)
                errs = validate_trajectory(rec)
                checked += 1
                if errs:
                    invalid += 1
                    if invalid <= 3:
                        problems.append(f"FAIL {name}: {sh['name']} invalid record "
                                        f"{rec.get('id')}: {errs[:2]}")
                if has_failure(rec):
                    recs_with_fail += 1
                    if has_recovery(rec):
                        fails += 1
                if i >= sample_per_shard:
                    break
    if checked and recs_with_fail / checked < 0.5:
        problems.append(f"FAIL {name}: failure_fraction {recs_with_fail/checked:.3f} < 0.5")
    info["invariants"] = {"checked": checked, "invalid": invalid,
                          "failure_fraction": round(recs_with_fail / max(1, checked), 4),
                          "recovery_fraction": round(fails / max(1, recs_with_fail), 4)}
    return problems


def _verify_shards(d: str, name: str, mf: dict, info: dict) -> list[str]:
    problems = []
    for sh in mf.get("shards", []):
        p = os.path.join(d, sh["name"])
        if not os.path.isfile(p):
            continue
        try:
            header = M.read_npz_keys(p)
        except Exception as e:  # noqa: BLE001
            problems.append(f"FAIL {name}: {sh['name']} unreadable npz ({e})")
            continue
        for key, (dtype, shape) in SPEC_SHAPES.items():
            if key not in header:
                problems.append(f"FAIL {name}: {sh['name']} missing array {key!r}")
                continue
            got_shape, got_dtype = header[key]
            if np.dtype(got_dtype) != np.dtype(dtype):
                problems.append(f"FAIL {name}: {sh['name']} {key} dtype {got_dtype} != {dtype}")
            if not _match(got_shape, shape):
                problems.append(f"FAIL {name}: {sh['name']} {key} shape {got_shape} != {shape}")
        ns = {key: header[key][0][0] for key in header}
        if len(set(ns.values())) != 1:
            problems.append(f"FAIL {name}: {sh['name']} arrays disagree on N: {ns}")
    return problems
